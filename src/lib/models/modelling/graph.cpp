#include <queue>
#include <vector>
#include <functional>
#include <limits>
#include <sstream>
#include "dynaplex/error.h"
#include "dynaplex/modelling/graph.h"



namespace DynaPlex {
	Graph::Edge::Edge(const VarGroup& config) {
		int64_t orig_, dest_;
		config.Get("orig", orig_);
		config.Get("dest", dest_);
		if (orig_<0 || orig_> std::numeric_limits<uint32_t>::max()
			|| dest_<0 || dest_> std::numeric_limits<uint32_t>::max())
			throw DynaPlex::Error("Graph::Edge - orig and dest must be non-negative integers that fit uint32_t.");

		orig = static_cast<uint32_t>(orig_);
		dest = static_cast<uint32_t>(dest_);
		if (orig == dest)
			throw DynaPlex::Error("Graph::Edge - self-connections not allowed (orig=dest)");

		config.GetOrDefault("weight", weight,1.0);
		if (weight <= 0.0)
		{
			throw DynaPlex::Error("Graph::Edge - weight must be strictly positive");
		}
	}

	Graph::Edge::Edge() : orig(0), dest(0), weight(1.0) {
	}

	Graph::Edge::Edge(uint32_t orig, uint32_t dest, WeightType weight) : orig(orig), dest(dest), weight(weight) {
	}



	Graph::Graph(const VarGroup& config) {
		std::string format{};
		config.GetOrDefault("format", format, "edge_list");
		
		if (format == std::string("edge_list"))
		{
			config.Get("edges", edges);
		}
		else
		{
			if (format == std::string("grid"))
			{
				format_is_grid = true;
			
				std::vector<std::string> rows;
				config.Get("rows", rows); // Assuming `config` can retrieve a vector of strings for "rows"

				// The number of rows is simply the size of the `rows` vector
				height = rows.size();
			
				// Assuming all rows are of the same length, so calculate width based on the first row
				// Counting the number of '|' plus one gives us the number of nodes per row
				width = std::count(rows[0].begin(), rows[0].end(), '|') + 1;
			
				std::vector<std::string> nodes;
				// Flatten the rows into a single vector of node strings
				for (const std::string& row : rows) {
					int row_width = std::count(row.begin(), row.end(), '|') + 1;
					if (row_width != width)
						throw DynaPlex::Error("Graph: number of | separators is different for different rows. Only rectangular grids are supported.");

					std::istringstream iss(row);
					std::string node;
					while (std::getline(iss, node, '|')) {
						// Trim spaces from the node string if necessary
						node.erase(std::remove_if(node.begin(), node.end(), isspace), node.end());
						nodes.push_back(node);
					}
				}		
				
				
				for (int64_t row = 0; row < height; ++row) {
					for (int64_t col = 0; col < width; ++col) {
						uint32_t idx = NodeAt(row, col);

						if (idx >= nodes.size()) continue;

						std::string connections = nodes[idx];

						for (char dir : connections) {
							int intDir = static_cast<unsigned char>(dir); // Cast to unsigned and then to int
							std::string loc = "(row,col)=(" + std::to_string(row) + "," + std::to_string(col) + ")";
							switch (intDir) {
							case 'U':
								if (row <= 0) throw DynaPlex::Error("Up (U) arrow at " + loc + " points outside of the grid");
								edges.push_back({ idx, static_cast<uint32_t>(NodeAt(row - 1, col)) });
								break;
							case 'D':
								if (row >= height - 1) throw DynaPlex::Error("Down (D) arrow at " + loc + " points outside of the grid");
								edges.push_back({ idx, static_cast<uint32_t>(NodeAt(row + 1, col)) });
								break;
							case 'L':
								if (col <= 0) throw DynaPlex::Error("Left (L) arrow at " + loc + " points outside of the grid");
								edges.push_back({ idx, static_cast<uint32_t>(NodeAt(row, col - 1)) });
								break;
							case 'R':
								if (col >= width - 1) throw DynaPlex::Error("Right (R) arrow at "+ loc + " points outside of the grid");
								edges.push_back({ idx, static_cast<uint32_t>(NodeAt(row, col + 1)) });
								break;
							case ' ':
								// Do nothing for space character
								break;
							default:
								throw DynaPlex::Error("Unrecognized character in node connections");
							}
						}
					}
				}
			}
			else
				throw DynaPlex::Error("Graph - format '" + format + "' not supported. Only support grid and edge_list format.");
		}
		

		

		std::string type{};
		config.GetOrDefault("type", type, "directed");
		if (type == std::string("undirected"))
		{  //add the opposite of all edges to the edges vector:
			size_t current = edges.size();
			edges.reserve(current * 2);
			for (size_t i = 0; i < current; i++)
			{
				//add the opposite edge for all edges currently in the graph.
				Edge edge{};
				edge.orig = edges[i].dest;
				edge.dest = edges[i].orig;
				edge.weight = edges[i].weight;
				edges.push_back(edge);
			}
		}
		else
			if (type != std::string("directed"))
				throw DynaPlex::Error("Graph:: type " + type + " not supported. Only support directed or undirected.");
	
		bool allow_double_edges{};
		config.GetOrDefault("allow_double_edges", allow_double_edges, false);
		ValidateEdgeVectorAndComputeNumNodes(allow_double_edges);
		PCSP();
	}


	int64_t Graph::NodeAt(int64_t row, int64_t col) const {
		if (!format_is_grid)
			throw DynaPlex::Error("Graph::NodeAt - format is not a grid.");
		if (row < 0 || row >= height)
			throw DynaPlex::Error("Graph::NodeAt - invalid row");

		if (col < 0 || col >= width)
			throw DynaPlex::Error("Graph::NodeAt - invalid col");
		return row * width + col;
	}

	int64_t Graph::Width() const {
		if (!format_is_grid)
			throw DynaPlex::Error("Graph::NodeAt - format is not a grid.");
		return width;
	}
	int64_t Graph::Height() const {
		if (!format_is_grid)
			throw DynaPlex::Error("Graph::NodeAt - format is not a grid.");
		return height;
	}

	Graph::Coords Graph::Coordinates(int64_t node) const	{
		if (!format_is_grid)
			throw DynaPlex::Error("Graph::Coordinates - format is not a grid.");
		if (node < 0 || node >= width * height)
			throw DynaPlex::Error("Graph::Coordinates - node not part of grid.");
		int64_t row = node / width;
		int64_t col = node - row * width;
		return { row,col };
	}


	Graph::Graph() : numNodes(0) {
	}

	bool Graph::ExistsPath(int64_t origin_node, int64_t destination_node) const {
		if (origin_node < 0 || origin_node >= numNodes || destination_node < 0 || destination_node >= numNodes) {
			throw DynaPlex::Error("Invalid node index in Graph::Distance query");
		}
		return distances[origin_node][destination_node] != std::numeric_limits<double>::max();
	}

	Graph::WeightType Graph::Distance(int64_t origin_node, int64_t destination_node) const {
		if (origin_node < 0 || origin_node >= numNodes || destination_node < 0 || destination_node >= numNodes) {
			throw DynaPlex::Error("Invalid node index in Graph::Distance query");
		}
		if (distances[origin_node][destination_node] == std::numeric_limits<double>::max()) {
			throw DynaPlex::Error("No valid path exists in Distance query");
		}
		return distances[origin_node][destination_node];
	}


	std::vector<Graph::Edge> Graph::Path(int64_t origin_node, int64_t destination_node) const {
		if (origin_node < 0 || origin_node >= numNodes || destination_node < 0 || destination_node >= numNodes) {
			throw DynaPlex::Error("Invalid node index in Graph::Path query");
		}
		if (distances[origin_node][destination_node] == std::numeric_limits<double>::max()) {
			throw DynaPlex::Error("No valid path exists in Path query");
		}

		std::vector<Edge> path;
		int64_t currentNode = origin_node;
		while (currentNode != destination_node) {
			int edgeIndex = edgeIndices[currentNode][destination_node];
			if (edgeIndex == -1) {
				throw DynaPlex::Error("Failed to construct path in Path query");
			}
			path.push_back(edges[edgeIndex]);
			currentNode = edges[edgeIndex].dest;

			if (currentNode == destination_node) {
				break;
			}
		}
		return path;
	}



	const Graph::Edge& Graph::NextEdge(int64_t origin_node, int64_t destination_node) const {
		if (origin_node < 0 || origin_node >= numNodes || destination_node < 0 || destination_node >= numNodes) {
			throw DynaPlex::Error("Invalid node index in NextEdge query");
		}
				
		if (distances[origin_node][destination_node] == std::numeric_limits<WeightType>::max()) {
			throw DynaPlex::Error("No valid path for NextEdge query");
		}

		if (origin_node == destination_node)
			throw DynaPlex::Error("origin equals destination in NextEdge query");

		int edgeIndex = edgeIndices[origin_node][destination_node];
		if (edgeIndex == -1 || edgeIndex >= edges.size()) {
			throw DynaPlex::Error("Failed to find next edge in NextEdge query");
		}

		return edges[edgeIndex];
	}
	void Graph::ValidateEdgeVectorAndComputeNumNodes(bool allow_double_edges) {
		for (auto& edge : edges)
		{
			if (edge.dest >= numNodes)
				numNodes = edge.dest + 1;
			if (edge.orig >= numNodes)
				numNodes = edge.orig + 1;
		}

		std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
			return a.orig < b.orig || (a.orig == b.orig && a.dest < b.dest);
			});
		if (!allow_double_edges)
		{	//we don't accept duplicate edges, because they are likely an input error. 

			for (size_t i = 1; i < edges.size(); ++i) {
				if (edges[i].orig == edges[i - 1].orig && edges[i].dest == edges[i - 1].dest) {
					throw DynaPlex::Error("Graph:: Duplicate edge found : " + std::to_string(edges[i].orig) + " -> " + std::to_string(edges[i].dest));
				}
			}
		}
		else
		{  //if explicitly requested, we allow double edges, removing any edges
			//with same origin and destination but with higher weight.

			std::vector<Edge> edges_with_duplicates = std::move(edges);
			edges.clear(); // Clear the original edges vector
			edges.reserve(edges_with_duplicates.size());

			// Iterate through the sorted edges with duplicates
			for (size_t i = 0; i < edges_with_duplicates.size(); ++i) {
				// Check if this edge is a duplicate
				if (!edges.empty() &&
					edges.back().orig == edges_with_duplicates[i].orig &&
					edges.back().dest == edges_with_duplicates[i].dest) {

					// Update the weight if the current edge has a lower weight
					if (edges_with_duplicates[i].weight < edges.back().weight) {
						edges.back().weight = edges_with_duplicates[i].weight;
					}
				}
				else {
					// If it's not a duplicate, add the edge to the vector
					edges.push_back(edges_with_duplicates[i]);
				}
			}
		}
	}

	
	//precomputeshortestpath
	void Graph::PCSP() {
		distances.resize(numNodes, std::vector<WeightType>(numNodes, std::numeric_limits<WeightType>::max()));
		edgeIndices.resize(numNodes, std::vector<int32_t>(numNodes, -1));

		for (int64_t source = 0; source < numNodes; ++source) {
			std::vector<WeightType> minDistance(numNodes, std::numeric_limits<WeightType>::max());
			std::vector<int> previous(numNodes, -1);

			minDistance[source] = 0;
			using QueueElement = std::pair<WeightType, int64_t>;
			std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> activeVertices;
			activeVertices.push({ 0.0, source });

			while (!activeVertices.empty()) {
				int64_t where = activeVertices.top().second;
				WeightType dist = activeVertices.top().first;
				activeVertices.pop();

				if (dist > minDistance[where]) continue;

				for (auto& edge : edges) {
					if (edge.orig == where) {
						if (minDistance[edge.dest] > minDistance[where] + edge.weight) {
							minDistance[edge.dest] = minDistance[where] + edge.weight;
							previous[edge.dest] = where;
							activeVertices.push({ minDistance[edge.dest], edge.dest });
						}
					}
				}
			}

			for (int64_t dest = 0; dest < numNodes; ++dest) {
				distances[source][dest] = minDistance[dest];
				// Backtrack to find the edge index
				int64_t current = dest;
				while (current != source && previous[current] != -1) {
					int64_t pred = previous[current];
					// Find the edge index from pred to current
					for (size_t i = 0; i < edges.size(); ++i) {
						if (edges[i].orig == pred && edges[i].dest == current) {
							edgeIndices[source][dest] = i;
							break;
						}
					}
					current = pred;
				}
			}
		}
	}
}
