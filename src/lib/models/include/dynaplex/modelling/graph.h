#pragma once
#include <vector>
#include "dynaplex/vargroup.h"

namespace DynaPlex {

    class Graph {
    public:
        using WeightType = double;
        class Edge {
        public:
            explicit Edge(const VarGroup& config); // Constructor using VarGroup config
            Edge(); // Default constructor
            Edge(uint32_t orig, uint32_t dest, WeightType weight = static_cast<WeightType>(1));

            uint32_t orig, dest;
            WeightType weight;            
        };

        struct Coords {
            int64_t row;
            int64_t col;
        };

        explicit Graph(const DynaPlex::VarGroup& config); // Constructor using VarGroup config
        Graph(); // Default constructor

        //for graphs that were created from format "grid": returns the index of the node
        //at a certain row and column. 
        int64_t NodeAt(int64_t row, int64_t col) const;


        Coords Coordinates(int64_t node) const;

        bool ExistsPath(int64_t origin_node, int64_t destination_node) const;
        WeightType Distance(int64_t origin_node, int64_t destination_node) const;
        std::vector<Edge> Path(int64_t origin_node, int64_t destination_node) const;
        const Edge& NextEdge(int64_t origin_node, int64_t destination_node) const;

    private:
        int64_t numNodes{ 0 };
        std::vector<Edge> edges;
        std::vector<std::vector<int32_t>> edgeIndices;
        std::vector<std::vector<WeightType>> distances;

        bool format_is_grid{ false };
        int64_t width, height;
        void ValidateEdgeVectorAndComputeNumNodes(bool allow_double_edges);
        void PCSP();
    };

} // namespace DynaPlex
