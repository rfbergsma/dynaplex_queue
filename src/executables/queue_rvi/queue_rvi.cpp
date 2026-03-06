#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/queue.h"
#include <vector>
#include <sstream>
#include <variant>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>
#include <cmath>

#include "../../../lib/models/models/queue_mdp/mdp.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <array>
#include <limits>


using namespace DynaPlex;
using DynaPlex::VarGroup;
using DynaPlex::DynaPlexProvider;
using DynaPlex::Models::queue_mdp::MDP;
using DynaPlex::RNG;



double expected_next_fil(int i, double lambda, double gamma) {
	if (i <= 0) return 0.0;
	double denom = lambda + gamma;
	if (denom <= 0.0) return 0.0;
	double alpha = lambda / denom;
	double beta = gamma / denom;

	double EX = 0.0;
	for (int j = 1; j <= i; ++j) {
		double p = std::pow(beta, i - j) * alpha;
		EX += j * p;
	}
	return EX;
}

static void CheckDistribution(
	const std::vector<DynaPlex::Models::queue_mdp::MDP::nextStateProbability>& dist,
	const std::string& label,
	double tol = 1e-12)
{
	if (dist.empty()) {
		std::cout << "[FAIL] " << label << ": distribution is empty\n";
		std::abort();
	}

	double sum = 0.0;
	double minp = 1.0, maxp = 0.0;
	int neg = 0;

	for (const auto& e : dist) {
		sum += e.probability;
		minp = std::min(minp, e.probability);
		maxp = std::max(maxp, e.probability);
		if (e.probability < -tol) neg++;
	}

	std::cout << "[CHECK] " << label
		<< "  n=" << dist.size()
		<< "  sum=" << std::setprecision(17) << sum
		<< "  min=" << minp
		<< "  max=" << maxp
		<< "  neg=" << neg
		<< "\n";

	if (neg > 0 || std::abs(sum - 1.0) > 1e-9) {
		std::cout << "[FAIL] " << label << ": probabilities invalid\n";
		std::abort();
	}
}

struct Transition {
	size_t next_state_idx;
	double probability;
};

struct StateEncoder {
	int M;
	std::vector<std::vector<int>> busy_dims;  // busy_dims[k][j] = servers[k] + 1
	int max_action_counter;

	// Call this once after constructing the MDP
	StateEncoder(const DynaPlex::Models::queue_mdp::MDP& mdp, int M) : M(M) {
		for (size_t k = 0; k < mdp.server_static_info.size(); ++k) {
			busy_dims.push_back({});
			for (size_t j = 0; j < mdp.server_static_info[k].can_serve.size(); ++j)
				busy_dims[k].push_back((int)mdp.server_static_info[k].servers + 1);
		}
		// upper bound: sum of can_serve sizes
		max_action_counter = 0;
		for (auto& si : mdp.server_static_info)
			max_action_counter += (int)si.can_serve.size();
	}

	uint64_t encode(const DynaPlex::Models::queue_mdp::MDP::State& state) const {
		uint64_t key = 0, stride = 1;

		// FIL: clamp to M, shift by 1 so -1 -> 0
		for (int64_t fil : state.queue_manager.FIL_waiting) {
			int64_t v = std::min(fil, (int64_t)M) + 1;
			key += (uint64_t)v * stride;
			stride *= (uint64_t)(M + 2);
		}

		// busy_on
		for (size_t k = 0; k < busy_dims.size(); ++k)
			for (size_t j = 0; j < busy_dims[k].size(); ++j) {
				key += (uint64_t)state.server_manager.busy_on[k][j] * stride;
				stride *= (uint64_t)busy_dims[k][j];
			}

		// cat
		key += (uint64_t)(state.cat == StateCategory::AwaitAction() ? 1 : 0) * stride;
		stride *= 2;

		// action_counter
		key += (uint64_t)state.server_manager.get_action_counter() * stride;

		return key;
	}
};

int main() {

	auto& dp = DynaPlexProvider::Get();


	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	auto path_to_json = dp.FilePath({ "mdp_config_examples", "queue_mdp" }, "mdp_config_0.json");
	//also possible to use a standard configuration file:
	config = VarGroup::LoadFromFile(path_to_json);

	

	// Optional: read id field if present
	if (config.HasKey("id")) {
		std::string id;
		config.Get("id", id);
		std::cout << "MDP id: " << id << "\n";
	}

	try
	{	
		//test 1 (check initial state)
		DynaPlex::Models::queue_mdp::MDP mdp(config);
		auto state = mdp.GetInitialState();
		auto dist = mdp.getNextStateProbability(state, /*action*/ 0); // action ignored in AwaitEvent
		CheckDistribution(dist, "Initial state (AwaitEvent)");

		//test 2

		// Force a waiting job type 0
		{
			state.queue_manager.FIL_waiting[0] = 5;
			state.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
			// if you keep tick rate state-dependent, update it too:
			state.queue_manager.update_total_tick_rate(mdp.tick_rate);

			state.server_manager.generate_actions(state.queue_manager.FIL_waiting);
			state.server_manager.set_action_counter(0);
			state.cat = state.server_manager.action_queue.empty()
				? DynaPlex::StateCategory::AwaitEvent()
				: DynaPlex::StateCategory::AwaitAction();

			if (state.cat == DynaPlex::StateCategory::AwaitAction()) {
				auto d0 = mdp.getNextStateProbability(state, 0);
				CheckDistribution(d0, "AwaitAction action=0");

				auto d1 = mdp.getNextStateProbability(state, 1);
				CheckDistribution(d1, "AwaitAction action=1");
			}
			else {
				std::cout << "No feasible actions created in forced state.\n";
			}
		}
		//test 3
		{
			auto dist = mdp.getNextStateProbability(state, 1);
			for (size_t idx = 0; idx < dist.size(); ++idx) {
				const auto& ns = dist[idx].next_state;

				// Example invariants:
				if (ns.cat == DynaPlex::StateCategory::AwaitAction()) {
					// should have an action queue and counter in range
					if (ns.server_manager.get_action_counter() < 0 ||
						ns.server_manager.get_action_counter() > (int64_t)ns.server_manager.action_queue.size()) {
						std::cout << "[FAIL] counter out of range in next state " << idx << "\n";
						std::abort();
					}
				}

				// Always true: FIL either -1 or >=0
				for (auto fil : ns.queue_manager.FIL_waiting) {
					if (fil < -1) {
						std::cout << "[FAIL] FIL < -1 in next state " << idx << "\n";
						std::abort();
					}
				}
			}
		}

		
		//step 4
		{
			std::mt19937_64 gen(123);
			std::uniform_real_distribution<double> U(0.0, 1.0);

			auto cur = state;
			for (int t = 0; t < 100; ++t) {
				int64_t a = 0;
				if (cur.cat == DynaPlex::StateCategory::AwaitAction()) {
					a = (U(gen) < 0.5) ? 0 : 1;
				}

				auto dist = mdp.getNextStateProbability(cur, a);
				CheckDistribution(dist, "step " + std::to_string(t));

				// sample next state
				double u = U(gen), cum = 0.0;
				size_t pick = 0;
				for (; pick < dist.size(); ++pick) {
					cum += dist[pick].probability;
					if (u <= cum) break;
				}
				if (pick >= dist.size()) pick = dist.size() - 1;
				cur = dist[pick].next_state;
			}


		}

		// ---- Phase 1: BFS transition table ----
		std::vector<int> M_values = { 10, 12, 15, 18, 20 };

		for (int M : M_values)
		{
			//const int M = 12;
			StateEncoder encoder(mdp, M);

			std::unordered_map<uint64_t, size_t> state_index;
			std::vector<DynaPlex::Models::queue_mdp::MDP::State> states;
			std::vector<std::array<std::vector<Transition>, 2>> transitions;
			std::vector<double> immediate_cost;
			std::queue<size_t> bfs;

			auto add_state = [&](DynaPlex::Models::queue_mdp::MDP::State s) -> size_t {
				for (auto& fil : s.queue_manager.FIL_waiting)
					fil = std::min(fil, (int64_t)M);
				uint64_t key = encoder.encode(s);
				auto it = state_index.find(key);
				if (it != state_index.end()) return it->second;
				size_t idx = states.size();
				state_index[key] = idx;
				states.push_back(s);
				transitions.push_back({});

				double cost = 0.0;
				if (s.cat == DynaPlex::StateCategory::AwaitEvent()) {
					for (size_t n = 0; n < (size_t)mdp.n_jobs; ++n)
						if (s.queue_manager.FIL_waiting[n] > mdp.due_times[n])
							cost += mdp.cost_rates[n];
					cost *= mdp.tick_rate / mdp.uniformization_rate;
				}
				immediate_cost.push_back(cost);

				bfs.push(idx);
				return idx;
				};

			add_state(mdp.GetInitialState());

			while (!bfs.empty()) {
				size_t i = bfs.front(); bfs.pop();
				DynaPlex::Models::queue_mdp::MDP::State s = states[i];

				int n_actions = (s.cat == DynaPlex::StateCategory::AwaitAction()) ? 2 : 1;
				for (int a = 0; a < n_actions; ++a) {
					if (a == 1 && !mdp.IsAllowedAction(s, 1)) continue;

					auto dist = mdp.getNextStateProbability(s, (int64_t)a);
					for (const auto& entry : dist) {
						DynaPlex::Models::queue_mdp::MDP::State s_prime = entry.next_state;
						for (auto& fil : s_prime.queue_manager.FIL_waiting)
							fil = std::min(fil, (int64_t)M);
						size_t j = add_state(s_prime);
						transitions[i][a].push_back({ j, entry.probability });
					}
				}
			}

			// stats
			size_t n_await_event = 0, n_await_action = 0;
			size_t total_transitions = 0;
			bool prob_error = false;
			for (size_t i = 0; i < states.size(); ++i) {
				bool is_action = (states[i].cat == DynaPlex::StateCategory::AwaitAction());
				is_action ? ++n_await_action : ++n_await_event;
				for (int a = 0; a < 2; ++a) {
					if (transitions[i][a].empty()) continue;
					double sum = 0.0;
					for (const auto& t : transitions[i][a]) sum += t.probability;
					total_transitions += transitions[i][a].size();
					if (std::abs(sum - 1.0) > 1e-9) {
						std::cout << "[FAIL] state " << i << " action " << a
							<< " prob sum=" << sum << "\n";
						prob_error = true;
					}
				}
			}
			std::cout << "\n--- Transition table (M=" << M << ") ---\n";
			std::cout << "Total states     : " << states.size() << "\n";
			std::cout << "  AwaitEvent     : " << n_await_event << "\n";
			std::cout << "  AwaitAction    : " << n_await_action << "\n";
			std::cout << "Total transitions: " << total_transitions << "\n";
			std::cout << "Prob errors      : " << (prob_error ? "YES" : "none") << "\n";

			// ---- Phase 2: RVI ----
			const size_t ref = 0;
			const double eps = 1e-8;
			const int max_iter = 4000;
			std::vector<double> h(states.size(), 0.0);
			double g_star = 0.0;

			for (int iter = 0; iter < max_iter; ++iter) {
				std::vector<double> h_new(states.size());

				for (size_t i = 0; i < states.size(); ++i) {
					if (states[i].cat == DynaPlex::StateCategory::AwaitEvent()) {
						double val = immediate_cost[i];
						for (const auto& t : transitions[i][0])
							val += t.probability * h[t.next_state_idx];
						h_new[i] = val;
					}
					else {
						double best = std::numeric_limits<double>::infinity();
						for (int a = 0; a < 2; ++a) {
							if (transitions[i][a].empty()) continue;
							double val = 0.0;
							for (const auto& t : transitions[i][a])
								val += t.probability * h[t.next_state_idx];
							best = std::min(best, val);
						}
						h_new[i] = best;
					}
				}

				g_star = h_new[ref];
				for (auto& v : h_new) v -= g_star;

				double delta = 0.0;
				for (size_t i = 0; i < states.size(); ++i)
					delta = std::max(delta, std::abs(h_new[i] - h[i]));

				h = std::move(h_new);

				if (iter % 500 == 0)
					std::cout << "iter " << std::setw(6) << iter
					<< "  g*=" << std::setprecision(10) << g_star
					<< "  delta=" << std::setprecision(6) << delta << "\n";

				if (delta < eps) {
					std::cout << "\nConverged at iter " << iter
						<< "  g* = " << std::setprecision(12) << g_star << "\n";
					break;
				}
			}
		
			std::cout << "  --> M=" << M
				<< "  g* = " << std::setprecision(12) << g_star << "\n";
	}


	}
	catch (const std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
	return 0;
}
