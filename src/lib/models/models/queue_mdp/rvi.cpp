#include "mdp.h"
#include <queue>
#include <unordered_map>
#include <array>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace DynaPlex::Models {
namespace queue_mdp {

namespace {

struct Transition {
	size_t next_state_idx;
	double probability;
};

struct StateEncoder {
	int M;
	std::vector<std::vector<int>> busy_dims;  // busy_dims[k][j] = servers[k] + 1

	StateEncoder(const MDP& mdp, int M) : M(M) {
		for (size_t k = 0; k < mdp.server_static_info.size(); ++k) {
			busy_dims.push_back({});
			for (size_t j = 0; j < mdp.server_static_info[k].can_serve.size(); ++j)
				busy_dims[k].push_back((int)mdp.server_static_info[k].servers + 1);
		}
	}

	uint64_t encode(const MDP::State& state) const {
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

		// state category
		key += (uint64_t)(state.cat == DynaPlex::StateCategory::AwaitAction() ? 1 : 0) * stride;
		stride *= 2;

		// action_counter
		key += (uint64_t)state.server_manager.get_action_counter() * stride;

		return key;
	}
};

} // anonymous namespace

// ---- runRVI(int M): BFS + RVI at a fixed truncation level ----
MDP::RVISolution MDP::runRVI(int M) const {
	StateEncoder encoder(*this, M);

	std::unordered_map<uint64_t, size_t> state_index;
	std::vector<MDP::State> states;
	std::vector<std::array<std::vector<Transition>, 2>> transitions;
	std::vector<double> immediate_cost;
	std::queue<size_t> bfs_queue;

	auto add_state = [&](MDP::State s) -> size_t {
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
			for (size_t n = 0; n < (size_t)n_jobs; ++n)
				if (s.queue_manager.FIL_waiting[n] > due_times[n])
					cost += cost_rates[n];
			cost *= tick_rate / uniformization_rate;
		}
		immediate_cost.push_back(cost);
		bfs_queue.push(idx);
		return idx;
	};

	add_state(GetInitialState());

	while (!bfs_queue.empty()) {
		size_t i = bfs_queue.front(); bfs_queue.pop();
		MDP::State s = states[i];

		int n_actions = (s.cat == DynaPlex::StateCategory::AwaitAction()) ? 2 : 1;
		for (int a = 0; a < n_actions; ++a) {
			if (a == 1 && !IsAllowedAction(s, 1)) continue;

			auto dist = getNextStateProbability(s, (int64_t)a);
			for (const auto& entry : dist) {
				MDP::State s_prime = entry.next_state;
				for (auto& fil : s_prime.queue_manager.FIL_waiting)
					fil = std::min(fil, (int64_t)M);
				size_t j = add_state(s_prime);
				transitions[i][a].push_back({ j, entry.probability });
			}
		}
	}

	// Print BFS stats
	size_t n_await_event = 0, n_await_action = 0, total_transitions = 0;
	for (size_t i = 0; i < states.size(); ++i) {
		(states[i].cat == DynaPlex::StateCategory::AwaitAction()) ? ++n_await_action : ++n_await_event;
		for (int a = 0; a < 2; ++a)
			total_transitions += transitions[i][a].size();
	}
	std::cout << "\n--- Transition table (M=" << M << ") ---\n"
		      << "Total states     : " << states.size() << "\n"
		      << "  AwaitEvent     : " << n_await_event << "\n"
		      << "  AwaitAction    : " << n_await_action << "\n"
		      << "Total transitions: " << total_transitions << "\n";

	// ---- RVI loop ----
	const size_t ref = 0;
	const double eps = 1e-10;
	const int max_iter = 10000;
	std::vector<double> h(states.size(), 0.0);
	double g_prev = 0.0, g_star = 0.0;
	int g_stable_count = 0;

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

		if (iter > 0 && g_star > eps && std::abs(g_star - g_prev) < eps)
			++g_stable_count;
		else
			g_stable_count = 0;
		g_prev = g_star;

		if (delta < eps || g_stable_count >= 5) {
			std::cout << "\nConverged at iter " << iter
				      << "  g* = " << std::setprecision(12) << g_star << "\n";
			break;
		}
	}

	return { g_star, M };
}

// ---- runRVI(): auto-select M via heuristic + convergence check ----
MDP::RVISolution MDP::runRVI() const {
	// Traffic-intensity heuristic for initial M
	double max_due_time = *std::max_element(due_times.begin(), due_times.end());
	double total_lambda = 0.0;
	for (double r : arrival_rates) total_lambda += r;
	double total_mu = 0.0;
	for (const auto& si : server_static_info)
		total_mu += si.servers * si.mu_k;
	double rho = (total_mu > 0.0) ? std::min(total_lambda / total_mu, 0.99) : 0.99;
	double mean_excess = tick_rate / (total_mu * (1.0 - rho));
	int M = std::max((int)std::ceil(max_due_time + 3.0 * mean_excess) + 5, 10);

	std::cout << "Initial M guess: " << M << "\n";

	const double M_conv_tol = 1e-4;
	double g_prev_M = -1.0;
	RVISolution sol{ 0.0, M };

	while (true) {
		sol = runRVI(M);
		std::cout << "  --> M=" << M
			      << "  g* = " << std::setprecision(12) << sol.g_star << "\n";

		if (g_prev_M >= 0.0) {
			double rel = std::abs(sol.g_star - g_prev_M) / std::max(g_prev_M, 1e-12);
			std::cout << "  rel change from prev M: " << std::setprecision(4) << rel << "\n";
			if (rel < M_conv_tol) {
				std::cout << "\nM converged. Final g* = " << std::setprecision(12)
					      << sol.g_star << "  (M=" << M << ")\n";
				break;
			}
		}
		g_prev_M = sol.g_star;
		M += 2;
		if (M > 500) { std::cout << "M limit reached.\n"; break; }
	}

	return sol;
}

} // namespace queue_mdp
} // namespace DynaPlex::Models
