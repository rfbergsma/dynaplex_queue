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

		// FIL: clamp to M, shift by 1 so -1 -> 0  (RVI is FIL-projected)
		for (int64_t fil : state.queue_manager.get_FIL_waiting()) {
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

// ---- runRVI(int M, int max_iter): BFS + RVI at a fixed truncation level ----
MDP::RVISolution MDP::runRVI(int M, int max_iter, bool silent) const {
	if (max_queue_depth > 1 && !silent)
		std::cout << "[RVI] WARNING: max_queue_depth=" << max_queue_depth
		          << " > 1.  RVI operates on FIL projection only.\n"
		          << "              SIL/TIL state is ignored.  Use RL for multi-position problems.\n";

	StateEncoder encoder(*this, M);

	std::unordered_map<uint64_t, size_t> state_index;
	std::vector<MDP::State> states;
	std::vector<std::array<std::vector<Transition>, 2>> transitions;
	std::vector<double> immediate_cost;
	std::queue<size_t> bfs_queue;

	auto add_state = [&](MDP::State s) -> size_t {
		s.queue_manager.clamp_fil(M);
		uint64_t key = encoder.encode(s);
		auto it = state_index.find(key);
		if (it != state_index.end()) return it->second;
		size_t idx = states.size();
		state_index[key] = idx;
		states.push_back(s);
		transitions.push_back({});

		// Delegate to GetImmediateCost so reward_type is respected
		// (reward_type=0 -> binary; reward_type=1 -> queue-lateness)
		immediate_cost.push_back(GetImmediateCost(s));
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
				s_prime.queue_manager.clamp_fil(M);
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
	if (!silent) {
		std::cout << "\n--- Transition table (M=" << M << ") ---\n"
			      << "Total states     : " << states.size() << "\n"
			      << "  AwaitEvent     : " << n_await_event << "\n"
			      << "  AwaitAction    : " << n_await_action << "\n"
			      << "Total transitions: " << total_transitions << "\n";
	}

	// ---- RVI loop ----
	const size_t ref = 0;
	const double eps = 1e-10;
	std::vector<double> h(states.size(), 0.0);
	double g_star = 0.0;
	double g_prev = 0.0;
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

		// Span seminorm: max(h_new[i] - h[i]) - min(h_new[i] - h[i]).
		// This is the theoretically correct RVI convergence criterion.
		// Unlike max|h_new - h|, it is not fooled by truncation self-loops
		// that add a near-constant offset to every Bellman residual -- those
		// shift all residuals by the same amount, leaving the span unchanged.
		// It is also scale-invariant: QL reward inflates h-values by ~100x
		// vs. binary reward, but the span converges to zero at the same rate.
		double max_diff = -std::numeric_limits<double>::infinity();
		double min_diff =  std::numeric_limits<double>::infinity();
		for (size_t i = 0; i < states.size(); ++i) {
			const double d = h_new[i] - h[i];
			if (d > max_diff) max_diff = d;
			if (d < min_diff) min_diff = d;
		}
		const double span = max_diff - min_diff;

		std::swap(h, h_new);

		if (!silent && iter % 500 == 0)
			std::cout << "iter " << std::setw(6) << iter
				      << "  g*=" << std::setprecision(10) << g_star
				      << "  span=" << std::setprecision(6) << span << "\n";

		// Primary criterion: span < eps (theoretically correct for ergodic MDPs).
		// Fallback: g_stable_count -- span does NOT converge to zero for truncated
		// MDPs (the self-loop at FIL=M permanently offsets some Bellman residuals),
		// but g* converges reliably and quickly.  Five consecutive stable g*
		// iterations is sufficient in practice.
		if (iter > 0 && g_star > eps && std::abs(g_star - g_prev) < eps)
			++g_stable_count;
		else
			g_stable_count = 0;
		g_prev = g_star;

		if (span < eps || g_stable_count >= 5) {
			if (!silent)
				std::cout << "\nConverged at iter " << iter
					      << (span < eps ? "  [span]" : "  [g_stable]")
					      << "  g* = " << std::setprecision(12) << g_star << "\n";
			break;
		}
	}

	// ---- Build action map from converged h ----
	RVISolution sol;
	sol.g_star = g_star;
	sol.M = M;

	for (size_t i = 0; i < states.size(); ++i) {
		if (states[i].cat != DynaPlex::StateCategory::AwaitAction()) continue;
		double best_val = std::numeric_limits<double>::infinity();
		int64_t best_a = 0;
		for (int a = 0; a < 2; ++a) {
			if (transitions[i][a].empty()) continue;
			double val = 0.0;
			for (const auto& t : transitions[i][a])
				val += t.probability * h[t.next_state_idx];
			if (val < best_val) { best_val = val; best_a = a; }
		}
		sol.action_map[encoder.encode(states[i])] = best_a;
	}

	return sol;
}

// ---- runRVI(double rel_tol): auto-select M via heuristic + convergence check ----
MDP::RVISolution MDP::runRVI(double rel_tol, bool silent) const {
	// Traffic-intensity heuristic for initial M
	double max_due_time = *std::max_element(due_times.begin(), due_times.end());
	double total_lambda = 0.0;
	for (double r : arrival_rates) total_lambda += r;
	double total_mu = 0.0;
	for (const auto& si : server_static_info) {
		double max_mu = *std::max_element(si.mu_kj.begin(), si.mu_kj.end());
		total_mu += si.servers * max_mu;
	}
	double rho = (total_mu > 0.0) ? std::min(total_lambda / total_mu, 0.99) : 0.99;
	double mean_excess = tick_rate / (total_mu * (1.0 - rho));
	int M = std::max((int)std::ceil(max_due_time + 3.0 * mean_excess) + 5, 10);

	if (!silent)
		std::cout << "Initial M guess: " << M << "\n";

	double g_prev_M = -1.0;
	RVISolution sol;

	while (true) {
		sol = runRVI(M, 10000, silent);
		if (!silent)
			std::cout << "  --> M=" << M
				      << "  g* = " << std::setprecision(12) << sol.g_star << "\n";

		if (g_prev_M >= 0.0) {
			double rel = std::abs(sol.g_star - g_prev_M) / std::max(g_prev_M, 1e-12);
			if (!silent)
				std::cout << "  rel change from prev M: " << std::setprecision(4) << rel << "\n";
			if (rel < rel_tol) {
				if (!silent)
					std::cout << "\nM converged. Final g* = " << std::setprecision(12)
						      << sol.g_star << "  (M=" << M << ")\n";
				break;
			}
		}
		g_prev_M = sol.g_star;
		M += 2;
		if (M > 500) {
			if (!silent)
				std::cout << "M limit reached.\n";
			break;
		}
	}

	return sol;
}

// ---- EvaluateRVIPolicy: look up optimal action for a given live state ----
int64_t MDP::EvaluateRVIPolicy(const RVISolution& sol, const State& state) const {
	if (state.cat != DynaPlex::StateCategory::AwaitAction()) return 0;

	StateEncoder enc(*this, sol.M);

	// Clamp FIL to sol.M before encoding (same truncation as during BFS)
	State clamped = state;
	clamped.queue_manager.clamp_fil(sol.M);

	uint64_t key = enc.encode(clamped);
	auto it = sol.action_map.find(key);
	// Fallback: state not in BFS map (e.g. multi-server pool creates action_counter
	// values the BFS never reached).  Default to assign (=1) rather than skip (=0):
	// skipping in an unknown state is worse than FIFO and violates the RVI <= FIFO
	// invariant that Section A checks.
	if (it == sol.action_map.end()) return 1;
	return it->second;
}

} // namespace queue_mdp
} // namespace DynaPlex::Models
