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
MDP::RVISolution MDP::runRVI(int M, int max_iter, bool silent,
	                         int min_policy_stable,
	                         bool post_tick_cost) const {
	if (max_queue_depth > 1 && !silent)
		std::cout << "[RVI] WARNING: max_queue_depth=" << max_queue_depth
		          << " > 1.  RVI operates on FIL projection only.\n"
		          << "              SIL/TIL state is ignored.  Use RL for multi-position problems.\n";

	StateEncoder encoder(*this, M);

	// Action-set size per decision state: 2 in candidate-queue mode
	// (skip/serve), n_jobs+1 in per-event mode (idle / serve type a-1).
	const int A_max = per_event_mode ? (int)n_jobs + 1 : 2;

	std::unordered_map<uint64_t, size_t> state_index;
	std::vector<MDP::State> states;
	std::vector<std::vector<std::vector<Transition>>> transitions;
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
		transitions.push_back(std::vector<std::vector<Transition>>((size_t)A_max));

		// Delegate to ComputeTickCost so reward_type is respected
		// (reward_type=0 -> binary; reward_type=1 -> queue-lateness).
		// reward_type=2/3 (potential-based shaping) deliberately use the
		// UNSHAPED base cost (0 resp. 1): shaping preserves the optimal policy
		// and long-run average, and RVI's state-cost structure cannot
		// represent the action-tied refund terms.
		const int64_t rvi_rtype = (reward_type == 2) ? 0
		                        : (reward_type == 3) ? 1 : reward_type;
		if (s.cat == DynaPlex::StateCategory::AwaitEvent()) {
			// The simulator charges queue cost after applying a tick.  The legacy
			// RVI state-cost used the pre-tick state, which is a different reward
			// whenever cost depends on age.  Keep both paths available for a clean
			// diagnostic A/B comparison.
			if (post_tick_cost) {
				State after_tick = s;
				after_tick.queue_manager.tick();
				immediate_cost.push_back(
					(tick_rate / uniformization_rate) * ComputeTickCost(after_tick, rvi_rtype));
			} else {
				immediate_cost.push_back(
					(tick_rate / uniformization_rate) * ComputeTickCost(s, rvi_rtype));
			}
		} else
			immediate_cost.push_back(0.0);
		bfs_queue.push(idx);
		return idx;
	};

	add_state(GetInitialState());

	while (!bfs_queue.empty()) {
		size_t i = bfs_queue.front(); bfs_queue.pop();
		MDP::State s = states[i];

		// NOTE: RVI deliberately stays on the binary action set {0,1} even when
		// enable_skip_all adds action 2 for RL.  Skip-all is value-degenerate with a
		// chain of single skips, so the {0,1}-optimal policy and g* are exactly
		// optimal in the extended MDP too — the benchmark is unaffected.
		// In per-event mode the action set is {0..n_jobs} (A_max slots).
		int n_actions = (s.cat == DynaPlex::StateCategory::AwaitAction()) ? A_max : 1;
		for (int a = 0; a < n_actions; ++a) {
			if (s.cat == DynaPlex::StateCategory::AwaitAction() &&
			    !IsAllowedAction(s, (int64_t)a)) continue;

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
		for (int a = 0; a < A_max; ++a)
			total_transitions += transitions[i][a].size();
	}
	if (!silent) {
		std::cout << "\n--- Transition table (M=" << M << ") ---\n"
			      << "Total states     : " << states.size() << "\n"
			      << "  AwaitEvent     : " << n_await_event << "\n"
			      << "  AwaitAction    : " << n_await_action << "\n"
			      << "Total transitions: " << total_transitions << "\n";
	}

	// Action decisions are instantaneous in the queue model.  Build a
	// topological ordering of the action-only part of the transition graph so
	// each decision epoch can be solved exactly inside one event-time Bellman
	// update.  Treating AwaitAction states as ordinary discrete-time states would
	// charge one unit of average-cost time per decision, even though no physical
	// time passes.  Since the number of decisions depends on the policy, that
	// changes the objective and can make strategic idling look attractive.
	std::vector<size_t> action_topological_order;
	action_topological_order.reserve(n_await_action);
	std::vector<size_t> action_indegree(states.size(), 0);
	for (size_t i = 0; i < states.size(); ++i) {
		if (states[i].cat != DynaPlex::StateCategory::AwaitAction()) continue;
		for (int a = 0; a < A_max; ++a) {
			for (const auto& t : transitions[i][a]) {
				if (states[t.next_state_idx].cat == DynaPlex::StateCategory::AwaitAction())
					++action_indegree[t.next_state_idx];
			}
		}
	}
	std::queue<size_t> zero_indegree;
	for (size_t i = 0; i < states.size(); ++i)
		if (states[i].cat == DynaPlex::StateCategory::AwaitAction() &&
		    action_indegree[i] == 0)
			zero_indegree.push(i);
	while (!zero_indegree.empty()) {
		const size_t i = zero_indegree.front();
		zero_indegree.pop();
		action_topological_order.push_back(i);
		for (int a = 0; a < A_max; ++a) {
			for (const auto& t : transitions[i][a]) {
				if (states[t.next_state_idx].cat != DynaPlex::StateCategory::AwaitAction())
					continue;
				if (--action_indegree[t.next_state_idx] == 0)
					zero_indegree.push(t.next_state_idx);
			}
		}
	}
	if (action_topological_order.size() != n_await_action)
		throw DynaPlex::Error(
			"queue_mdp RVI: zero-time action transitions contain a cycle; "
			"the decision epoch cannot be embedded into the event-time chain");

	// ---- RVI loop ----
	const size_t ref = 0;
	if (states[ref].cat != DynaPlex::StateCategory::AwaitEvent())
		throw DynaPlex::Error("queue_mdp RVI: the reference state must await a timed event");
	const double eps = 1e-10;
	std::vector<double> h(states.size(), 0.0);
	std::vector<double> action_h(states.size(), 0.0);
	double g_star = 0.0;
	double g_prev = 0.0;
	int g_stable_count = 0;
	std::vector<int16_t> previous_policy(states.size(), -1);
	std::vector<int16_t> current_policy(states.size(), -1);
	int policy_stable_count = 0;
	int final_policy_changes = -1;
	double final_span = std::numeric_limits<double>::infinity();
	int iterations = 0;
	bool stopped_by_span = false;
	bool stopped_by_g_stable = false;

	// Given relative values on timed AwaitEvent states, solve every zero-time
	// decision epoch backwards.  Successor AwaitAction values are already known
	// in reverse topological order; successor AwaitEvent values come from event_h.
	auto solve_action_epochs = [&](const std::vector<double>& event_h,
	                               std::vector<double>& values,
	                               std::vector<int16_t>* policy) {
		for (auto rit = action_topological_order.rbegin();
		     rit != action_topological_order.rend(); ++rit) {
			const size_t i = *rit;
			double best = std::numeric_limits<double>::infinity();
			int16_t best_a = -1;
			for (int a = 0; a < A_max; ++a) {
				if (transitions[i][a].empty()) continue;
				double val = 0.0;
				for (const auto& t : transitions[i][a]) {
					const bool next_is_action =
						states[t.next_state_idx].cat == DynaPlex::StateCategory::AwaitAction();
					val += t.probability *
						(next_is_action ? values[t.next_state_idx] : event_h[t.next_state_idx]);
				}
				if (val < best) {
					best = val;
					best_a = (int16_t)a;
				}
			}
			if (best_a < 0)
				throw DynaPlex::Error("queue_mdp RVI: reachable action state has no allowed transition");
			values[i] = best;
			if (policy != nullptr) (*policy)[i] = best_a;
		}
	};

	for (int iter = 0; iter < max_iter; ++iter) {
		std::vector<double> h_new(states.size());
		int policy_changes = 0;
		solve_action_epochs(h, action_h, &current_policy);

		for (size_t i = 0; i < states.size(); ++i) {
			if (states[i].cat == DynaPlex::StateCategory::AwaitEvent()) {
				double val = immediate_cost[i];
				for (const auto& t : transitions[i][0]) {
					const bool next_is_action =
						states[t.next_state_idx].cat == DynaPlex::StateCategory::AwaitAction();
					val += t.probability *
						(next_is_action ? action_h[t.next_state_idx] : h[t.next_state_idx]);
				}
				h_new[i] = val;
			}
			else if (iter > 0 && current_policy[i] != previous_policy[i])
				++policy_changes;
		}

		g_star = h_new[ref];
		// One RVI step is one uniformized, timed event.  AwaitAction states have
		// duration zero and were eliminated above, so g* is subtracted only from
		// event-state Bellman updates.
		for (size_t i = 0; i < states.size(); ++i)
			if (states[i].cat == DynaPlex::StateCategory::AwaitEvent())
				h_new[i] -= g_star;

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
			if (states[i].cat != DynaPlex::StateCategory::AwaitEvent()) continue;
			const double d = h_new[i] - h[i];
			if (d > max_diff) max_diff = d;
			if (d < min_diff) min_diff = d;
		}
		const double span = max_diff - min_diff;
		final_span = span;
		final_policy_changes = (iter > 0) ? policy_changes : -1;
		iterations = iter + 1;
		if (iter > 0 && policy_changes == 0)
			++policy_stable_count;
		else
			policy_stable_count = 0;
		previous_policy.swap(current_policy);

		std::swap(h, h_new);

		if (!silent && iter % 500 == 0)
			std::cout << "iter " << std::setw(6) << iter
				      << "  g*=" << std::setprecision(10) << g_star
				      << "  span=" << std::setprecision(6) << span
				      << "  policy_changes=" << policy_changes
				      << "  policy_stable=" << policy_stable_count << "\n";

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

		const bool span_converged = span < eps;
		const bool g_fallback_converged = g_stable_count >= 5 &&
			policy_stable_count >= min_policy_stable;
		if (span_converged || g_fallback_converged) {
			stopped_by_span = span_converged;
			stopped_by_g_stable = !span_converged && g_fallback_converged;
			if (!silent)
				std::cout << "\nConverged at iter " << iter
					      << (span_converged ? "  [span]" : "  [g_stable]")
					      << "  g* = " << std::setprecision(12) << g_star
					      << "  span=" << final_span
					      << "  policy_changes=" << final_policy_changes
					      << "  policy_stable=" << policy_stable_count << "\n";
			break;
		}
	}

	// ---- Build action map and gap map from converged h ----
	RVISolution sol;
	sol.g_star = g_star;
	sol.M = M;
	sol.iterations = iterations;
	sol.final_span = final_span;
	sol.final_policy_changes = final_policy_changes;
	sol.policy_stable_count = policy_stable_count;
	sol.stopped_by_span = stopped_by_span;
	sol.stopped_by_g_stable = stopped_by_g_stable;
	sol.reached_max_iterations = !stopped_by_span && !stopped_by_g_stable;
	sol.post_tick_cost = post_tick_cost;
	sol.state_count = states.size();
	sol.transition_count = total_transitions;

	// Align the zero-time continuation values and final greedy policy with the
	// last normalized event-state bias vector before extracting the action map.
	solve_action_epochs(h, action_h, &current_policy);

	for (size_t i = 0; i < states.size(); ++i) {
		if (states[i].cat != DynaPlex::StateCategory::AwaitAction()) continue;

		// Compute Q(s, a) for all actions (A_max slots; unreachable = inf).
		std::vector<double> q((size_t)A_max, std::numeric_limits<double>::infinity());
		for (int a = 0; a < A_max; ++a) {
			if (transitions[i][a].empty()) continue;
			q[a] = 0.0;
			for (const auto& t : transitions[i][a]) {
				const bool next_is_action =
					states[t.next_state_idx].cat == DynaPlex::StateCategory::AwaitAction();
				q[a] += t.probability *
					(next_is_action ? action_h[t.next_state_idx] : h[t.next_state_idx]);
			}
		}

		int64_t best_a = 0;
		for (int a = 1; a < A_max; ++a)
			if (q[a] < q[best_a]) best_a = a;
		uint64_t key = encoder.encode(states[i]);
		sol.action_map[key] = best_a;

		// Store the action-value gap |Q(s,0) - Q(s,1)| whenever both actions
		// are reachable (candidate-queue mode diagnostics only).
		if (!per_event_mode &&
		    q[0] < std::numeric_limits<double>::infinity() &&
		    q[1] < std::numeric_limits<double>::infinity()) {
			sol.gap_map[key] = std::abs(q[0] - q[1]);
			sol.q_map[key]   = { q[0], q[1] };
		}
	}

	return sol;
}

// ---- EvaluateRVIGap: |Q(s,0)-Q(s,1)| for a live state ----
double MDP::EvaluateRVIGap(const RVISolution& sol, const State& state) const {
	if (state.cat != DynaPlex::StateCategory::AwaitAction()) return -1.0;

	// Identical clamping and encoding to EvaluateRVIPolicy.
	StateEncoder enc(*this, sol.M);
	State clamped = state;
	clamped.queue_manager.clamp_fil(sol.M);

	uint64_t key = enc.encode(clamped);
	auto it = sol.gap_map.find(key);
	if (it == sol.gap_map.end()) return -1.0;
	return it->second;
}

// ---- EvaluateRVIQValues: {Q(s,0), Q(s,1)} for a live state ----
std::pair<double,double> MDP::EvaluateRVIQValues(const RVISolution& sol, const State& state) const {
	const std::pair<double,double> missing = { -1.0, -1.0 };
	if (state.cat != DynaPlex::StateCategory::AwaitAction()) return missing;

	StateEncoder enc(*this, sol.M);
	State clamped = state;
	clamped.queue_manager.clamp_fil(sol.M);

	uint64_t key = enc.encode(clamped);
	auto it = sol.q_map.find(key);
	if (it == sol.q_map.end()) return missing;
	return it->second;
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
	if (it == sol.action_map.end()) {
		if (!per_event_mode) return 1;
		// per-event fallback: FIFO choice (oldest feasible FIL; idle if none)
		const Action& cur = state.server_manager.action_queue.at(
			(size_t)state.server_manager.get_action_counter());
		int64_t best = 0, best_fil = -1;
		for (int64_t n = 0; n < n_jobs; ++n) {
			const auto& qn = state.queue_manager.waiting[(size_t)n];
			if (qn.empty()) continue;
			if (!state.server_manager.can_assign_job(cur.server_index, n)) continue;
			if (qn.front() > best_fil) { best_fil = qn.front(); best = n + 1; }
		}
		return best;
	}
	return it->second;
}

} // namespace queue_mdp
} // namespace DynaPlex::Models
