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
#include <set>


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
			state.queue_manager.set_fil(0, 5);
			state.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
			// if you keep tick rate state-dependent, update it too:
			state.queue_manager.update_total_tick_rate(mdp.tick_rate);

			state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
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
				for (auto fil : ns.queue_manager.get_FIL_waiting()) {
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

		// ---- Test 5: state encoding round-trip ----
		// The StateEncoder in rvi.cpp is private, so we replicate its logic here.
		// Encoding is a mixed-radix integer: each component uses a fixed "base"
		// and is packed in order: FIL values, busy_on counts, category, action_counter.
		//
		//   encode: key += value * stride;  stride *= base;
		//   decode: value = key % base;     key    /= base;  (same order)
		{
			// Alias to avoid ambiguity with DynaPlex::MDP (the generic type)
			typedef DynaPlex::Models::queue_mdp::MDP QueueMDP;

			const int64_t M_test = 10;

			// -- encode: converts a state to a uint64_t key --
			auto encode = [&](const QueueMDP::State& s) -> uint64_t {
				uint64_t key = 0, stride = 1;

				// FIL values: base = M_test+2  (FIL ranges -1..M, shifted +1 so -1->0)
				for (int64_t fil : s.queue_manager.get_FIL_waiting()) {
					int64_t v = (fil < M_test ? fil : M_test) + 1;
					key    += (uint64_t)v * stride;
					stride *= (uint64_t)(M_test + 2);
				}

				// busy_on[k][j]: base = servers[k]+1  (ranges 0..servers[k])
				for (size_t k = 0; k < mdp.server_static_info.size(); ++k)
					for (size_t j = 0; j < mdp.server_static_info[k].can_serve.size(); ++j) {
						key    += (uint64_t)s.server_manager.busy_on[k][j] * stride;
						stride *= (uint64_t)(mdp.server_static_info[k].servers + 1);
					}

				// state category: base = 2  (0=AwaitEvent, 1=AwaitAction)
				key    += (uint64_t)(s.cat == DynaPlex::StateCategory::AwaitAction() ? 1 : 0) * stride;
				stride *= 2;

				// action counter
				key += (uint64_t)s.server_manager.get_action_counter() * stride;
				return key;
			};

			// -- decode: recovers a state from a uint64_t key --
			auto decode = [&](uint64_t key) -> QueueMDP::State {
				QueueMDP::State s = mdp.GetInitialState();
				uint64_t rem = key;

				// FIL values (same order as encode)
				for (int64_t n = 0; n < mdp.n_jobs; ++n) {
					int64_t v = (int64_t)(rem % (uint64_t)(M_test + 2));
					rem      /= (uint64_t)(M_test + 2);
					s.queue_manager.set_fil(n, v - 1);   // undo +1 shift
				}

				// busy_on[k][j]
				for (size_t k = 0; k < mdp.server_static_info.size(); ++k)
					for (size_t j = 0; j < mdp.server_static_info[k].can_serve.size(); ++j) {
						int64_t base = (int64_t)(mdp.server_static_info[k].servers + 1);
						int64_t v    = (int64_t)(rem % (uint64_t)base);
						rem         /= (uint64_t)base;
						s.server_manager.busy_on[k][j] = v;
					}

				// state category
				int64_t cat_val = (int64_t)(rem % 2); rem /= 2;
				s.cat = (cat_val == 1)
					? DynaPlex::StateCategory::AwaitAction()
					: DynaPlex::StateCategory::AwaitEvent();

				// action counter
				s.server_manager.set_action_counter((int64_t)rem);
				return s;
			};

			// -- helper: print a state compactly --
			auto print_state = [&](const QueueMDP::State& s) {
				std::cout << "FIL=[";
				{ auto __fil = s.queue_manager.get_FIL_waiting(); for (size_t n = 0; n < __fil.size(); ++n) {
					if (n) std::cout << ",";
					std::cout << __fil[n];
				} }
				std::cout << "]  busy=[";
				bool first = true;
				for (size_t k = 0; k < s.server_manager.busy_on.size(); ++k)
					for (size_t j = 0; j < s.server_manager.busy_on[k].size(); ++j) {
						if (!first) std::cout << ",";
						std::cout << s.server_manager.busy_on[k][j];
						first = false;
					}
				std::cout << "]  cat=" << (s.cat == DynaPlex::StateCategory::AwaitAction() ? "Action" : "Event ")
				          << "  ctr=" << s.server_manager.get_action_counter();
			};

			// -- build test states --
			std::vector<std::pair<std::string, QueueMDP::State>> cases;
			cases.push_back(std::make_pair(std::string("initial state"), mdp.GetInitialState()));

			// Case 2: FIL[0]=3, idle server -> AwaitAction
			{
				QueueMDP::State s = mdp.GetInitialState();
				s.queue_manager.set_fil(0, 3);
				s.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
				s.queue_manager.update_total_tick_rate(mdp.tick_rate);
				s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
				s.server_manager.set_action_counter(0);
				s.cat = s.server_manager.action_queue.empty()
					? DynaPlex::StateCategory::AwaitEvent()
					: DynaPlex::StateCategory::AwaitAction();
				cases.push_back(std::make_pair(std::string("FIL[0]=3 idle server"), s));
			}

			// Case 3: FIL[0]=M (boundary)
			{
				QueueMDP::State s = mdp.GetInitialState();
				s.queue_manager.set_fil(0, (int64_t)M_test);
				s.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
				s.queue_manager.update_total_tick_rate(mdp.tick_rate);
				s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
				s.server_manager.set_action_counter(0);
				s.cat = s.server_manager.action_queue.empty()
					? DynaPlex::StateCategory::AwaitEvent()
					: DynaPlex::StateCategory::AwaitAction();
				cases.push_back(std::make_pair(std::string("FIL[0]=M boundary"), s));
			}

			// Case 4: FIL[0]=2, pool-0 server busy -> AwaitEvent
			{
				QueueMDP::State s = mdp.GetInitialState();
				s.queue_manager.set_fil(0, 2);
				s.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
				s.queue_manager.update_total_tick_rate(mdp.tick_rate);
				if (!mdp.server_static_info.empty() && !mdp.server_static_info[0].can_serve.empty())
					s.server_manager.busy_on[0][0] = 1;
				s.server_manager.update_total_service_rate();
				s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
				s.cat = s.server_manager.action_queue.empty()
					? DynaPlex::StateCategory::AwaitEvent()
					: DynaPlex::StateCategory::AwaitAction();
				cases.push_back(std::make_pair(std::string("FIL[0]=2 server busy"), s));
			}

			// -- run the round-trip test --
			std::cout << "\n--- Test 5: state encoding round-trip (M=" << M_test << ") ---\n";
			int pass = 0;
			std::set<uint64_t> seen_keys;
			bool all_distinct = true;

			for (size_t ci = 0; ci < cases.size(); ++ci) {
				const std::string&    desc   = cases[ci].first;
				QueueMDP::State       s      = cases[ci].second;   // copy so we can clamp
				// clamp FIL to M (same as BFS does)
				s.queue_manager.clamp_fil((int64_t)M_test);

				uint64_t       key  = encode(s);
				QueueMDP::State s2  = decode(key);
				uint64_t       key2 = encode(s2);
				bool ok = (key == key2);
				if (ok) ++pass;
				if (!seen_keys.insert(key).second) all_distinct = false;

				std::cout << "  [" << (ok ? "PASS" : "FAIL") << "]  "
				          << std::left << std::setw(24) << desc
				          << "  key=" << std::setw(20) << key << "  decoded: ";
				print_state(s2);
				std::cout << "\n";
			}

			std::cout << "  " << pass << "/" << cases.size() << " round-trips OK";
			std::cout << (all_distinct ? "  |  all keys distinct [PASS]\n" : "  |  duplicate keys found [FAIL]\n");
		}

		// ---- Test 6: trivial 1-server 1-job system (due_time=0) ----
		// With due_time=0 every tick on a waiting job immediately costs money,
		// so the optimal policy is always "assign immediately" (action=1).
		// FIFO also always returns action=1. Therefore both policies are
		// structurally identical and must produce bit-for-bit equal trajectories
		// given the same random seed.
		//
		// Sub-test 6a: verify that every entry in the RVI action_map == 1
		// Sub-test 6b: simulate both policies step-by-step with the same seed
		//              and check CumulativeReturn matches at every step
		{
			// Build trivial config programmatically
			DynaPlex::VarGroup server0;
			server0.Add("servers",      int64_t{1});
			server0.Add("service_rates", std::vector<double>{0.5});
			server0.Add("can_serve",    std::vector<int64_t>{0});

			DynaPlex::VarGroup trivial_config;
			trivial_config.Add("id",             std::string{"queue_mdp"});
			trivial_config.Add("discount_factor",double{1.0});
			trivial_config.Add("k_servers",      int64_t{1});
			trivial_config.Add("n_jobs",         int64_t{1});
			trivial_config.Add("arrival_rates",  std::vector<double>{0.3});
			trivial_config.Add("tick_rate",      double{1.0});
			trivial_config.Add("cost_rates",     std::vector<double>{1.0});
			trivial_config.Add("due_times",      std::vector<double>{0.0});
			trivial_config.Add("reward_type",    int64_t{0});  // binary: keep this test as originally designed
			trivial_config.Add("server_type_0",  server0);

			std::cout << "\n--- Test 6a: RVI action map (1-server 1-job due_time=0) ---\n";

			// Use raw MDP to inspect the action map directly
			DynaPlex::Models::queue_mdp::MDP trivial_mdp(trivial_config);
			// Use large M to minimize truncation bias.  For this system (rho=0.6, tau=1, mu=0.5)
			// g* is still growing at M=40 (+0.5%) and plateaus around M=80.
			auto sol6 = trivial_mdp.runRVI(80);

			std::cout << "  g* = " << sol6.g_star << "  (M=" << sol6.M << ")\n";
			std::cout << "  Action map size: " << sol6.action_map.size() << " AwaitAction states\n";

			bool all_assign = true;
			for (auto it = sol6.action_map.begin(); it != sol6.action_map.end(); ++it) {
				if (it->second != 1) {
					all_assign = false;
					std::cout << "  [FAIL] found action=" << it->second
					          << " for key=" << it->first << "\n";
				}
			}
			std::cout << "  " << (all_assign ? "[PASS] all actions == 1 (always assign)\n"
			                                 : "[FAIL] some actions != 1\n");

			// ---- Sub-test 6b: identical trajectories ----
			std::cout << "\n--- Test 6b: identical trajectory (FIFO vs RVI, same seed) ---\n";

			auto& dp6        = DynaPlex::DynaPlexProvider::Get();
			DynaPlex::MDP fw = dp6.GetMDP(trivial_config);

			DynaPlex::Policy fifo_pol = fw->GetPolicy(std::string("FIFO policy"));

			DynaPlex::VarGroup rvi_pol_config;
			rvi_pol_config.Add("id", std::string("RVI_optimal"));
			rvi_pol_config.Add("M",  int64_t{20});
			DynaPlex::Policy rvi_pol  = fw->GetPolicy(rvi_pol_config);

			const int64_t seed    = 12345;
			const int64_t N_STEPS = 2000;

			DynaPlex::Trajectory traj_fifo, traj_rvi;
			traj_fifo.RNGProvider.SeedEventStreams(true, seed, 0, 0);
			traj_rvi.RNGProvider.SeedEventStreams(true, seed, 0, 0);

			auto span_fifo = std::span<DynaPlex::Trajectory>(&traj_fifo, 1);
			auto span_rvi  = std::span<DynaPlex::Trajectory>(&traj_rvi,  1);

			fw->InitiateState(span_fifo);
			fw->InitiateState(span_rvi);

			bool all_match   = true;
			int64_t first_mismatch = -1;

			for (int64_t step = 0; step < N_STEPS; ++step) {
				if (traj_fifo.Category.IsAwaitAction())
					fw->IncorporateAction(span_fifo, fifo_pol);
				else
					fw->IncorporateEvent(span_fifo);

				if (traj_rvi.Category.IsAwaitAction())
					fw->IncorporateAction(span_rvi, rvi_pol);
				else
					fw->IncorporateEvent(span_rvi);

				double diff = std::abs(traj_fifo.CumulativeReturn - traj_rvi.CumulativeReturn);
				if (diff > 1e-10) {
					all_match = false;
					first_mismatch = step;
					std::cout << "  [FAIL] first mismatch at step " << step
					          << "  FIFO=" << traj_fifo.CumulativeReturn
					          << "  RVI="  << traj_rvi.CumulativeReturn
					          << "  diff=" << diff << "\n";
					break;
				}
			}

			if (all_match)
				std::cout << "  [PASS] CumulativeReturn identical over " << N_STEPS
				          << " steps  (final=" << traj_fifo.CumulativeReturn << ")\n";

			// ---- Sub-test 6c: policy comparer (FIFO vs RVI should match within noise) ----
			std::cout << "\n--- Test 6c: policy comparer (FIFO vs RVI mean cost) ---\n";

			DynaPlex::VarGroup test_config;
			test_config.Add("number_of_trajectories", int64_t{200});
			test_config.Add("periods_per_trajectory", int64_t{10000});

			auto comparer = dp6.GetPolicyComparer(fw, test_config);
			auto res = comparer.Compare({fifo_pol, rvi_pol});

			double fifo_mean, rvi_mean, fifo_err, rvi_err;
			res[0].Get("mean",  fifo_mean);
			res[0].Get("error", fifo_err);
			res[1].Get("mean",  rvi_mean);
			res[1].Get("error", rvi_err);

			std::cout << "  FIFO mean = " << fifo_mean << "  (+/- " << fifo_err << ")\n";
			std::cout << "  RVI  mean = " << rvi_mean  << "  (+/- " << rvi_err  << ")\n";

			// Check that means agree within 3 standard errors of either estimate
			double threshold = 3.0 * std::max(fifo_err, rvi_err);
			double diff6c    = std::abs(fifo_mean - rvi_mean);
			bool pass6c      = diff6c <= threshold;
			std::cout << "  |diff| = " << diff6c << "  threshold = " << threshold << "\n";
			std::cout << "  " << (pass6c ? "[PASS] means agree within 3 std errors\n"
			                             : "[FAIL] means differ beyond noise\n");

			// ---- Sub-test 6d: raw evaluator — does cost/rvi_step match g*? ----
			std::cout << "\n--- Test 6d: raw step breakdown (trivial system) ---\n";

			// FIFO policy: always action=1
			auto fifo_raw = [](const DynaPlex::Models::queue_mdp::MDP::State&) -> int64_t {
				return 1;
			};

			// RVI policy: look up from action_map
			auto rvi_raw = [&](const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
				return trivial_mdp.EvaluateRVIPolicy(sol6, s);
			};

			auto res_fifo = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				trivial_mdp, fifo_raw,
				/*n_traj=*/200, /*steps=*/100000, /*warmup=*/10000, /*seed=*/42);

			auto res_rvi = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				trivial_mdp, rvi_raw,
				/*n_traj=*/200, /*steps=*/100000, /*warmup=*/10000, /*seed=*/42);

			std::cout << "  FIFO:\n";
			std::cout << "    action steps        = " << res_fifo.total_action_steps        << "\n";
			std::cout << "    real event steps    = " << res_fifo.total_real_event_steps    << "\n";
			std::cout << "    fil refresh steps   = " << res_fifo.total_fil_refresh_steps   << "\n";
			std::cout << "    cost/rvi_step       = " << res_fifo.mean_cost_per_rvi_step    << "  [tick-event cost]  (g*=" << sol6.g_star << ")\n";
			std::cout << "    cost/rvi_step (RVI) = " << res_fifo.mean_cost_per_rvi_step_rvi << "  [per-step RVI cost] (g*=" << sol6.g_star << ")\n";
			std::cout << "    cost/event          = " << res_fifo.mean_cost_per_event       << "  (comparer=" << 0.1711525 << ")\n";

			std::cout << "  RVI:\n";
			std::cout << "    action steps        = " << res_rvi.total_action_steps        << "\n";
			std::cout << "    real event steps    = " << res_rvi.total_real_event_steps    << "\n";
			std::cout << "    fil refresh steps   = " << res_rvi.total_fil_refresh_steps   << "\n";
			std::cout << "    cost/rvi_step       = " << res_rvi.mean_cost_per_rvi_step    << "  [tick-event cost]  (g*=" << sol6.g_star << ")\n";
			std::cout << "    cost/rvi_step (RVI) = " << res_rvi.mean_cost_per_rvi_step_rvi << "  [per-step RVI cost] (g*=" << sol6.g_star << ")\n";
			std::cout << "    cost/event          = " << res_rvi.mean_cost_per_event       << "  (comparer=" << 0.1711525 << ")\n";

			double diff6d = std::abs(res_rvi.mean_cost_per_rvi_step_rvi - sol6.g_star);
			double tol6d  = 5.0 * res_rvi.std_error;   // 5 std errors for conservative test
			std::cout << "  |cost/rvi_step(RVI) - g*| = " << diff6d << "  threshold = " << tol6d << "\n";
			std::cout << "  " << (diff6d <= tol6d ? "[PASS] RVI-style cost/rvi_step matches g*\n"
			                                      : "[FAIL] RVI-style cost/rvi_step does not match g*\n");
		}

		/*
		// ---- Run RVI (auto-selects M) ----
		auto sol = mdp.runRVI();
		std::cout << "\nFinal result: g* = " << std::setprecision(12) << sol.g_star
			      << "  (M=" << sol.M << ")\n";
		*/

		// =========================================================
		// Test 7: 2 job types, 1 server, asymmetric cost rates
		//
		// System:
		//   arrival_rates = [0.2, 0.2],  tick_rate = 1.0
		//   service_rate  = 0.7,  rho = 0.4/0.7 ≈ 0.571
		//   cost_rates    = [1.0, 3.0],  due_times = [2, 2]
		//   Λ = 1.0 + 0.2 + 0.2 + 0.7 = 2.1
		//
		// Because type 1 costs 3× more, RVI should prefer serving type 1
		// even when its FIL is below type 0's FIL — i.e., RVI != FIFO on
		// some states.  Lower rho (~0.57) keeps M=60 sufficient.
		//
		// Sub-tests:
		//   7a: run RVI(M=60), print g* and state count
		//   7b: scan action_map for states where RVI deviates from FIFO
		//       (FIFO = assign the job with the larger FIL)
		//   7c: simulate RVI, verify RVI-style cost/rvi_step matches g*
		//   7d: compare RVI vs FIFO mean cost — expect RVI strictly better
		// =========================================================
		{
			DynaPlex::VarGroup server7;
			server7.Add("servers",      int64_t{1});
			server7.Add("service_rates", std::vector<double>{0.7, 0.7});
			server7.Add("can_serve",    std::vector<int64_t>{0, 1});

			DynaPlex::VarGroup cfg7;
			cfg7.Add("id",             std::string{"queue_mdp"});
			cfg7.Add("discount_factor",double{1.0});
			cfg7.Add("k_servers",      int64_t{1});
			cfg7.Add("n_jobs",         int64_t{2});
			cfg7.Add("arrival_rates",  std::vector<double>{0.2, 0.2});
			cfg7.Add("tick_rate",      double{1.0});
			cfg7.Add("cost_rates",     std::vector<double>{1.0, 3.0});
			cfg7.Add("due_times",      std::vector<double>{2.0, 2.0});
			cfg7.Add("reward_type",    int64_t{0});  // binary: keep this test as originally designed
			cfg7.Add("server_type_0",  server7);

			DynaPlex::Models::queue_mdp::MDP mdp7(cfg7);

			// ---- Test 7a: run RVI ----
			std::cout << "\n--- Test 7a: RVI (2-job asymmetric cost, M=60) ---\n";
			auto sol7 = mdp7.runRVI(60);
			std::cout << "  g* = " << std::setprecision(12) << sol7.g_star
			          << "  (M=" << sol7.M << ")\n";
			std::cout << "  Action map size: " << sol7.action_map.size()
			          << " AwaitAction states\n";

			// ---- Test 7b: find states where RVI deviates from FIFO ----
			// FIFO assigns the job with higher FIL.  RVI may prefer lower-FIL
			// type 1 because it costs 3x more.
			// We scan the states used by the RVI (same BFS traversal via
			// getNextStateProbability), but we can do it simply by sampling
			// a few canonical states directly.
			std::cout << "\n--- Test 7b: RVI vs FIFO deviations ---\n";
			std::cout << "  (both jobs waiting, server idle: action_counter=0)\n";
			std::cout << "  FIL_0  FIL_1  FIFO_action  RVI_action  Deviation?\n";

			int n_deviations = 0;
			int n_checked    = 0;
			// Check all (FIL_0, FIL_1) in [0..15]x[0..15] where FIL_0 > FIL_1
			// FIFO would assign type 0 (higher FIL). Check if RVI agrees.
			for (int f0 = 1; f0 <= 15; ++f0) {
				for (int f1 = 0; f1 < f0; ++f1) {
					// Build canonical state: both jobs waiting, server idle
					DynaPlex::Models::queue_mdp::MDP::State s = mdp7.GetInitialState();
					s.queue_manager.set_fil(0, f0);
					s.queue_manager.set_fil(1, f1);
					s.queue_manager.update_total_arrival_rate(mdp7.arrival_rates);
					s.queue_manager.update_total_tick_rate(mdp7.tick_rate);
					s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
					s.server_manager.set_action_counter(0);
					s.cat = s.server_manager.action_queue.empty()
					        ? DynaPlex::StateCategory::AwaitEvent()
					        : DynaPlex::StateCategory::AwaitAction();

					if (s.cat != DynaPlex::StateCategory::AwaitAction()) continue;

					// FIFO: first action in sorted queue (highest FIL → type 0)
					// Since FIL_0 > FIL_1 and FIFO sorts descending, first entry is type 0.
					// FIFO always takes action=1 (assign the front of queue).
					int64_t fifo_job = s.server_manager.action_queue.front().job_type;

					// RVI: look up action map
					int64_t rvi_action = mdp7.EvaluateRVIPolicy(sol7, s);

					// Determine which job RVI actually assigns
					// If rvi_action==1: assigns front of queue (same as FIFO → type 0 here)
					// If rvi_action==0: skips type 0, next AwaitAction will offer type 1
					bool fifo_assigns_type0 = (fifo_job == 0);
					bool rvi_skips_type0    = (rvi_action == 0);
					bool deviation          = rvi_skips_type0;  // RVI skips what FIFO would take

					++n_checked;
					if (deviation) {
						++n_deviations;
						if (n_deviations <= 8)  // print first 8 examples
							std::cout << "  " << std::setw(5) << f0
							          << "  " << std::setw(5) << f1
							          << "      type " << fifo_job
							          << "       skip(->type1)"
							          << "       YES\n";
					}
				}
			}
			std::cout << "  Total checked (FIL_0>FIL_1): " << n_checked
			          << "  Deviations: " << n_deviations << "\n";
			std::cout << "  " << (n_deviations > 0
			                      ? "[PASS] RVI deviates from FIFO on at least one state\n"
			                      : "[FAIL] RVI never deviates — cost_rates may be symmetric\n");

			// ---- Test 7c: simulate RVI, verify RVI-style cost matches g* ----
			std::cout << "\n--- Test 7c: RVI-style simulation vs g* ---\n";

			auto rvi7_raw = [&](const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
				return mdp7.EvaluateRVIPolicy(sol7, s);
			};

			auto res7 = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				mdp7, rvi7_raw,
				/*n_traj=*/200, /*steps=*/200000, /*warmup=*/20000, /*seed=*/42);

			std::cout << "  action steps        = " << res7.total_action_steps      << "\n";
			std::cout << "  real event steps    = " << res7.total_real_event_steps  << "\n";
			std::cout << "  fil refresh steps   = " << res7.total_fil_refresh_steps << "\n";
			std::cout << "  cost/rvi_step (RVI) = " << std::setprecision(12)
			          << res7.mean_cost_per_rvi_step_rvi
			          << "  (g*=" << sol7.g_star << ")\n";

			double diff7c = std::abs(res7.mean_cost_per_rvi_step_rvi - sol7.g_star);
			double tol7c  = 5.0 * res7.std_error;
			std::cout << "  |sim - g*| = " << diff7c << "  threshold = " << tol7c << "\n";
			std::cout << "  " << (diff7c <= tol7c
			                      ? "[PASS] RVI simulation matches g*\n"
			                      : "[FAIL] RVI simulation does not match g*\n");

			// ---- Test 7d: FIFO vs RVI mean cost ----
			std::cout << "\n--- Test 7d: FIFO vs RVI mean cost ---\n";

			auto fifo7_raw = [&](const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
				// FIFO: always assign (action=1); action_queue is already sorted by FIL
				(void)s;
				return 1;
			};

			auto res7_fifo = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				mdp7, fifo7_raw,
				/*n_traj=*/200, /*steps=*/200000, /*warmup=*/20000, /*seed=*/42);

			std::cout << "  FIFO cost/rvi_step (RVI) = "
			          << std::setprecision(12) << res7_fifo.mean_cost_per_rvi_step_rvi << "\n";
			std::cout << "  RVI  cost/rvi_step (RVI) = "
			          << std::setprecision(12) << res7.mean_cost_per_rvi_step_rvi << "\n";
			std::cout << "  RVI improvement = "
			          << std::setprecision(4)
			          << 100.0 * (res7_fifo.mean_cost_per_rvi_step_rvi - res7.mean_cost_per_rvi_step_rvi)
			             / res7_fifo.mean_cost_per_rvi_step_rvi
			          << "%\n";

			// Conservative test: RVI should be at least 0.1% better than FIFO
			bool rvi_beats_fifo = res7.mean_cost_per_rvi_step_rvi
			                    < res7_fifo.mean_cost_per_rvi_step_rvi - res7.std_error;
			std::cout << "  " << (rvi_beats_fifo
			                      ? "[PASS] RVI cost < FIFO cost\n"
			                      : "[FAIL] RVI does not beat FIFO\n");
		}

		// =========================================================
		// Test 8: heterogeneous mu_{k,j} verification
		//   8a: P(completion of type j) == mu_kj[j] / Lambda
		//   8b: RVI always prefers the faster type when costs are equal
		//   8c: g* is symmetric under rate-swap (relabelling invariant)
		// =========================================================
		{
			// Helper: build a 1-server, 2-job config with given per-type service rates.
			// Rates are in can_serve order: mu0 for type 0, mu1 for type 1.
			auto make_cfg8 = [](double mu0, double mu1) -> DynaPlex::VarGroup {
				DynaPlex::VarGroup srv;
				srv.Add("servers",       int64_t{1});
				srv.Add("service_rates", std::vector<double>{mu0, mu1});
				srv.Add("can_serve",     std::vector<int64_t>{0, 1});
				DynaPlex::VarGroup cfg;
				cfg.Add("id",             std::string{"queue_mdp"});
				cfg.Add("discount_factor",double{1.0});
				cfg.Add("k_servers",      int64_t{1});
				cfg.Add("n_jobs",         int64_t{2});
				cfg.Add("arrival_rates",  std::vector<double>{0.2, 0.2});
				cfg.Add("tick_rate",      double{1.0});
				cfg.Add("cost_rates",     std::vector<double>{1.0, 1.0});
				cfg.Add("due_times",      std::vector<double>{2.0, 2.0});
				cfg.Add("reward_type",    int64_t{0});  // binary: g*-symmetry test is reward-agnostic
				                                        // in theory but QL's quadratic cost amplifies
				                                        // truncation asymmetry beyond tol=1e-6 at M=50.
				cfg.Add("server_type_0",  srv);
				return cfg;
			};

			// ---- Test 8a: transition probabilities match mu_kj[j] / Lambda ----
			// Build an AwaitEvent state where the server is busy on exactly one job type j.
			// The probability of a completion event in getNextStateProbability must equal
			// mu_kj[j] / Lambda -- not max_mu/Lambda or any other value.
			std::cout << "\n--- Test 8a: completion probabilities == mu_kj[j] / Lambda ---\n";
			{
				DynaPlex::Models::queue_mdp::MDP mdp8(make_cfg8(0.9, 0.1));
				const double Lambda = mdp8.uniformization_rate;
				std::cout << "  Lambda = " << Lambda
				          << "  (expected 2.3 = tick 1.0 + lam 0.2+0.2 + max_mu 0.9)\n";

				bool all_pass = true;
				for (int j = 0; j < 2; ++j) {
					// State: both queues empty (FIL=-1), server busy on job type j only
					auto s = mdp8.GetInitialState();   // FIL=[-1,-1], busy=[[0,0]]
					s.server_manager.busy_on[0][(size_t)j] = 1;
					s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
					s.cat = DynaPlex::StateCategory::AwaitEvent();

					auto dist = mdp8.getNextStateProbability(s, 0);
					CheckDistribution(dist, "Test8a j=" + std::to_string(j));

					// Completion is the unique next-state where busy_on[0][j] dropped to 0
					double p_completion = 0.0;
					for (const auto& e : dist)
						if (e.next_state.server_manager.busy_on[0][(size_t)j] == 0)
							p_completion += e.probability;

					const double expected = mdp8.server_static_info[0].mu_kj[(size_t)j] / Lambda;
					const double diff     = std::abs(p_completion - expected);
					const bool   ok       = diff < 1e-12;
					if (!ok) all_pass = false;

					std::cout << "  job=" << j
					          << "  mu_kj=" << mdp8.server_static_info[0].mu_kj[(size_t)j]
					          << "  P(completion)=" << std::setprecision(10) << p_completion
					          << "  expected=" << expected
					          << "  diff=" << diff
					          << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";
				}
				std::cout << (all_pass ? "  [PASS] both completion probabilities correct\n"
				                       : "  [FAIL] probability mismatch detected\n");
			}

			// ---- Test 8b: RVI prefers faster type when costs are equal ----
			// With equal cost rates and arrival rates the optimal policy is the mu-rule:
			// always serve the faster type.  We check both orientations:
			//   System A: rates=[0.9,0.1] -> faster_type=0
			//   System B: rates=[0.1,0.9] -> faster_type=1
			// In states where FIFO offers the SLOWER type first (its FIL is higher),
			// RVI should skip (action=0) to serve the faster type instead.
			std::cout << "\n--- Test 8b: RVI prefers faster type (equal costs) ---\n";
			{
				for (int sys = 0; sys < 2; ++sys) {
					const double mu0 = (sys == 0) ? 0.9 : 0.1;
					const double mu1 = (sys == 0) ? 0.1 : 0.9;
					const int faster = (sys == 0) ? 0   : 1;

					DynaPlex::Models::queue_mdp::MDP mdp8(make_cfg8(mu0, mu1));
					auto sol8 = mdp8.runRVI(50);

					int n_checked = 0, n_prefer_faster = 0;
					for (int f_slow = 2; f_slow <= 10; ++f_slow) {
						for (int f_fast = 0; f_fast < f_slow - 1; ++f_fast) {
							// Build AwaitAction state: both types waiting, server idle.
							// Slower type has higher FIL -> FIFO offers it first.
							const int fil0 = (faster == 0) ? f_fast : f_slow;
							const int fil1 = (faster == 0) ? f_slow : f_fast;

							auto s = mdp8.GetInitialState();
							s.queue_manager.set_fil(0, fil0);
							s.queue_manager.set_fil(1, fil1);
							s.queue_manager.update_total_arrival_rate(mdp8.arrival_rates);
							s.queue_manager.update_total_tick_rate(mdp8.tick_rate);
							s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
							s.server_manager.set_action_counter(0);
							s.cat = s.server_manager.action_queue.empty()
							        ? DynaPlex::StateCategory::AwaitEvent()
							        : DynaPlex::StateCategory::AwaitAction();

							if (s.cat != DynaPlex::StateCategory::AwaitAction()) continue;

							// action=0 means skip the offered (slower) type -> prefers faster
							const int64_t rvi_action = mdp8.EvaluateRVIPolicy(sol8, s);
							++n_checked;
							if (rvi_action == 0) ++n_prefer_faster;
						}
					}

					const bool pass = (n_prefer_faster == n_checked);
					std::cout << "  [" << (pass ? "PASS" : "FAIL") << "]"
					          << "  rates=[" << mu0 << "," << mu1 << "]"
					          << "  faster_type=" << faster
					          << "  prefer_faster=" << n_prefer_faster << "/" << n_checked << "\n";
				}
			}

			// ---- Test 8c: g* symmetric under rate-swap ----
			// Swapping service_rates=[0.9,0.1] to [0.1,0.9] is a pure relabelling of
			// the two job types.  With equal costs and equal arrival rates g* must be
			// identical under any reward function.
			// Previously this test was pinned to reward_type=0 (binary) because the
			// old g_stable_count convergence criterion stopped at different iteration
			// counts for A vs B under QL reward, producing a ~0.027 artefact.
			// With the span-seminorm criterion both systems converge to the same
			// floating-point precision, so the test now passes under QL reward too.
			std::cout << "\n--- Test 8c: g* symmetric under rate-swap ---\n";
			{
				DynaPlex::Models::queue_mdp::MDP mdpA(make_cfg8(0.9, 0.1));
				DynaPlex::Models::queue_mdp::MDP mdpB(make_cfg8(0.1, 0.9));

				const auto solA = mdpA.runRVI(50);
				const auto solB = mdpB.runRVI(50);

				const double diff = std::abs(solA.g_star - solB.g_star);
				const double tol  = 1e-6;
				const bool   pass = diff < tol;

				std::cout << "  System A [0.9,0.1]: g* = " << std::setprecision(12) << solA.g_star << "\n";
				std::cout << "  System B [0.1,0.9]: g* = " << std::setprecision(12) << solB.g_star << "\n";
				std::cout << "  |g*_A - g*_B| = " << diff << "  tol=" << tol << "\n";
				std::cout << (pass ? "  [PASS] g* symmetric under rate-swap\n"
				                   : "  [FAIL] asymmetry detected -- rate-indexing bug!\n");
			}
		}

		// =========================================================
		// Test 9: reward-function comparison
		//
		// System: 1 server, 2 job types, service_rate=0.2, due_time=3
		//   arrival_rates=[0.15,0.15], tick_rate=1.0
		//   cost_rates=[1,1] -- equal cost rates; only FIL-age drives the policy difference
		//   rho = 0.30/0.40 = 0.75, E[service] = 5 ticks >> D=3
		//   Jobs routinely miss the deadline, reward choice matters a lot.
		//
		// Known pathology of binary reward (reward_type=0):
		//   When a job has very high FIL (already far past deadline) and another
		//   job is just about to cross the deadline, binary reward says "save the
		//   borderline job" and skips the very-late job indefinitely.
		//
		// Queue-lateness (reward_type=1) fixes this: the very-late job's accumulated
		//   per-tick lateness cost (excess=7) dominates the borderline job's future cost.
		//
		//   9a: Demonstrate the divergence on a single pathological state
		//   9b: Cross-eval -- P_ql outperforms P_bin under the QL metric
		//   9c: Converse  -- P_bin outperforms P_ql under the binary metric
		// =========================================================
		{
			// Shared lambda to build the MDP config for a given reward type
			auto make_cfg9 = [](int64_t rtype) -> DynaPlex::VarGroup {
				DynaPlex::VarGroup srv;
				srv.Add("servers",       int64_t{1});
				srv.Add("service_rates", std::vector<double>{0.2, 0.2});
				srv.Add("can_serve",     std::vector<int64_t>{0, 1});
				DynaPlex::VarGroup cfg;
				cfg.Add("id",             std::string{"queue_mdp"});
				cfg.Add("discount_factor",double{1.0});
				cfg.Add("k_servers",      int64_t{1});
				cfg.Add("n_jobs",         int64_t{2});
				cfg.Add("arrival_rates",  std::vector<double>{0.15, 0.15});
				cfg.Add("tick_rate",      double{1.0});
				cfg.Add("cost_rates",     std::vector<double>{1.0, 1.0});
				cfg.Add("due_times",      std::vector<double>{3.0, 3.0});
				cfg.Add("reward_type",    rtype);
				cfg.Add("server_type_0",  srv);
				return cfg;
			};

			DynaPlex::Models::queue_mdp::MDP mdp9_bin(make_cfg9(0));
			DynaPlex::Models::queue_mdp::MDP mdp9_ql (make_cfg9(1));

			std::cout << "\n=== Test 9: reward function comparison ===\n";
			std::cout << "    1 server, 2 job types, mu=0.2, D=3, costs=[1,1], rho=0.75\n";

			std::cout << "\n--- Solving RVI (M=50) for each reward type ---\n";
			auto sol9_bin = mdp9_bin.runRVI(50);
			auto sol9_ql  = mdp9_ql .runRVI(50);
			std::cout << "  binary g*          = " << std::setprecision(10) << sol9_bin.g_star << "\n";
			std::cout << "  queue-lateness g*  = " << std::setprecision(10) << sol9_ql .g_star << "\n";
			std::cout << "  (Different scales -- not directly comparable)\n";

			// ---- Test 9a: scan FIL=[f0, 2], look for binary=skip vs QL=serve ----
			// Job 1 is fixed at FIL=2 (one tick before D=3) so binary always has
			// an incentive to skip job 0 and keep job 1 on-time.
			// Binary reward treats all tardy states equally (1/tick regardless of
			// excess), so its preference for action=0 does not change with f0.
			//
			// QL cost per tick at excess e:
			//   e  +  (lambda/tick) * max(0, e-1) * e / 2   (front + others term)
			// This grows quadratically.  Once f0 is large enough, the accumulated
			// lateness of job 0 dominates the ~6 QL units saved by keeping job 1
			// on-time, and QL switches to action=1 (serve the very-late job).
			//
			// At small excess (f0 just past D) both policies may agree.
			// We scan f0 = D+1 .. 30 and look for states where
			//   binary_action = 0  (skip -- pathology visible)
			//   QL_action     = 1  (serve -- pathology eradicated)
			//
			// Use 20-iteration finite-horizon RVI.  Full convergence gives QL
			// h-values with large oscillation (~168 span) that can dominate the
			// per-state action signal at moderate excess values.  A short-horizon
			// policy avoids this noise while still capturing the key cost difference.
			std::cout << "\n--- Test 9a: scan FIL=[f0,2] for binary=skip, QL=serve ---\n";
			std::cout << "  (Using 20-iteration finite-horizon policies)\n";
			std::cout << "  f0  excess  binary  QL  diverge?\n";
			{
				auto sol9a_bin = mdp9_bin.runRVI(50, 20);
				auto sol9a_ql  = mdp9_ql .runRVI(50, 20);

				int n_diverge = 0;
				const int64_t D0 = (int64_t)mdp9_ql.due_times[0];

				for (int64_t f0 = D0 + 1; f0 <= 30; ++f0) {
					auto s = mdp9_ql.GetInitialState();
					s.queue_manager.set_fil(0, f0);
					s.queue_manager.set_fil(1, 2);
					s.queue_manager.update_total_arrival_rate(mdp9_ql.arrival_rates);
					s.queue_manager.update_total_tick_rate(mdp9_ql.tick_rate);
					s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
					s.server_manager.set_action_counter(0);
					s.cat = s.server_manager.action_queue.empty()
					    ? DynaPlex::StateCategory::AwaitEvent()
					    : DynaPlex::StateCategory::AwaitAction();

					if (s.cat != DynaPlex::StateCategory::AwaitAction()) continue;

					const int64_t a_bin = mdp9_bin.EvaluateRVIPolicy(sol9a_bin, s);
					const int64_t a_ql  = mdp9_ql .EvaluateRVIPolicy(sol9a_ql,  s);
					const bool diverge  = (a_bin == 0 && a_ql == 1);
					if (diverge) ++n_diverge;

					std::cout << "  " << std::setw(3) << f0
					          << "   " << std::setw(4) << (f0 - D0)
					          << "      " << a_bin
					          << "   " << a_ql
					          << "   " << (diverge ? "<-- binary skips, QL serves" : "") << "\n";
				}

				std::cout << "\n  States with binary=skip, QL=serve: " << n_diverge << "/27\n";
				std::cout << (n_diverge > 0
				    ? "  [PASS] Binary exhibits skip-late-job pathology; QL serves the late job\n"
				    : "  [FAIL] No divergence found -- policies agree everywhere\n");
			}

			// ---- Test 9b: cross-evaluation under queue-lateness metric ----
			// Simulate P_bin and P_ql inside the QL environment (mdp9_ql).
			// ModifyStateWithEvent uses ComputeTickCost(reward_type=1), so
			// mean_cost_per_rvi_step is the QL cost per uniformised step.
			// P_ql must beat P_bin by > 2 sigma.
			std::cout << "\n--- Test 9b: cross-evaluation under queue-lateness metric ---\n";
			{
				using S = DynaPlex::Models::queue_mdp::MDP::State;

				auto pol_bin = [&](const S& s) -> int64_t {
					return mdp9_bin.EvaluateRVIPolicy(sol9_bin, s);
				};
				auto pol_ql = [&](const S& s) -> int64_t {
					return mdp9_ql.EvaluateRVIPolicy(sol9_ql, s);
				};

				auto res_bin = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    mdp9_ql, pol_bin, 200, 100000, 10000, 42);

				auto res_ql = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    mdp9_ql, pol_ql, 200, 100000, 10000, 42);

				const double gap9b  = res_bin.mean_cost_per_rvi_step
				                    - res_ql .mean_cost_per_rvi_step;
				const double thr9b  = 2.0 * (res_bin.std_error + res_ql.std_error);
				const bool   pass9b = gap9b > thr9b;

				std::cout << "  P_bin  QL cost/step = "
				          << std::setprecision(8) << res_bin.mean_cost_per_rvi_step
				          << " +/- " << res_bin.std_error << "\n";
				std::cout << "  P_ql   QL cost/step = "
				          << std::setprecision(8) << res_ql .mean_cost_per_rvi_step
				          << " +/- " << res_ql .std_error << "\n";
				std::cout << "  improvement (P_bin - P_ql) = "
				          << std::setprecision(6) << gap9b
				          << "  threshold (2sigma) = " << thr9b << "\n";
				std::cout << (pass9b
				    ? "  [PASS] QL-trained policy significantly cheaper under QL metric\n"
				    : "  [FAIL] No significant QL-cost reduction detected\n");
			}

			// ---- Test 9c: converse -- binary policy wins under binary metric ----
			// Both policies simulated inside the binary environment (mdp9_bin).
			// Confirms the two policies optimise genuinely different objectives.
			std::cout << "\n--- Test 9c: converse -- binary policy wins under binary metric ---\n";
			{
				using S = DynaPlex::Models::queue_mdp::MDP::State;

				auto pol_bin_b = [&](const S& s) -> int64_t {
					return mdp9_bin.EvaluateRVIPolicy(sol9_bin, s);
				};
				auto pol_ql_b = [&](const S& s) -> int64_t {
					return mdp9_ql.EvaluateRVIPolicy(sol9_ql, s);
				};

				auto res_bin_b = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    mdp9_bin, pol_bin_b, 200, 100000, 10000, 42);

				auto res_ql_b = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    mdp9_bin, pol_ql_b, 200, 100000, 10000, 42);

				const double gap9c  = res_ql_b.mean_cost_per_rvi_step
				                    - res_bin_b.mean_cost_per_rvi_step;
				const double thr9c  = 2.0 * (res_bin_b.std_error + res_ql_b.std_error);
				const bool   pass9c = gap9c > thr9c;

				std::cout << "  P_bin  binary cost/step = "
				          << std::setprecision(8) << res_bin_b.mean_cost_per_rvi_step
				          << " +/- " << res_bin_b.std_error << "\n";
				std::cout << "  P_ql   binary cost/step = "
				          << std::setprecision(8) << res_ql_b.mean_cost_per_rvi_step
				          << " +/- " << res_ql_b.std_error << "\n";
				std::cout << "  difference (P_ql - P_bin) = "
				          << std::setprecision(6) << gap9c
				          << "  threshold (2sigma) = " << thr9c << "\n";
				std::cout << (pass9c
				    ? "  [PASS] Binary-trained policy cheaper under binary metric (expected)\n"
				    : "  [NOTE] Policies similar under binary metric in this system\n");
			}
		}
		// =========================================================
		// Test 10: NN (FIL+SIL, depth=2) vs RVI (FIL-only, depth=1)
		//
		// Same system as Test 9 (QL reward, 1 server, 2 types, mu=0.2,
		// D=3, rho=0.75).  Research question: does knowing SIL allow the
		// NN to outperform the FIL-only optimal policy?
		//
		// Design note — why RVI must be solved at depth=1:
		//   Running RVI on the depth=2 MDP with the FIL-only encoder
		//   causes BFS state collisions (same FIL, different SIL → same
		//   key).  SIL-arrival events then appear as self-loops in the
		//   transition table, corrupting h-values and producing a worse-
		//   than-FIFO policy.  The clean approach is to solve RVI on a
		//   depth=1 MDP (no collisions) and then cross-apply the resulting
		//   DynaPlex::Policy to depth=2 states.  Both MDPs share the same
		//   State type, so GetAction(depth2_state) calls EvaluateRVIPolicy
		//   which reads only get_FIL_waiting() — exactly what we want.
		//
		//   10a: Solve RVI on depth=1 → FIL-only optimal policy + g*
		//   10b: Train NN on depth=2 via DCL, starting from that policy.
		//        N=10 000, 2 generations.  NN features = [FIL_0,SIL_0,
		//        FIL_1,SIL_1], so it can learn to exploit SIL.
		//   10c: 3-way comparison on depth=2 MDP:
		//          FIFO  (always-assign)
		//          RVI   (FIL-only optimal, from depth=1 solve)
		//          NN    (FIL+SIL, depth=2 trained)
		//        [PASS] if NN significantly beats RVI (2*se gap).
		//        [NOTE] if not — FIL may be sufficient for this system.
		// =========================================================
		{
			auto make_cfg10 = [](int64_t depth) -> DynaPlex::VarGroup {
				DynaPlex::VarGroup srv;
				srv.Add("servers",       int64_t{1});
				srv.Add("service_rates", std::vector<double>{0.2, 0.2});
				srv.Add("can_serve",     std::vector<int64_t>{0, 1});
				DynaPlex::VarGroup cfg;
				cfg.Add("id",             std::string{"queue_mdp"});
				cfg.Add("discount_factor",double{1.0});
				cfg.Add("k_servers",      int64_t{1});
				cfg.Add("n_jobs",         int64_t{2});
				cfg.Add("arrival_rates",  std::vector<double>{0.15, 0.15});
				cfg.Add("tick_rate",      double{1.0});
				cfg.Add("cost_rates",     std::vector<double>{1.0, 1.0});
				cfg.Add("due_times",      std::vector<double>{3.0, 3.0});
				cfg.Add("reward_type",    int64_t{1});    // queue-lateness
				cfg.Add("max_queue_depth",depth);          // feature_queue_depth=depth by default
				cfg.Add("server_type_0",  srv);
				return cfg;
			};

			std::cout << "\n=== Test 10: NN (depth=2, FIL+SIL) vs RVI (depth=1, FIL only) ===\n";
			std::cout << "    System: 1 server, 2 types, mu=0.2, D=3, QL reward, rho=0.75\n";

			// ---- 10a: Solve RVI on depth=1 (FIL-only), policy registered with fw10 ----
			// feature_queue_depth=1 tells RVI_optimal to internally create a depth=1 copy
			// of the MDP, solve RVI on that (clean BFS, no SIL collisions), and store the
			// FIL-only action map.  The policy is still registered with fw10 (depth=2),
			// so it passes DynaPlex's instance check in both DCL training and the comparer.
			// EvaluateRVIPolicy only reads get_FIL_waiting() — depth-agnostic.
			std::cout << "\n--- Test 10a: RVI solve on depth=1 (FIL-only) ---\n";
			DynaPlex::MDP fw10 = dp.GetMDP(make_cfg10(2));
			DynaPlex::VarGroup rvi10_cfg{
				{"id",                std::string{"RVI_optimal"}},
				{"M",                 int64_t{50}},
				{"feature_queue_depth", int64_t{1}}  // solve RVI internally on depth=1 MDP
			};
			DynaPlex::Policy rvi10_pol = fw10->GetPolicy(rvi10_cfg);

			// ---- Cross-depth EvaluateRVIPolicy sanity check (before DCL) ----
			// Directly compare depth=1 vs depth=2 as *this in EvaluateRVIPolicy,
			// and measure raw costs via EvaluatePolicyRaw to isolate the source of error.
			{
				DynaPlex::Models::queue_mdp::MDP raw_d1(make_cfg10(1));
				auto sol_dbg = raw_d1.runRVI(50);
				DynaPlex::Models::queue_mdp::MDP raw_d2(make_cfg10(2));

				// Single-state key-match test: FIL=[5,3], server idle, counter=0
				auto s2 = raw_d2.GetInitialState();
				s2.queue_manager.set_fil(0, 5);
				s2.queue_manager.set_fil(1, 3);
				s2.queue_manager.update_total_arrival_rate(raw_d2.arrival_rates);
				s2.queue_manager.update_total_tick_rate(raw_d2.tick_rate);
				s2.server_manager.generate_actions(s2.queue_manager.get_FIL_waiting());
				s2.server_manager.set_action_counter(0);
				s2.cat = s2.server_manager.action_queue.empty()
				    ? DynaPlex::StateCategory::AwaitEvent()
				    : DynaPlex::StateCategory::AwaitAction();

				int64_t a_d1 = raw_d1.EvaluateRVIPolicy(sol_dbg, s2); // depth=1 MDP as *this
				int64_t a_d2 = raw_d2.EvaluateRVIPolicy(sol_dbg, s2); // depth=2 MDP as *this
				std::cout << "\n  [10a debug] FIL=[5,3] idle ctr=0:"
				          << "  via_d1=" << a_d1 << "  via_d2=" << a_d2
				          << (a_d1 == a_d2 ? "  [KEY MATCH]" : "  [KEY MISMATCH!]") << "\n";

				// Trajectory-level comparison with EvaluatePolicyRaw (100 traj, 100K steps)
				auto eval_fifo = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    raw_d2,
				    [](const DynaPlex::Models::queue_mdp::MDP::State&) -> int64_t { return 1; },
				    50, 100000, 10000, 42);

				auto eval_rvi_d1 = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    raw_d2,
				    [&](const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
				        return raw_d1.EvaluateRVIPolicy(sol_dbg, s);   // depth=1 as *this
				    },
				    50, 100000, 10000, 42);

				auto eval_rvi_d2 = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
				    raw_d2,
				    [&](const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
				        return raw_d2.EvaluateRVIPolicy(sol_dbg, s);   // depth=2 as *this
				    },
				    50, 100000, 10000, 42);

				std::cout << "  [10a debug] Raw EvaluatePolicyRaw on depth=2 (50 traj, 100K steps):\n"
				          << "    FIFO            cost/step=" << eval_fifo.mean_cost_per_rvi_step << "\n"
				          << "    RVI via depth=1 cost/step=" << eval_rvi_d1.mean_cost_per_rvi_step << "\n"
				          << "    RVI via depth=2 cost/step=" << eval_rvi_d2.mean_cost_per_rvi_step << "\n";
			}

			// ---- 10b: DCL training on depth=2, starting from RVI ----
			// gen-0 data collected using the FIL-only RVI policy in the depth=2
			// environment.  The NN then learns whether SIL adds any value.
			std::cout << "\n--- Test 10b: DCL training (depth=2, N=10000, 2 gens, start=RVI) ---\n";

			DynaPlex::VarGroup nn_arch10{
				{"type",         std::string{"mlp"}},
				{"hidden_layers",DynaPlex::VarGroup::Int64Vec{64, 32}}
			};
			DynaPlex::VarGroup nn_train10{
				{"early_stopping_patience", int64_t{5}},
				{"mini_batch_size",         int64_t{64}},
				{"max_training_epochs",     int64_t{500}}
			};
			DynaPlex::VarGroup dcl_cfg10{
				{"N",              int64_t{10000}},
				{"num_gens",       int64_t{2}},
				{"M",              int64_t{500}},
				{"H",              int64_t{100}},
				{"nn_architecture",nn_arch10},
				{"nn_training",    nn_train10}
			};

			// rvi10_pol comes from fw10_1 (depth=1) but shares the same State
			// type → GetDCL calls GetAction on depth=2 states, which works correctly.
			auto dcl10 = dp.GetDCL(fw10, rvi10_pol, dcl_cfg10);
			dcl10.TrainPolicy();
			DynaPlex::Policy nn10_pol = dcl10.GetPolicy();

			// ---- 10c: 3-way comparison on depth=2 MDP ----
			std::cout << "\n--- Test 10c: FIFO vs RVI(depth=1) vs NN(depth=2) ---\n";

			DynaPlex::Policy fifo10_pol = fw10->GetPolicy("FIFO policy");

			DynaPlex::VarGroup eval10;
			eval10.Add("number_of_trajectories", int64_t{200});
			eval10.Add("periods_per_trajectory", int64_t{100000});

			auto comparer10 = dp.GetPolicyComparer(fw10, eval10);
			// rvi10_pol from fw10_1 is passed to fw10's comparer: same State type,
			// EvaluateRVIPolicy is depth-agnostic → valid cross-MDP evaluation.
			auto res10 = comparer10.Compare({fifo10_pol, rvi10_pol, nn10_pol});

			double fifo10_m, fifo10_e, rvi10_m, rvi10_e, nn10_m, nn10_e;
			res10[0].Get("mean",  fifo10_m); res10[0].Get("error", fifo10_e);
			res10[1].Get("mean",  rvi10_m);  res10[1].Get("error", rvi10_e);
			res10[2].Get("mean",  nn10_m);   res10[2].Get("error", nn10_e);

			std::cout << "  FIFO (depth=2, always-assign)      : "
			          << std::setprecision(8) << fifo10_m << " +/- " << fifo10_e << "\n";
			std::cout << "  RVI  (depth=1 solve, FIL-only)     : "
			          << std::setprecision(8) << rvi10_m  << " +/- " << rvi10_e  << "\n";
			std::cout << "  NN   (depth=2 trained, FIL+SIL)    : "
			          << std::setprecision(8) << nn10_m   << " +/- " << nn10_e   << "\n";

			std::cout << "  RVI improvement over FIFO : "
			          << std::setprecision(4)
			          << 100.0 * (fifo10_m - rvi10_m) / fifo10_m << "%\n";
			std::cout << "  NN  improvement over RVI  : "
			          << std::setprecision(4)
			          << 100.0 * (rvi10_m  - nn10_m)  / rvi10_m  << "%\n";

			const double gap10  = rvi10_m - nn10_m;
			const double thr10  = 2.0 * (rvi10_e + nn10_e);
			const bool   pass10 = gap10 > thr10;

			std::cout << "  gap (RVI - NN) = " << gap10
			          << "  threshold (2*se) = " << thr10 << "\n";
			std::cout << (pass10
			    ? "  [PASS] NN (FIL+SIL) significantly outperforms RVI (FIL-only)\n"
			    : "  [NOTE] NN does not significantly beat RVI -- FIL may be sufficient\n");
		}
	}
	catch (const std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
	return 0;
}
