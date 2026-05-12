#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
#include <memory>
#include <iostream>
namespace DynaPlex::Models {
	namespace queue_mdp /*keep this namespace name in line with the name space in which the mdp corresponding to this policy is defined*/
	{

		//MDP and State refer to the specific ones defined in current namespace
		FIFOPolicy::FIFOPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			//Here, you may initiate any policy parameters.
		}

		int64_t FIFOPolicy::GetAction(const MDP::State& state) const
		{
			//Implement custom policy, and remove below line.
			//throw DynaPlex::NotImplementedError();
			return 1; // always assign job
		}


		//MDP and State refer to the specific ones defined in current namespace
		FIFOPolicySorted::FIFOPolicySorted(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			:mdp{ mdp }
		{
			//Here, you may initiate any policy parameters.
		}

		int64_t FIFOPolicySorted::GetAction(const MDP::State& state) const
		{
			//Implement custom policy, and remove below line.
			//throw DynaPlex::NotImplementedError();
			return 1; // always assign job
		}

		// RVI_optimal: runs RVI once in the constructor, caches the action map.
		// Supports an optional "feature_queue_depth" config key (default = mdp->max_queue_depth).
		// When feature_queue_depth=1 is requested on a depth>1 MDP, a temporary depth=1 MDP is
		// constructed from the same parameters and RVI is solved on that instead.  The resulting
		// action_map uses the FIL-only encoder, which is also what EvaluateRVIPolicy reads via
		// get_FIL_waiting() -- so the look-up is depth-agnostic and the policy is valid for the
		// original (depth=2) MDP despite the internal depth-1 solve.
		RVI_optimal::RVI_optimal(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			: mdp{ mdp }, varGroup{ config }
		{
			// Extract silent flag from config
			bool silent = false;
			if (config.HasKey("silent")) {
				int64_t s;
				config.Get("silent", s);
				silent = (s != 0);
			}

			// Determine which MDP to solve RVI on
			int64_t rvi_depth = mdp->max_queue_depth;
			if (config.HasKey("feature_queue_depth"))
				config.Get("feature_queue_depth", rvi_depth);

			// If a reduced depth is requested, build a temporary MDP for the solve
			std::unique_ptr<MDP> temp_mdp;
			const MDP* solve_on = mdp.get();
			if (rvi_depth < mdp->max_queue_depth) {
				// The MDP constructor normalises cost_rates (/tick_rate) and due_times (*tick_rate)
				// on load. mdp->cost_rates / mdp->due_times are already normalised, so we must
				// invert before passing them through the constructor again to avoid double-scaling.
				std::vector<double> orig_cost_rates(mdp->cost_rates.size());
				for (size_t i = 0; i < orig_cost_rates.size(); ++i)
					orig_cost_rates[i] = mdp->cost_rates[i] * mdp->tick_rate;
				std::vector<double> orig_due_times(mdp->due_times.size());
				for (size_t i = 0; i < orig_due_times.size(); ++i)
					orig_due_times[i] = mdp->due_times[i] / mdp->tick_rate;

				VarGroup rc;
				rc.Add("discount_factor", mdp->discount_factor);
				rc.Add("k_servers",       mdp->k_servers);
				rc.Add("n_jobs",          mdp->n_jobs);
				rc.Add("arrival_rates",   mdp->arrival_rates);
				rc.Add("tick_rate",       mdp->tick_rate);
				rc.Add("cost_rates",      orig_cost_rates);
				rc.Add("due_times",       orig_due_times);
				rc.Add("reward_type",     mdp->reward_type);
				rc.Add("max_queue_depth", rvi_depth);
				for (int64_t i = 0; i < mdp->k_servers; ++i) {
					VarGroup srv;
					srv.Add("servers",       mdp->server_static_info[(size_t)i].servers);
					srv.Add("service_rates", mdp->server_static_info[(size_t)i].mu_kj);
					srv.Add("can_serve",     mdp->server_static_info[(size_t)i].can_serve);
					rc.Add("server_type_" + std::to_string(i), srv);
				}
				temp_mdp  = std::make_unique<MDP>(rc);
				solve_on  = temp_mdp.get();
			}

			if (!silent) {
				std::cout << "[RVI_optimal] rvi_depth=" << rvi_depth
				          << "  mdp->max_queue_depth=" << mdp->max_queue_depth
				          << "  solving on depth=" << solve_on->max_queue_depth << "\n";
			}

			if (config.HasKey("M")) {
				int64_t M;
				config.Get("M", M);
				sol = solve_on->runRVI((int)M, 10000, silent);
			}
			else {
				double rel_tol = 1e-4;
				if (config.HasKey("rel_tol"))
					config.Get("rel_tol", rel_tol);
				sol = solve_on->runRVI(rel_tol, silent);
			}

			// Debug: count action=0 vs action=1 in the map
			if (!silent) {
				int n0 = 0, n1 = 0;
				for (auto& kv : sol.action_map) { if (kv.second == 0) ++n0; else ++n1; }
				std::cout << "[RVI_optimal] action_map size=" << sol.action_map.size()
				          << "  a=0:" << n0 << "  a=1:" << n1 << "\n";
			}
		}

		int64_t RVI_optimal::GetAction(const MDP::State& state) const
		{
			return mdp->EvaluateRVIPolicy(sol, state);
		}
	}
}
