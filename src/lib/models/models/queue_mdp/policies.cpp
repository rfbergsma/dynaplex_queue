#include "policies.h"
#include "mdp.h"
#include "dynaplex/error.h"
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
		RVI_optimal::RVI_optimal(std::shared_ptr<const MDP> mdp, const VarGroup& config)
			: mdp{ mdp }, varGroup{ config }
		{
			if (config.HasKey("M")) {
				int64_t M;
				config.Get("M", M);
				sol = mdp->runRVI((int)M);
			}
			else {
				double rel_tol = 1e-4;
				if (config.HasKey("rel_tol"))
					config.Get("rel_tol", rel_tol);
				sol = mdp->runRVI(rel_tol);
			}
		}

		int64_t RVI_optimal::GetAction(const MDP::State& state) const
		{
			return mdp->EvaluateRVIPolicy(sol, state);
		}
	}
}
