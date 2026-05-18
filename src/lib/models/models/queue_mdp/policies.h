#pragma once
#include <cstdint>
#include "mdp.h"
#include "dynaplex/vargroup.h"
#include <memory>

namespace DynaPlex::Models {
	namespace queue_mdp /*must be consistent everywhere for complete mdp defininition and associated policies.*/
	{
		class FIFOPolicy
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
		public:
			FIFOPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class FIFOPolicySorted
		{
			//this is the MDP defined inside the current namespace!
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
		public:
			FIFOPolicySorted(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		class RVI_optimal
		{
			// holds the converged value-function action map from RVI
			std::shared_ptr<const MDP> mdp;
			const VarGroup varGroup;
			MDP::RVISolution sol;
		public:
			RVI_optimal(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		/// FIFO with probabilistic skipping.
		/// At each AwaitAction step, if stochastic_draws[action_counter] < threshold
		/// the policy skips the current candidate (action=0); otherwise it assigns (action=1).
		/// threshold=0.0 (default) is identical to plain FIFO.
		/// Useful as a DCL base policy to inject action=0 training examples.
		class StochasticFIFOPolicy
		{
			std::shared_ptr<const MDP> mdp;
			double threshold;
		public:
			StochasticFIFOPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		/// c-mu scheduling policy.
		/// At each AwaitAction step, checks whether any remaining candidate in the action
		/// queue for the SAME server has a strictly higher c*µ value than the current
		/// candidate.  If yes, skips the current candidate (action=0) so the higher-priority
		/// job type is presented next.  Otherwise assigns immediately (action=1).
		/// Ties are broken in FIFO order (the MDP's SortActionsFIFO pre-sorts by FIL).
		/// This provides a much better training base than FIFO for asymmetric-cost problems.
		class CmuPolicy
		{
			std::shared_ptr<const MDP> mdp;
		public:
			CmuPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

	}
}

