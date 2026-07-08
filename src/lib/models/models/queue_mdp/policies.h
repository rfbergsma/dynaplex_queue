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
		/// Uses the precomputed is_cmu_winner label: assigns (action=1) when the current
		/// candidate is among the top-capacity_k candidates by c*µ for its pool, skips
		/// otherwise.  With capacity-aware labeling all idle servers in the pool get served
		/// in sequence through the alpha steps.
		class CmuPolicy
		{
			std::shared_ptr<const MDP> mdp;
		public:
			CmuPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		/// Reverse-FIFO (newest-first) policy.
		/// Uses the precomputed is_rfq_winner label: assigns (action=1) when the current
		/// candidate is among the top-capacity_k newest (lowest-FIL) jobs for its pool.
		/// Useful as a DCL training base for problems where the optimal policy prefers
		/// newer / higher-cost jobs over the FIFO ordering.
		class ReverseFIFOPolicy
		{
			std::shared_ptr<const MDP> mdp;
		public:
			ReverseFIFOPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

		/// Enforced-FIFO policy: assigns (action=1) ONLY on the FIFO winner
		/// (is_fifo_winner label), explicit 0 on every other candidate — unlike the
		/// greedy "FIFO policy" (always 1).  Designed to pair with
		/// action_sort="reverse_fifo" (winner presented last): DCL's one-step
		/// deviations then cover "serve this alternative instead" at non-winners and
		/// "serve vs fully idle" at the winner (the strategic-idleness counterfactual).
		/// NEVER pair ascending sort with the greedy FIFO policy (that gives LIFO).
		class EnforcedFIFOPolicy
		{
			std::shared_ptr<const MDP> mdp;
		public:
			EnforcedFIFOPolicy(std::shared_ptr<const MDP> mdp, const VarGroup& config);
			int64_t GetAction(const MDP::State& state) const;
		};

	}
}

