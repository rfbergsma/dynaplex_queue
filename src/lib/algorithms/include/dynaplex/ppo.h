#pragma once
#include <memory>
#include <vector>
#include "dynaplex/mdp.h"
#include "dynaplex/policy.h"
#include "dynaplex/system.h"
#include "dynaplex/vargroup.h"

namespace DynaPlex::Algorithms {

	/**
	 * Proximal Policy Optimization (PPO) for DynaPlex MDPs.
	 *
	 * Mirrors the DCL interface: construct with (system, mdp, policy_0, config),
	 * call TrainPolicy(), then GetPolicy()/GetPolicies().  The trained policy is a
	 * deterministic (argmax) policy usable directly in PolicyComparer and heatmaps,
	 * exactly like the DCL-trained NN policy.
	 *
	 * Average-cost note: DynaPlex queue MDPs are undiscounted (discount=1.0).  PPO
	 * uses its OWN discount "gae_gamma" (default 0.99) purely for advantage/return
	 * computation; it does not touch the MDP's discount factor.  Per-decision reward
	 * is Objective()*delta(CumulativeReturn), so PPO maximises -cost = minimises cost.
	 *
	 * Config keys (all optional):
	 *   rng_seed (15112017), silent (false)
	 *   num_envs (16)          parallel rollout trajectories
	 *   rollout_steps (256)    decisions collected per env per update
	 *   num_updates (200)      PPO outer iterations
	 *   epochs_per_update (10) optimisation passes over each rollout buffer
	 *   mini_batch_size (256)
	 *   gae_gamma (0.99), gae_lambda (0.95)
	 *   clip_epsilon (0.2), entropy_coef (0.01), value_coef (0.5)
	 *   learning_rate (3e-4), max_grad_norm (0.5)
	 *   normalize_advantages (true)
	 *   nn_architecture (mlp {hidden_layers:[64,32]})  -- shared trunk; heads are added internally
	 *   eval_every (0)         if >0 and a PolicyComparer hook is set, ignored here (printed externally)
	 */
	class PPO
	{
	public:
		PPO(const DynaPlex::System& system, DynaPlex::MDP mdp,
		    DynaPlex::Policy policy_0 = nullptr, const DynaPlex::VarGroup& config = VarGroup{});

		/// Runs the full PPO training loop (num_updates iterations).
		void TrainPolicy();

		/// Returns the current trained (deterministic, argmax) policy.
		/// iteration is accepted for interface symmetry with DCL; only the latest
		/// trained policy is maintained, so any value returns it (and -1 = latest).
		DynaPlex::Policy GetPolicy(int64_t iteration = -1);

		/// Returns {policy_0, trained_policy} for interface symmetry with DCL.
		std::vector<DynaPlex::Policy> GetPolicies();

	private:
		struct Impl;
		std::shared_ptr<Impl> impl;
	};
}
