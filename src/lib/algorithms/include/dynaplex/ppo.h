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
	 * Average-cost note: DynaPlex queue MDPs are undiscounted (discount=1.0).  By
	 * default PPO runs in AVERAGE-REWARD mode (Dai & Gluzman style): each step's
	 * reward is replaced by the differential reward r - rho*dperiods, where rho is
	 * a running estimate of the average reward per period, and GAE runs undiscounted
	 * (gamma=1).  The critic then learns the differential (relative) value function —
	 * the same object RVI computes.  Idling is charged rho per elapsed period, so the
	 * never-serve attractor loses its discounting subsidy, and long-run gains (e.g.
	 * cmu-style priorities over FIFO) are not truncated by an effective horizon.
	 * Set average_reward=false for classic discounted GAE with gae_gamma.
	 * Per-decision reward is Objective()*delta(CumulativeReturn), so PPO maximises
	 * -cost = minimises cost.
	 *
	 * Config keys (all optional):
	 *   rng_seed (15112017), silent (false)
	 *   num_envs (16)          parallel rollout trajectories
	 *   rollout_steps (256)    decisions collected per env per update
	 *   num_updates (200)      PPO outer iterations
	 *   epochs_per_update (10) optimisation passes over each rollout buffer
	 *   mini_batch_size (256)
	 *   average_reward (true)  differential rewards r - rho*dperiods with undiscounted
	 *                          GAE; gae_gamma is only used when this is false
	 *   rho_step (0.1)         EMA step for the running average-reward estimate rho
	 *   gae_gamma (0.99), gae_lambda (0.95)
	 *   clip_epsilon (0.2), entropy_coef (0.01), value_coef (0.5)
	 *   entropy_anneal (true)  keep entropy_coef for the first half of training,
	 *                          then decay linearly to 0 so the policy sharpens and
	 *                          argmax extraction matches the stochastic policy
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

		/// Returns the trained policy in STOCHASTIC mode: actions are sampled from
		/// the softmax distribution instead of argmax.  This is the policy PPO
		/// actually optimises; comparing it against GetPolicy() exposes
		/// extraction mismatch (stochastic good, argmax bad = probabilities
		/// hovering below 0.5 in states where repeated stochastic tries suffice).
		DynaPlex::Policy GetStochasticPolicy();

		/// Returns {policy_0, trained_policy} for interface symmetry with DCL.
		std::vector<DynaPlex::Policy> GetPolicies();

	private:
		struct Impl;
		std::shared_ptr<Impl> impl;
	};
}
