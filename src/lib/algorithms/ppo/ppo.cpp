#include "dynaplex/ppo.h"
#include "dynaplex/error.h"
#include "dynaplex/rng.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

#if DP_TORCH_AVAILABLE
#include <torch/torch.h>
#endif

namespace DynaPlex::Algorithms {

#if DP_TORCH_AVAILABLE

	// ----------------------------------------------------------------------
	// ActorCritic network: shared MLP trunk + policy-logits head + value head.
	// forward(x) -> [B, num_actions + 1]; columns [0..A-1] are policy logits,
	// column A is the state value V(s).
	// ----------------------------------------------------------------------
	struct ActorCriticImpl : torch::nn::Module {
		torch::nn::Sequential trunk{ nullptr };
		torch::nn::Linear     policy_head{ nullptr };
		torch::nn::Linear     value_head{ nullptr };
		torch::nn::Linear     adv_head{ nullptr };
		int64_t num_actions{ 0 };

		ActorCriticImpl(int64_t in_dim, int64_t num_actions_,
		                const std::vector<int64_t>& hidden)
			: num_actions(num_actions_)
		{
			trunk = torch::nn::Sequential();
			int64_t last = in_dim;
			for (size_t i = 0; i < hidden.size(); ++i) {
				trunk->push_back(torch::nn::Linear(last, hidden[i]));
				trunk->push_back(torch::nn::ReLU());
				last = hidden[i];
			}
			register_module("trunk", trunk);
			policy_head = register_module("policy_head", torch::nn::Linear(last, num_actions));
			value_head  = register_module("value_head",  torch::nn::Linear(last, 1));
			// auxiliary advantage head: trained on the MEASURED (GAE) advantages of
			// taken actions.  Its argmax is an alternative deterministic readout that
			// reads simulated consequences instead of policy-logit signs (which the
			// re-presentation forgiveness leaves untrained near ties).
			adv_head    = register_module("adv_head",    torch::nn::Linear(last, num_actions));

			// Small-gain init on the policy head so the initial policy is near-uniform.
			// Without this the default (kaiming) init produces large logits that saturate
			// within a few updates -> premature entropy collapse to a deterministic, often
			// suboptimal, policy.  This is the standard PPO initialisation trick.
			{
				torch::NoGradGuard ng;
				policy_head->weight.mul_(0.01);
				policy_head->bias.zero_();
				adv_head->weight.mul_(0.01);
				adv_head->bias.zero_();
			}
		}

		torch::Tensor forward(torch::Tensor x) {
			torch::Tensor h = trunk->forward(x);
			torch::Tensor logits = policy_head->forward(h);          // [B, A]
			torch::Tensor value  = value_head->forward(h);           // [B, 1]
			torch::Tensor adv    = adv_head->forward(h);             // [B, A]
			return torch::cat({ logits, value, adv }, /*dim=*/1);    // [B, 2A+1]
		}
	};
	TORCH_MODULE(ActorCritic);

	// ----------------------------------------------------------------------
	// Deterministic (argmax) evaluation policy wrapping a trained ActorCritic.
	// Slices off the value column and reuses the MDP's masked argmax logic, so
	// it plugs into PolicyComparer / heatmaps exactly like an NN_Policy.
	// ----------------------------------------------------------------------
	// Flexible readout policy over a trained ActorCritic:
	//   temperature <= 0 : argmax (deterministic)
	//   temperature  > 0 : sample softmax(scores / temperature)
	//   serve_bias       : added to the scores of all actions >= 1 ("serve"-type
	//                      actions) before argmax/sampling.  Near-tie sign errors
	//                      are the diagnosed argmax failure; ties broken toward
	//                      serving are nearly free by the same re-presentation
	//                      argument that makes them invisible to training.
	//   use_adv          : score with the advantage head (measured consequences)
	//                      instead of the policy logits.
	class PPOActorPolicy : public DynaPlex::PolicyInterface {
		DynaPlex::MDP mdp;
		ActorCritic net;
		int64_t num_actions;
		double temperature, serve_bias;
		bool use_adv;
		DynaPlex::VarGroup config;
	public:
		PPOActorPolicy(DynaPlex::MDP mdp, ActorCritic net, int64_t num_actions,
		               double temperature = 0.0, double serve_bias = 0.0, bool use_adv = false)
			: mdp(mdp), net(net), num_actions(num_actions),
			  temperature(temperature), serve_bias(serve_bias), use_adv(use_adv)
		{
			config.Add("id", std::string("PPO_Policy"));
			config.Add("temperature", temperature);
			config.Add("serve_bias", serve_bias);
			config.Add("use_adv", use_adv ? int64_t(1) : int64_t(0));
		}
		std::string TypeIdentifier() const override { return "PPO_Policy"; }
		const DynaPlex::VarGroup& GetConfig() const override { return config; }

		void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override {
			const int64_t B  = static_cast<int64_t>(trajectories.size());
			if (B == 0) return;
			const int64_t in = mdp->NumFlatFeatures();

			torch::Tensor feats = torch::empty({ B, in }, torch::kFloat32);
			mdp->GetFlatFeatures(trajectories,
				std::span<float>(feats.data_ptr<float>(), static_cast<size_t>(B * in)));

			torch::NoGradGuard no_grad;
			ActorCritic net_local = net;   // copy holder (shares module) to call non-const forward
			torch::Tensor out = net_local->forward(feats);           // [B, 2A+1]
			torch::Tensor scores = use_adv
				? out.narrow(1, num_actions + 1, num_actions).contiguous()
				: out.narrow(1, 0, num_actions).contiguous();
			scores = torch::nan_to_num(scores, 0.0, 30.0, -30.0).clamp(-30.0, 30.0);
			if (serve_bias != 0.0 && num_actions >= 2)
				scores.narrow(1, 1, 1) += serve_bias;   // bias ONLY action 1 (serve), not macro-skips

			if (temperature > 0.0) {
				torch::Tensor mask = torch::zeros({ B, num_actions }, torch::kBool);
				mdp->GetMask(trajectories,
					std::span<bool>(mask.data_ptr<bool>(), static_cast<size_t>(B * num_actions)));
				torch::Tensor masked = (scores / temperature).masked_fill(mask.logical_not(), -1e9);
				torch::Tensor probs  = torch::softmax(masked, 1);
				torch::Tensor action = torch::multinomial(probs, 1, true).squeeze(1);
				auto acc = action.accessor<int64_t, 1>();
				for (int64_t b = 0; b < B; ++b)
					trajectories[(size_t)b].NextAction = acc[b];
				return;
			}

			mdp->SetArgMaxAction(trajectories,
				std::span<float>(scores.data_ptr<float>(), static_cast<size_t>(B * num_actions)));
		}
	};

#endif // DP_TORCH_AVAILABLE

	// ======================================================================
	//  PPO::Impl
	// ======================================================================
	struct PPO::Impl {
		DynaPlex::System system;
		DynaPlex::MDP    mdp;
		DynaPlex::Policy policy_0;

		// hyperparameters
		int64_t rng_seed, num_envs, rollout_steps, num_updates, epochs_per_update, mini_batch_size;
		int64_t env_reset_every;
		double  gae_gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef, learning_rate, max_grad_norm;
		double  rho_step, temp_min, skip_all_bias;
		bool    silent, normalize_advantages, entropy_anneal, average_reward, temp_anneal;
		bool    dper_clamp, value_norm;   // ablation knobs; both true = the recipe
		// anneal-guard variants (heavy-tailed rewards defeat the stock guard):
		// guard_tol_sigma > 0: tolerance = k*sigma_hat(health) instead of 5%*|ref|
		// guard_robust: health = median over envs of per-period rates (tail-robust)
		// guard_leak > 0: reference relaxes toward current EMA (un-poisons peaks)
		double  guard_tol_sigma, guard_leak;
		bool    guard_robust;
		std::vector<int64_t> hidden_layers;

		Impl(const DynaPlex::System& system, DynaPlex::MDP mdp, DynaPlex::Policy policy_0, const VarGroup& config)
			: system(system), mdp(mdp), policy_0(policy_0)
		{
			if (!mdp) throw DynaPlex::Error("PPO: mdp should not be null");
			config.GetOrDefault("rng_seed",          rng_seed,          (int64_t)15112017);
			config.GetOrDefault("silent",            silent,            false);
			config.GetOrDefault("num_envs",          num_envs,          (int64_t)16);
			config.GetOrDefault("rollout_steps",     rollout_steps,     (int64_t)256);
			config.GetOrDefault("num_updates",       num_updates,       (int64_t)200);
			config.GetOrDefault("epochs_per_update", epochs_per_update, (int64_t)10);
			config.GetOrDefault("mini_batch_size",   mini_batch_size,   (int64_t)256);
			config.GetOrDefault("gae_gamma",         gae_gamma,         0.99);
			config.GetOrDefault("gae_lambda",        gae_lambda,        0.95);
			config.GetOrDefault("clip_epsilon",      clip_epsilon,      0.2);
			config.GetOrDefault("entropy_coef",      entropy_coef,      0.01);
			config.GetOrDefault("value_coef",        value_coef,        0.5);
			config.GetOrDefault("learning_rate",     learning_rate,     3e-4);
			config.GetOrDefault("max_grad_norm",     max_grad_norm,     0.5);
			config.GetOrDefault("normalize_advantages", normalize_advantages, true);
			config.GetOrDefault("entropy_anneal",    entropy_anneal,    true);
			// default FALSE: the validated recipe is discounted PPO (gamma^dperiods);
			// average-reward mode underperformed at our rollout sizes (needs big batches).
			config.GetOrDefault("average_reward",    average_reward,    false);
			config.GetOrDefault("rho_step",          rho_step,          0.1);
			config.GetOrDefault("temp_anneal",       temp_anneal,       false);
			config.GetOrDefault("temp_min",          temp_min,          0.25);
			config.GetOrDefault("env_reset_every",   env_reset_every,   (int64_t)16);
			// pessimistic init for a macro-skip action at index 2 (queue MDP
			// enable_skip_all): uniform init gives "idle the whole tick" a 1/3
			// prior at every decision, enough to spiral into the never-serve
			// attractor before learning starts.  A bias of -3 starts it at ~2%:
			// rare enough not to poison the data, sampled enough to get gradient.
			config.GetOrDefault("skip_all_bias",     skip_all_bias,     0.0);
			// ABLATION knobs (default = the recipe).  dper_clamp=false restores the
			// original semi-MDP discounting bug: intra-tick decisions (dperiods=0)
			// discounted as if one period passed (gamma^max(1,dp)), subsidizing
			// idleness.  value_norm=false disables running return-std normalization
			// of value targets (value-scale domination).
			config.GetOrDefault("dper_clamp",        dper_clamp,        true);
			config.GetOrDefault("value_norm",        value_norm,        true);
			config.GetOrDefault("guard_tol_sigma",   guard_tol_sigma,   0.0);
			config.GetOrDefault("guard_robust",      guard_robust,      false);
			config.GetOrDefault("guard_leak",        guard_leak,        0.0);

			if (config.HasKey("nn_architecture")) {
				VarGroup arch; config.Get("nn_architecture", arch);
				if (arch.HasKey("hidden_layers"))
					arch.Get("hidden_layers", hidden_layers);
			}
			if (hidden_layers.empty()) hidden_layers = { 64, 32 };
		}

#if DP_TORCH_AVAILABLE
		ActorCritic net{ nullptr };

		void Build() {
			torch::manual_seed(static_cast<uint64_t>(rng_seed));
			net = ActorCritic(mdp->NumFlatFeatures(), mdp->NumValidActions(), hidden_layers);
			if (skip_all_bias != 0.0 && mdp->NumValidActions() >= 3) {
				torch::NoGradGuard ng;
				net->policy_head->bias.data_ptr<float>()[2] = static_cast<float>(skip_all_bias);
			}
		}

		void Train() {
			if (!net) Build();
			const int64_t A = mdp->NumValidActions();
			const int64_t in = mdp->NumFlatFeatures();
			const int64_t E = num_envs;
			const int64_t T = rollout_steps;
			const double  obj = mdp->Objective();   // +1 max, -1 min

			torch::optim::Adam optimizer(net->parameters(),
				torch::optim::AdamOptions(learning_rate).betas({ 0.9, 0.999 }));

			// Persistent rollout trajectories (infinite-horizon chain continues across updates).
			std::vector<DynaPlex::Trajectory> trajs;
			trajs.reserve(static_cast<size_t>(E));
			for (int64_t i = 0; i < E; ++i) {
				trajs.emplace_back(i);
				trajs[(size_t)i].RNGProvider.SeedEventStreams(true, rng_seed, i);
			}
			mdp->InitiateState(trajs);
			mdp->IncorporateUntilNonTrivialAction(trajs);

			// Running scale of the (raw) returns.  The value head predicts NORMALISED
			// returns (return / ret_std_running); we multiply its output by this scale
			// to recover raw values for GAE.  Without this, large binary-cost returns
			// (O(1e2-1e3)) make the value MSE dominate the shared trunk and collapse the
			// policy (value-scale domination).
			double ret_std_running = 1.0;

			// Running estimate of the average reward per period (rho), for
			// average_reward mode.  Rewards are negative costs here, so rho < 0;
			// the differential reward r - rho*dperiods then charges each elapsed
			// period against the long-run average.
			double rho_running = 0.0;

			// guarded-temperature-annealing state (see comment in the update loop)
			double temp_T = 1.0;            // current behavior temperature
			double rew_ema = 0.0;           // EMA of rollout mean reward (health signal)
			bool   rew_ema_init = false;
			double ema_ref = 0.0;           // best EMA since annealing started (ratchet)
			double health_var_ema = 0.0;    // EMA of squared health deviations (adaptive tol)

			// best-sharp snapshot: the guard retreating late in training means the
			// FINAL network is often not the BEST network — a sharp-and-healthy
			// moment mid-training can be lost.  We snapshot the parameters whenever
			// the behavior is meaningfully sharp (T <= 2*temp_min) and the health
			// EMA is not degraded, and restore the snapshot at the end unless the
			// final state matches it.
			std::vector<torch::Tensor> snap_params;
			double snap_score = -std::numeric_limits<double>::infinity();
			double snap_T = 1.0;

			for (int64_t update = 0; update < num_updates; ++update) {
				// STAGGERED ENV RESETS: without resets the persistent envs are a trap —
				// one bad excursion drives all envs into the deep-late region (where the
				// urgency shaping is saturated and no dense gradient exists); training
				// data then only covers that region, the policy's behavior on healthy
				// short-queue states rots unmaintained, and eval from the empty state
				// collapses.  Resetting one env per update keeps the data distribution
				// anchored to the region the deployed policy actually starts in.
				if (env_reset_every > 0) {
					for (int64_t e = 0; e < E; ++e) {
						if ((update + e) % env_reset_every == 0) {
							std::span<DynaPlex::Trajectory> one(&trajs[(size_t)e], 1);
							mdp->InitiateState(one);
							mdp->IncorporateUntilNonTrivialAction(one);
						}
					}
				}
				// entropy annealing: full entropy_coef during the first half of training
				// (exploration), then linear decay to 0 so the policy sharpens toward a
				// deterministic one that argmax extraction reads out faithfully.
				double ent_coef_now = entropy_coef;
				if (entropy_anneal && num_updates > 1) {
					const double frac = (double)update / (double)(num_updates - 1);
					ent_coef_now = entropy_coef * std::clamp(2.0 * (1.0 - frac), 0.0, 1.0);
				}

				// GUARDED temperature annealing of the BEHAVIOR policy: sample from
				// softmax(logits / T).  As T drops, rollouts increasingly reflect the
				// argmax readout, so states where the argmax action is wrong actually
				// hurt the return and receive gradient — this trains the policy that
				// will be deployed, closing the stochastic-vs-argmax extraction gap.
				// The guard: T only steps DOWN when the EMA of the rollout reward is
				// not degrading, and steps back UP when it is.  Blind (clock-driven)
				// annealing amplifies whichever mode the policy is in mid-training:
				// good seeds sharpen to the best results, bad seeds lock into collapse.
				// (T_now is maintained across updates in temp_T; the EMA update happens
				// after the rollout, the T decision just before the next one.)
				double T_now = 1.0;
				if (temp_anneal && num_updates > 1) {
					const int64_t anneal_start = num_updates / 2;
					if (update == anneal_start) ema_ref = rew_ema;   // baseline at anneal start
					if (update > anneal_start && rew_ema_init
					    && (update - anneal_start) % 5 == 0) {
						// tolerance: fixed 5% of |ref| (stock), or k*sigma of the health
						// signal itself (guard_tol_sigma) — heavy-tailed rewards make the
						// fixed band read ordinary fluctuation as degradation.
						const double tol = (guard_tol_sigma > 0.0)
							? guard_tol_sigma * std::sqrt(health_var_ema)
							: 0.05 * (std::abs(ema_ref) + 1e-9);
						if (rew_ema >= ema_ref - tol) {
							if (temp_T > temp_min) temp_T = std::max(temp_min, temp_T * 0.9);
							if (rew_ema > ema_ref) ema_ref = rew_ema;     // ratchet reference up
						} else {
							temp_T = std::min(1.0, temp_T / 0.9);         // degrading: back off
							// leaky ratchet: relax the reference toward the current EMA so
							// one lucky peak cannot permanently poison the guard
							if (guard_leak > 0.0)
								ema_ref += guard_leak * (rew_ema - ema_ref);
						}
					}
					T_now = temp_T;
				}
				// ----- Rollout buffers (flat: index = t*E + e) -----
				torch::Tensor buf_feat = torch::empty({ T * E, in }, torch::kFloat32);
				torch::Tensor buf_mask = torch::zeros({ T * E, A }, torch::kBool);
				torch::Tensor buf_act  = torch::empty({ T * E }, torch::kInt64);
				torch::Tensor buf_logp = torch::empty({ T * E }, torch::kFloat32);
				torch::Tensor buf_val  = torch::empty({ T * E }, torch::kFloat32);
				torch::Tensor buf_rew  = torch::empty({ T * E }, torch::kFloat32);
				// periods (events) elapsed between this decision and the next; used for
				// semi-MDP time-aware discounting (gamma^dperiods) so that skip vs assign,
				// which take different amounts of time, are credited consistently.
				// NOTE: 0 is a valid value — the queue MDP presents multiple candidates per
				// tick (cat stays AwaitAction), so consecutive decisions within one tick span
				// zero periods and must get discount gamma^0 = 1.
				torch::Tensor buf_dp   = torch::empty({ T * E }, torch::kFloat32);

				for (int64_t t = 0; t < T; ++t) {
					const int64_t base = t * E;
					// features for this step
					torch::Tensor feats = buf_feat.narrow(0, base, E);
					mdp->GetFlatFeatures(trajs,
						std::span<float>(feats.data_ptr<float>(), static_cast<size_t>(E * in)));
					// mask (zero-init, GetMask sets allowed=true)
					torch::Tensor mask = buf_mask.narrow(0, base, E);
					mdp->GetMask(trajs,
						std::span<bool>(mask.data_ptr<bool>(), static_cast<size_t>(E * A)));

					torch::Tensor logits, value;
					{
						torch::NoGradGuard ng;
						torch::Tensor out = net->forward(feats);
						logits = out.narrow(1, 0, A);
						value  = out.narrow(1, A, 1).squeeze(1) * ret_std_running; // raw value
					}
					// guard against network blow-up (inf/nan logits crash multinomial)
					logits = torch::nan_to_num(logits, 0.0, 30.0, -30.0).clamp(-30.0, 30.0);
					torch::Tensor masked = (logits / T_now).masked_fill(mask.logical_not(), -1e9);
					torch::Tensor probs  = torch::softmax(masked, 1);
					torch::Tensor logp_all = torch::log_softmax(masked, 1);
					torch::Tensor action = torch::multinomial(probs, 1, true).squeeze(1); // [E]
					torch::Tensor logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1);

					buf_act.narrow(0, base, E).copy_(action);
					buf_logp.narrow(0, base, E).copy_(logp);
					buf_val.narrow(0, base, E).copy_(value);

					// apply actions and advance to next decision; reward = obj * delta(CumulativeReturn)
					auto act_acc = action.accessor<int64_t, 1>();
					std::vector<double> c_before(static_cast<size_t>(E));
					std::vector<int64_t> p_before(static_cast<size_t>(E));
					for (int64_t e = 0; e < E; ++e) {
						trajs[(size_t)e].NextAction = act_acc[e];
						c_before[(size_t)e] = trajs[(size_t)e].CumulativeReturn;
						p_before[(size_t)e] = trajs[(size_t)e].PeriodCount;
					}
					mdp->IncorporateAction(trajs);
					mdp->IncorporateUntilNonTrivialAction(trajs);
					torch::Tensor rew_slice = buf_rew.narrow(0, base, E);
					torch::Tensor dp_slice  = buf_dp.narrow(0, base, E);
					auto rew_acc = rew_slice.accessor<float, 1>();
					auto dp_acc  = dp_slice.accessor<float, 1>();
					for (int64_t e = 0; e < E; ++e) {
						rew_acc[e] = static_cast<float>(obj * (trajs[(size_t)e].CumulativeReturn - c_before[(size_t)e]));
						int64_t dper = trajs[(size_t)e].PeriodCount - p_before[(size_t)e];
						dp_acc[e] = static_cast<float>(dper);
					}
				}

				// ----- rollout health EMA (drives the guarded temperature controller) -----
				// PER-PERIOD rate, NOT per-decision mean: intra-tick skip decisions carry
				// zero reward, so an idling policy with a full queue makes many free
				// decisions per tick and DILUTES the per-decision mean — a guard on that
				// signal happily sharpens into idle-collapse while "improving".
				{
					double mr;
					if (guard_robust) {
						// median over envs of per-env per-period rates: a single
						// deep-queue env cannot drag the health signal (heavy tails)
						std::vector<double> rates;
						rates.reserve((size_t)E);
						auto racc = buf_rew.accessor<float, 1>();
						auto dacc = buf_dp.accessor<float, 1>();
						for (int64_t e = 0; e < E; ++e) {
							double sr = 0.0, sd = 0.0;
							for (int64_t t = 0; t < T; ++t) {
								sr += racc[t * E + e];
								sd += dacc[t * E + e];
							}
							rates.push_back(sr / std::max(1.0, sd));
						}
						std::nth_element(rates.begin(), rates.begin() + rates.size() / 2, rates.end());
						mr = rates[rates.size() / 2];
					} else {
						const double sum_dp = buf_dp.sum().item<double>();
						mr = buf_rew.sum().item<double>() / std::max(1.0, sum_dp);
					}
					if (!rew_ema_init) { rew_ema = mr; rew_ema_init = true; }
					else {
						const double dev = mr - rew_ema;
						health_var_ema = 0.9 * health_var_ema + 0.1 * dev * dev;
						rew_ema = 0.9 * rew_ema + 0.1 * mr;
					}
				}

				// best-sharp snapshot: the rollout just taken measured (params, T_now);
				// capture params BEFORE this update's optimizer steps change them.
				if (temp_anneal && T_now <= 2.0 * temp_min + 1e-9) {
					const double tol = 0.05 * (std::abs(ema_ref) + 1e-9);
					const bool healthy = rew_ema >= ema_ref - tol;
					const bool sharper  = T_now < snap_T - 1e-9;
					const bool better   = T_now <= snap_T + 1e-9 && rew_ema > snap_score;
					if (healthy && (sharper || better)) {
						snap_T = T_now; snap_score = rew_ema;
						torch::NoGradGuard ng;
						snap_params.clear();
						for (const auto& p : net->parameters()) snap_params.push_back(p.detach().clone());
					}
				}

				// ----- bootstrap value V(s_T) for each env -----
				torch::Tensor boot;
				{
					torch::Tensor feats = torch::empty({ E, in }, torch::kFloat32);
					mdp->GetFlatFeatures(trajs,
						std::span<float>(feats.data_ptr<float>(), static_cast<size_t>(E * in)));
					torch::NoGradGuard ng;
					torch::Tensor out = net->forward(feats);
					boot = (out.narrow(1, A, 1).squeeze(1) * ret_std_running).contiguous();   // raw [E]
				}

				// ----- update rho (average reward per period) from this rollout -----
				if (average_reward) {
					const double sum_rew = buf_rew.sum().item<double>();
					const double sum_dp  = buf_dp.sum().item<double>();
					if (sum_dp > 0.0) {
						const double batch_rho = sum_rew / sum_dp;
						if (update == 0) rho_running = batch_rho;
						else             rho_running = (1.0 - rho_step) * rho_running + rho_step * batch_rho;
					}
				}

				// ----- GAE (per env, backwards) -----
				// average_reward mode: differential rewards r - rho*dperiods, gamma=1
				// (the critic learns the RVI-style relative value function).
				// discounted mode: time-aware discount gamma^dperiods (semi-MDP).
				torch::Tensor buf_adv = torch::empty({ T * E }, torch::kFloat32);
				torch::Tensor buf_ret = torch::empty({ T * E }, torch::kFloat32);
				{
					auto rew = buf_rew.accessor<float, 1>();
					auto val = buf_val.accessor<float, 1>();
					auto dp  = buf_dp.accessor<float, 1>();
					auto adv = buf_adv.accessor<float, 1>();
					auto ret = buf_ret.accessor<float, 1>();
					auto bv  = boot.accessor<float, 1>();
					for (int64_t e = 0; e < E; ++e) {
						double gae = 0.0;
						for (int64_t t = T - 1; t >= 0; --t) {
							const int64_t idx = t * E + e;
							const double next_val = (t == T - 1) ? bv[e] : val[(t + 1) * E + e];
							double delta, decay;
							if (average_reward) {
								const double r_diff = rew[idx] - rho_running * (double)dp[idx];
								delta = r_diff + next_val - val[idx];
								decay = gae_lambda;
							} else {
								const double dpx = dper_clamp ? (double)dp[idx]
								                              : std::max(1.0, (double)dp[idx]);
								const double disc = std::pow(gae_gamma, dpx);
								delta = rew[idx] + disc * next_val - val[idx];
								decay = disc * gae_lambda;
							}
							gae = delta + decay * gae;
							adv[idx] = static_cast<float>(gae);
							ret[idx] = static_cast<float>(gae + val[idx]);
						}
					}
				}
				if (normalize_advantages) {
					double mean = buf_adv.mean().item<double>();
					double std  = buf_adv.std().item<double>();
					buf_adv = (buf_adv - mean) / (std + 1e-8);
				}

				// Update the running return scale (used to normalise value targets and to
				// rescale the value head's output back to raw units).  Initialise from the
				// first rollout so even update 0 trains on O(1) value targets.
				if (value_norm) {
					double cur_std = buf_ret.std().item<double>();
					if (cur_std < 1e-6) cur_std = 1e-6;
					if (update == 0) ret_std_running = cur_std;
					else             ret_std_running = 0.95 * ret_std_running + 0.05 * cur_std;
				}   // else: ret_std_running stays 1.0 — raw value targets
				torch::Tensor buf_ret_norm = buf_ret / ret_std_running;   // value-head target

				// ----- PPO update: K epochs over minibatches -----
				const int64_t NB = T * E;
				std::vector<int64_t> idx(static_cast<size_t>(NB));
				std::iota(idx.begin(), idx.end(), 0);
				DynaPlex::RNG shuffle_rng{ false, rng_seed + update + 1 };

				double last_ploss = 0, last_vloss = 0, last_ent = 0;
				for (int64_t epoch = 0; epoch < epochs_per_update; ++epoch) {
					std::shuffle(idx.begin(), idx.end(), shuffle_rng.gen());
					for (int64_t start = 0; start + mini_batch_size <= NB; start += mini_batch_size) {
						torch::Tensor sel = torch::from_blob(idx.data() + start,
							{ mini_batch_size }, torch::kInt64).clone();

						torch::Tensor mb_feat = buf_feat.index_select(0, sel);
						torch::Tensor mb_mask = buf_mask.index_select(0, sel);
						torch::Tensor mb_act  = buf_act.index_select(0, sel);
						torch::Tensor mb_oldlp= buf_logp.index_select(0, sel);
						torch::Tensor mb_adv  = buf_adv.index_select(0, sel);
						torch::Tensor mb_ret  = buf_ret_norm.index_select(0, sel);   // normalised value target

						torch::Tensor out = net->forward(mb_feat);
						torch::Tensor logits = out.narrow(1, 0, A);
						torch::Tensor value  = out.narrow(1, A, 1).squeeze(1);       // normalised value prediction
						// same temperature as the rollout that produced this data: the
						// PPO ratio must compare old/new within the same tempered family.
						torch::Tensor masked = (logits / T_now).masked_fill(mb_mask.logical_not(), -1e9);
						torch::Tensor logp_all = torch::log_softmax(masked, 1);
						torch::Tensor probs = torch::softmax(masked, 1);
						torch::Tensor new_logp = logp_all.gather(1, mb_act.unsqueeze(1)).squeeze(1);

						torch::Tensor ratio = torch::exp(new_logp - mb_oldlp);
						torch::Tensor surr1 = ratio * mb_adv;
						torch::Tensor surr2 = torch::clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_adv;
						torch::Tensor policy_loss = -torch::min(surr1, surr2).mean();
						torch::Tensor value_loss = torch::mse_loss(value, mb_ret);
						torch::Tensor entropy = -(probs * logp_all).sum(1).mean();

						// auxiliary advantage head: regress the taken action's predicted
						// advantage onto the measured (normalized) GAE advantage.
						torch::Tensor adv_pred = out.narrow(1, A + 1, A)
							.gather(1, mb_act.unsqueeze(1)).squeeze(1);
						torch::Tensor adv_loss = torch::mse_loss(adv_pred, mb_adv);

						torch::Tensor loss = policy_loss + value_coef * value_loss
						                   - ent_coef_now * entropy + 0.5 * adv_loss;

						optimizer.zero_grad();
						loss.backward();
						torch::nn::utils::clip_grad_norm_(net->parameters(), max_grad_norm);
						optimizer.step();

						last_ploss = policy_loss.item<double>();
						last_vloss = value_loss.item<double>();
						last_ent   = entropy.item<double>();
					}
				}

				if (update == num_updates - 1 && !snap_params.empty()) {
					// restore the best-sharp snapshot unless the final net matches it
					const double tol = 0.05 * (std::abs(ema_ref) + 1e-9);
					const bool final_ok = (temp_T <= snap_T + 1e-9) && (rew_ema >= snap_score - tol);
					if (!final_ok) {
						torch::NoGradGuard ng;
						auto params = net->parameters();
						for (size_t i = 0; i < params.size() && i < snap_params.size(); ++i)
							params[i].copy_(snap_params[i]);
						if (!silent)
							system << "[PPO] restored best-sharp snapshot (T=" << snap_T
							       << ", ema=" << snap_score << ")" << std::endl;
					}
				}

				if (!silent && (update % 10 == 0 || update == num_updates - 1)) {
					// per-period rate is the honest health number; the per-decision mean
					// is diluted by zero-reward intra-tick decisions (see EMA comment).
					const double mean_rew = buf_rew.mean().item<double>();
					const double rate = buf_rew.sum().item<double>()
					                  / std::max(1.0, buf_dp.sum().item<double>());
					system << "[PPO] update " << update
					       << "  rew/period=" << rate
					       << "  mean_reward=" << mean_rew;
					if (average_reward) system << "  rho=" << rho_running;
					if (temp_anneal)    system << "  T=" << T_now;
					system << "  ploss=" << last_ploss
					       << "  vloss=" << last_vloss
					       << "  entropy=" << last_ent;
					// taken-action fractions this rollout: the direct behavioral signal
					// (an eval-side pathology shows healthy fractions here yet a
					// degenerate argmax; a training collapse shows one fraction -> 1).
					{
						torch::Tensor counts = torch::bincount(buf_act, {}, A);
						auto cacc = counts.accessor<int64_t, 1>();
						system << "  act%=[";
						for (int64_t a = 0; a < A; ++a)
							system << (a ? " " : "")
							       << std::round(10000.0 * cacc[a] / (double)NB) / 100.0;
						system << "]";
					}
					system << std::endl;
				}
			}
		}

		DynaPlex::Policy GetTrainedPolicy(double temperature = 0.0, double serve_bias = 0.0, bool use_adv = false) {
			if (!net) throw DynaPlex::Error("PPO::GetPolicy - call TrainPolicy() first.");
			return std::make_shared<PPOActorPolicy>(mdp, net, mdp->NumValidActions(),
			                                        temperature, serve_bias, use_adv);
		}
#else
		void Train() {
			throw DynaPlex::Error("PPO: Torch not available. Set dynaplex_enable_pytorch=true.");
		}
		DynaPlex::Policy GetTrainedPolicy(double = 0.0, double = 0.0, bool = false) {
			throw DynaPlex::Error("PPO: Torch not available. Set dynaplex_enable_pytorch=true.");
		}
#endif
	};

	// ======================================================================
	//  PPO public interface
	// ======================================================================
	PPO::PPO(const DynaPlex::System& system, DynaPlex::MDP mdp, DynaPlex::Policy policy_0, const VarGroup& config)
		: impl(std::make_shared<Impl>(system, mdp, policy_0, config))
	{
	}

	void PPO::TrainPolicy() { impl->Train(); }

	DynaPlex::Policy PPO::GetPolicy(int64_t /*iteration*/) { return impl->GetTrainedPolicy(); }

	DynaPlex::Policy PPO::GetStochasticPolicy() { return impl->GetTrainedPolicy(1.0); }

	DynaPlex::Policy PPO::GetReadoutPolicy(double temperature, double serve_bias, bool use_adv) {
		return impl->GetTrainedPolicy(temperature, serve_bias, use_adv);
	}

	std::vector<DynaPlex::Policy> PPO::GetPolicies() {
		std::vector<DynaPlex::Policy> out;
		if (impl->policy_0) out.push_back(impl->policy_0);
		out.push_back(impl->GetTrainedPolicy());
		return out;
	}
}
