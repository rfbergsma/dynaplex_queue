// One-cell scaling study for Snellius. Each invocation evaluates FIFO and c-mu,
// optionally runs fixed-depth RVI, and optionally trains one PPO seed.
//
// Examples:
//   queue_scale_study level=small phase=all seed=1 preset=0 rvi_m=40
//   queue_scale_study level=frontier phase=baseline rvi_m=18
//   queue_scale_study level=large phase=ppo seed=1 preset=2

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

static VarGroup exp3_config()
{
    VarGroup s0;
    s0.Add("servers", int64_t(1));
    s0.Add("can_serve", VarGroup::Int64Vec{0});
    s0.Add("service_rates", VarGroup::DoubleVec{1.0});
    VarGroup s1;
    s1.Add("servers", int64_t(1));
    s1.Add("can_serve", VarGroup::Int64Vec{0, 1});
    s1.Add("service_rates", VarGroup::DoubleVec{1.0, 1.0});

    VarGroup cfg;
    cfg.Add("id", std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers", int64_t(2));
    cfg.Add("n_jobs", int64_t(2));
    cfg.Add("tick_rate", 1.0);
    cfg.Add("reward_type", int64_t(4));
    cfg.Add("arrival_rates", VarGroup::DoubleVec{0.8, 0.2});
    cfg.Add("cost_rates", VarGroup::DoubleVec{100.0, 300.0});
    cfg.Add("due_times", VarGroup::DoubleVec{3.0, 3.0});
    cfg.Add("server_type_0", s0);
    cfg.Add("server_type_1", s1);
    return cfg;
}

static VarGroup chain_config(int jobs)
{
	const int pools = jobs;
    VarGroup::DoubleVec arrivals, costs, dues;
    for (int j = 0; j < jobs; ++j) {
        arrivals.push_back(j == jobs - 1 ? 0.15 : 0.20);
        costs.push_back(j == jobs - 1 ? 200.0 : 50.0 + 25.0 * j);
        dues.push_back(std::max(3.0, 8.0 - j));
    }

    VarGroup cfg;
    cfg.Add("id", std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers", int64_t(pools));
    cfg.Add("n_jobs", int64_t(jobs));
    cfg.Add("tick_rate", 1.0);
    cfg.Add("reward_type", int64_t(4));
    cfg.Add("arrival_rates", arrivals);
    cfg.Add("cost_rates", costs);
    cfg.Add("due_times", dues);
	for (int k = 0; k < pools; ++k) {
		VarGroup srv;
		srv.Add("servers", int64_t(1));
		if (k < jobs - 1) {
			srv.Add("can_serve", VarGroup::Int64Vec{int64_t(k), int64_t(k + 1)});
			srv.Add("service_rates", VarGroup::DoubleVec{
				1.2 - 0.1 * k, 1.1 - 0.1 * k});
		} else {
			srv.Add("can_serve", VarGroup::Int64Vec{int64_t(jobs - 1)});
			srv.Add("service_rates", VarGroup::DoubleVec{1.2 - 0.1 * (jobs - 1)});
		}
        cfg.Add("server_type_" + std::to_string(k), srv);
    }
    return cfg;
}

int main(int argc, char** argv)
{
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq != std::string::npos) kv[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
    auto S = [&](const std::string& key, const std::string& value) {
        auto it = kv.find(key); return it == kv.end() ? value : it->second;
    };
    auto I = [&](const std::string& key, int64_t value) {
        auto it = kv.find(key); return it == kv.end() ? value : int64_t(std::atoll(it->second.c_str()));
    };
    auto D = [&](const std::string& key, double value) {
        auto it = kv.find(key); return it == kv.end() ? value : std::atof(it->second.c_str());
    };

    const std::string level = S("level", "small");
    const std::string phase = S("phase", "ppo");
    const int64_t seed = I("seed", 1);
    const int preset = int(I("preset", 0));
    const int jobs = level == "small" ? 2 : level == "frontier" ? 3 : level == "medium" ? 4 : 6;
    if (level != "small" && level != "frontier" && level != "medium" && level != "large") {
        std::cerr << "level must be small, frontier, medium, or large\n";
        return 2;
    }
    if (phase != "baseline" && phase != "ppo" && phase != "all") {
        std::cerr << "phase must be baseline, ppo, or all\n";
        return 2;
    }

    auto& dp = DynaPlexProvider::Get();
    VarGroup cfg;
    if (level == "small") {
        cfg = exp3_config();
    } else if (level == "large") {
        cfg = VarGroup::LoadFromFile(dp.FilePath(
            {"mdp_config_examples", "queue_mdp"}, "mdp_config_large_6j5s.json"));
    } else {
        cfg = chain_config(jobs);
    }

    const double tick = D("tick", 3.0);
    const double load_scale = D("load", 1.0);
    VarGroup::DoubleVec arrivals;
    cfg.Get("arrival_rates", arrivals);
    for (double& x : arrivals) x *= load_scale;
    cfg.Set("arrival_rates", arrivals);
    cfg.Set("tick_rate", tick);
    cfg.Set("reward_type", I("reward", 4));
    cfg.Set("action_mode", std::string("per_event"));
    cfg.Set("action_sort", std::string("fifo"));
    cfg.Set("action_labels", S("labels", "all"));
    if (I("force_late", 1) != 0) cfg.Set("force_late_service", true);

    qm::MDP raw(cfg);
    const double Lambda = raw.uniformization_rate;
    auto mdp = dp.GetMDP(cfg);
    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", I("eval_traj", 64));
    eval_cfg.Add("periods_per_trajectory", I("eval_periods", 200000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    auto fifo = mdp->GetPolicy("FIFO policy");
    auto cmu = mdp->GetPolicy("cmu");
    auto base_results = comparer.Compare({fifo, cmu});
    double fifo_mean = 0.0, cmu_mean = 0.0;
    base_results[0].Get("mean", fifo_mean);
    base_results[1].Get("mean", cmu_mean);

	std::cout << "SCALE_RESULT kind=config level=" << level
			  << " jobs=" << jobs << " pools=" << (level == "large" ? 5 : jobs)
              << " seed=" << seed << " preset=" << preset
              << " tick=" << tick << " load=" << load_scale
              << " reward=" << I("reward", 4)
              << " force_late=" << I("force_late", 1) << "\n";
    std::cout << std::fixed << std::setprecision(6)
              << "SCALE_RESULT kind=baseline level=" << level
              << " FIFO=" << fifo_mean * Lambda
              << " CMU=" << cmu_mean * Lambda << "\n" << std::flush;

    const bool run_rvi = I("rvi", phase == "all" ? 1 : 0) != 0;
    if (run_rvi) {
        if (level == "medium" || level == "large") {
            std::cerr << "RVI is disabled for medium and large levels\n";
            return 2;
        }
        const int64_t default_m = level == "small" ? 40 : 16;
        VarGroup rvi_cfg{{"id", std::string("RVI_optimal")},
                         {"M", I("rvi_m", default_m)}, {"silent", int64_t(0)}};
        auto rvi = mdp->GetPolicy(rvi_cfg);
        double rvi_mean = 0.0;
        comparer.Compare({rvi})[0].Get("mean", rvi_mean);
        std::cout << "SCALE_RESULT kind=rvi level=" << level
                  << " RVI=" << rvi_mean * Lambda
                  << " FIFO_over_RVI=" << fifo_mean / rvi_mean
                  << " CMU_over_RVI=" << cmu_mean / rvi_mean << "\n" << std::flush;
    }

    if (phase == "baseline") return 0;

    int64_t width1 = 64, width2 = 32;
    double lr = 3e-4, entropy = 0.01;
    if (preset == 1) { width1 = 128; width2 = 64; }
    if (preset == 2) { lr = 1e-4; }
    if (preset == 3) { entropy = 0.003; }
    if (preset < 0 || preset > 3) {
        std::cerr << "preset must be 0..3\n";
        return 2;
    }

    VarGroup ppo;
    ppo.Add("num_envs", I("envs", 16));
    ppo.Add("rollout_steps", I("rollout", 256));
    ppo.Add("num_updates", I("updates", 3000));
    ppo.Add("epochs_per_update", I("epochs", 4));
    ppo.Add("mini_batch_size", I("minibatch", 256));
    ppo.Add("learning_rate", D("lr", lr));
    ppo.Add("gae_gamma", D("gamma", 0.99));
    ppo.Add("gae_lambda", D("lambda", 0.95));
    ppo.Add("entropy_coef", D("entropy", entropy));
    ppo.Add("average_reward", I("avg", 1) != 0);
    ppo.Add("rho_step", D("rho_step", 1.0));
    ppo.Add("temp_anneal", I("temp_anneal", 1) != 0);
    ppo.Add("temp_min", D("temp_min", 0.25));
    ppo.Add("env_reset_every", I("resets", 16));
    ppo.Add("value_norm", I("vnorm", 1) != 0);
    ppo.Add("dper_clamp", I("dper", 1) != 0);
    ppo.Add("guard_tol_sigma", D("gtol_sigma", 0.0));
    ppo.Add("guard_robust", I("grobust", 1) != 0);
    ppo.Add("guard_leak", D("gleak", 0.0));
    ppo.Add("rng_seed", seed);
    ppo.Add("silent", false);
    VarGroup arch;
    arch.Add("hidden_layers", VarGroup::Int64Vec{width1, width2});
    ppo.Add("nn_architecture", arch);

    std::cout << "SCALE_RESULT kind=ppo_config level=" << level
              << " seed=" << seed << " preset=" << preset
              << " width=" << width1 << "," << width2
              << " lr=" << D("lr", lr) << " entropy=" << D("entropy", entropy)
              << " updates=" << I("updates", 3000) << "\n" << std::flush;
    auto trainer = dp.GetPPO(mdp, nullptr, ppo);
    trainer.TrainPolicy();
    auto policies = std::vector<DynaPlex::Policy>{trainer.GetPolicy(), trainer.GetStochasticPolicy()};
    auto results = comparer.Compare(policies);
    double argmax_mean = 0.0, stoch_mean = 0.0;
    results[0].Get("mean", argmax_mean);
    results[1].Get("mean", stoch_mean);
    std::cout << "SCALE_RESULT kind=ppo level=" << level
              << " seed=" << seed << " preset=" << preset
              << " argmax=" << argmax_mean * Lambda
              << " stochastic=" << stoch_mean * Lambda
              << " argmax_over_FIFO=" << argmax_mean / fifo_mean
              << " argmax_over_CMU=" << argmax_mean / cmu_mean << "\n";
    return 0;
}
