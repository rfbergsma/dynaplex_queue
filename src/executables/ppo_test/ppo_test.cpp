// ppo_test.cpp
//
// Integration tests for the PPO subsystem (dp.GetPPO).  Exercises the public API
// end-to-end on the M/M/1 queue, where the optimal policy is FIFO (assign whenever
// a job is present and the server is idle) and RVI gives the exact optimum.
//
// Tests:
//   T1: PPO trains and produces a policy usable in PolicyComparer (no throw).
//   T2: PPO converges to near-optimal on M/M/1 (NN/RVI within tolerance).
//   T3: PPO policy is deterministic (repeated evaluation gives identical cost).
//
// Exit code 0 = all pass, 1 = any fail.

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"

using namespace DynaPlex;

static int g_pass = 0, g_fail = 0;
static void CHECK(bool cond, const std::string& msg) {
    std::cout << (cond ? "  [PASS] " : "  [FAIL] ") << msg << "\n";
    if (cond) ++g_pass; else ++g_fail;
}

static VarGroup mm1_config(double lam, double mu, double tick_rate) {
    VarGroup srv;
    srv.Add("servers",      int64_t(1));
    srv.Add("can_serve",    VarGroup::Int64Vec{0});
    srv.Add("service_rate", mu);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(1));
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0});
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0});
    cfg.Add("server_type_0",   srv);
    return cfg;
}

int main() {
    auto& dp = DynaPlexProvider::Get();

    std::cout << "================ ppo_test ================\n";

    const double mu = 1.0, rho = 0.6, lam = rho * mu;
    auto cfg = mm1_config(lam, mu, /*tick_rate=*/1.0);
    auto mdp = dp.GetMDP(cfg);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(100));
    eval_cfg.Add("periods_per_trajectory",  int64_t(200000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    auto fifo = mdp->GetPolicy("FIFO policy");
    VarGroup rvi_cfg{{"id", std::string("RVI_optimal")}, {"rel_tol", 0.01}, {"silent", int64_t(1)}};
    auto rvi = mdp->GetPolicy(rvi_cfg);

    double fifo_mean = 0.0, rvi_mean = 0.0;
    {
        auto b = comparer.Compare({fifo, rvi});
        b[0].Get("mean", fifo_mean);
        b[1].Get("mean", rvi_mean);
    }
    std::cout << "M/M/1 rho=" << rho << "  FIFO=" << std::fixed << std::setprecision(5) << fifo_mean
              << "  RVI=" << rvi_mean << "  (FIFO should equal RVI)\n";

    // --- PPO training (modest budget; M/M/1 is trivial) ---
    VarGroup ppo_cfg;
    ppo_cfg.Add("num_envs",          int64_t(16));
    ppo_cfg.Add("rollout_steps",     int64_t(128));
    ppo_cfg.Add("num_updates",       int64_t(80));
    ppo_cfg.Add("epochs_per_update", int64_t(4));
    ppo_cfg.Add("mini_batch_size",   int64_t(256));
    ppo_cfg.Add("learning_rate",     3e-4);
    ppo_cfg.Add("gae_gamma",         0.99);
    ppo_cfg.Add("silent",            false);
    VarGroup arch; arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32});
    ppo_cfg.Add("nn_architecture", arch);

    DynaPlex::Policy ppo_policy;
    bool trained = true;
    try {
        auto ppo = dp.GetPPO(mdp, nullptr, ppo_cfg);
        ppo.TrainPolicy();
        ppo_policy = ppo.GetPolicy();
    } catch (const std::exception& e) {
        std::cout << "  [exception during PPO] " << e.what() << "\n";
        trained = false;
    }
    CHECK(trained && ppo_policy != nullptr, "T1: PPO trains and returns a policy");
    if (!trained || !ppo_policy) {
        std::cout << "Results: " << g_pass << " passed, " << (g_fail + 1) << " failed\n";
        return 1;
    }

    // --- T2: convergence ---
    double ppo_mean = 0.0;
    comparer.Compare({ppo_policy})[0].Get("mean", ppo_mean);
    const double ppo_ratio = (rvi_mean > 1e-12) ? ppo_mean / rvi_mean : 1.0;
    std::cout << "PPO=" << ppo_mean << "  PPO/RVI=" << std::setprecision(4) << ppo_ratio << "\n";
    CHECK(ppo_ratio < 1.10, "T2: PPO within 10% of RVI optimum on M/M/1");

    // --- T3: determinism ---
    double ppo_mean2 = 0.0;
    comparer.Compare({ppo_policy})[0].Get("mean", ppo_mean2);
    CHECK(std::abs(ppo_mean - ppo_mean2) < 1e-9, "T3: PPO policy evaluation is deterministic");

    std::cout << "==========================================\n";
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
