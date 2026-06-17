// ppo_validate.cpp
//
// Validates the PPO implementation on NON-trivial, computationally light, DISCRETE-TIME
// average-cost MDPs with EXACT references. Discrete-time (one decision per period, fixed
// timing) is essential: it removes the semi-MDP variable-duration confound that affects the
// queue MDPs, so a clean result here isolates "is the PPO algorithm correct?".
//
// Problems:
//   1. lost_sales        - classic DCL benchmark; base-stock optimal; ~5 order-quantity actions.
//   2. perishable_systems - inventory with perishability (age-tracked stock); richer state, the
//                           optimum is more than a simple base-stock -> a structurally different test.
//
// Each policy is scored EXACTLY by the exact solver (no simulation noise) against the true optimum.
//   PPO/opt ~ 1.0 on both -> PPO algorithm is sound; the queue Exp2/Exp3 failure is the semi-MDP timing.
//   PPO/opt >> 1 while DCL/opt ~ 1 -> PPO implementation bug.
//
// NOTE: resource_allocation was considered but is event-driven (arrival rates -> a semi-MDP),
// so it would confound bug vs. timing and is intentionally excluded from this correctness test.

#include <iostream>
#include <iomanip>
#include <string>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"

using namespace DynaPlex;

static void run_validation(DynaPlexProvider& dp, const std::string& label,
                           const VarGroup& cfg, const std::string& ref_policy_id)
{
    std::cout << "\n================ " << label << " ================\n";
    auto mdp = dp.GetMDP(cfg);

    auto solver = dp.GetExactSolver(mdp, { {"silent", true}, {"max_states", int64_t(2000000)} });
    const double opt = solver.ComputeCosts(true);
    auto ref = mdp->GetPolicy(ref_policy_id);
    const double ref_cost = solver.ComputeCosts(true, ref);

    std::cout << std::fixed << std::setprecision(5)
              << "optimal     = " << opt << "\n"
              << ref_policy_id << std::string(std::max<int>(1, 12 - (int)ref_policy_id.size()), ' ')
              << "= " << ref_cost << "   (" << ref_policy_id << "/opt = "
              << std::setprecision(4) << ref_cost / opt << ")\n";

    // DCL reference (trusted method)
    {
        VarGroup dcl_cfg;
        dcl_cfg.Add("N",         int64_t(5000));
        dcl_cfg.Add("M",         int64_t(500));
        dcl_cfg.Add("H",         int64_t(40));
        dcl_cfg.Add("num_gens",  int64_t(2));
        dcl_cfg.Add("silent",    true);
        VarGroup arch; arch.Add("type", std::string("mlp"));
        arch.Add("hidden_layers", VarGroup::Int64Vec{64, 64});
        dcl_cfg.Add("nn_architecture", arch);

        auto dcl = dp.GetDCL(mdp, ref, dcl_cfg);
        dcl.TrainPolicy();
        auto dcl_nn = dcl.GetPolicies().back();
        const double dcl_cost = solver.ComputeCosts(true, dcl_nn);
        std::cout << std::setprecision(5)
                  << "DCL         = " << dcl_cost << "   (DCL/opt = "
                  << std::setprecision(4) << dcl_cost / opt << ")\n";
    }

    // PPO (under test)
    {
        VarGroup ppo_cfg;
        ppo_cfg.Add("num_envs",          int64_t(16));
        ppo_cfg.Add("rollout_steps",     int64_t(128));
        ppo_cfg.Add("num_updates",       int64_t(300));
        ppo_cfg.Add("epochs_per_update", int64_t(4));
        ppo_cfg.Add("mini_batch_size",   int64_t(256));
        ppo_cfg.Add("learning_rate",     3e-4);
        ppo_cfg.Add("gae_gamma",         0.99);
        ppo_cfg.Add("entropy_coef",      0.01);
        ppo_cfg.Add("rng_seed",          int64_t(1));
        ppo_cfg.Add("silent",            true);
        VarGroup arch; arch.Add("hidden_layers", VarGroup::Int64Vec{64, 64});
        ppo_cfg.Add("nn_architecture", arch);

        auto ppo = dp.GetPPO(mdp, nullptr, ppo_cfg);
        ppo.TrainPolicy();
        auto ppo_nn = ppo.GetPolicy();
        const double ppo_cost = solver.ComputeCosts(true, ppo_nn);
        std::cout << std::setprecision(5)
                  << "PPO         = " << ppo_cost << "   (PPO/opt = "
                  << std::setprecision(4) << ppo_cost / opt << ")\n";
    }
}

int main() {
    auto& dp = DynaPlexProvider::Get();
    std::cout << "================ ppo_validate ================\n";

    // 1. lost_sales
    {
        VarGroup cfg;
        cfg.Add("id",              std::string("lost_sales"));
        cfg.Add("h",               1.0);
        cfg.Add("p",               4.0);
        cfg.Add("leadtime",        int64_t(2));
        cfg.Add("discount_factor", 1.0);
        cfg.Add("demand_dist",     VarGroup({ {"type", std::string("geometric")}, {"mean", 3.0} }));
        run_validation(dp, "lost_sales", cfg, "base_stock");
    }

    // 2. perishable_systems (age-tracked inventory; richer state than lost_sales)
    {
        VarGroup cfg;
        cfg.Add("id",              std::string("perishable_systems"));
        cfg.Add("o",               100.0);
        cfg.Add("h",               0.0);
        cfg.Add("c",               0.0);
        cfg.Add("p",               100.0);
        cfg.Add("mu",              4.0);
        cfg.Add("cvr",             1.0);
        cfg.Add("f",               1.0);
        cfg.Add("LeadTime",        int64_t(1));
        cfg.Add("ProductLife",     int64_t(3));
        cfg.Add("discount_factor", 1.0);
        run_validation(dp, "perishable_systems", cfg, "base_stock");
    }

    std::cout << "\n==============================================\n";
    return 0;
}
