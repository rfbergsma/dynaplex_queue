// mm1_baseline.cpp
//
// M/M/1 comparison: Random vs FIFO vs RVI (optimal) vs Neural Network (DCL).
//
// Evaluates using EvaluatePolicyRaw -> mean_cost_per_event
//   = cost / real_event_steps  (FIL-refresh steps NOT in denominator)
// Theoretical value for FIFO (binary cost, D=0): rho^2 / Lambda
//   where Lambda = uniformization_rate = tick_rate + lambda + mu
//
// NN policy trained via DCL from a random starting policy (N=10K, M=100, H=50).
//
// NOTE: raw_mdp must be created AFTER DCL training to avoid a clash with
// VarGroup::Int64Hash() being called twice on the same config object.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

static VarGroup mm1_config(double lam, double mu)
{
    VarGroup srv;
    srv.Add("servers",      int64_t(1));
    srv.Add("can_serve",    VarGroup::Int64Vec{0});
    srv.Add("service_rate", mu);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(1));
    cfg.Add("tick_rate",       1.0);
    cfg.Add("reward_type",     int64_t(0));       // binary cost: 1{FIL > D}
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0});
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0});
    cfg.Add("server_type_0",   srv);
    return cfg;
}

int main()
{
    auto& dp = DynaPlexProvider::Get();
    const double mu = 1.0;

    dp.System() << "\n=== mm1_baseline: Random / FIFO / RVI / NN ===\n";
    dp.System() << "  reward_type=0 (binary cost), D=0, tick_rate=1, mu=1\n";
    dp.System() << "  Evaluator : EvaluatePolicyRaw -> mean_cost_per_event\n";
    dp.System() << "               = cost / real_event_steps  (FIL-refresh NOT in denom)\n";
    dp.System() << "  Theory    : rho^2 / Lambda  (Lambda = tick+lambda+mu)\n";
    dp.System() << "  NN trained via DCL from random policy (N=10K, M=100, H=50)\n\n";

    // Print header
    dp.System() << std::left
                << std::setw(5)  << "rho"
                << std::setw(11) << "Random"
                << std::setw(11) << "FIFO"
                << std::setw(11) << "RVI"
                << std::setw(11) << "NN"
                << std::setw(9)  << "NN/RVI"
                << std::setw(11) << "Theory"
                << std::setw(9)  << "FIFO%err"
                << "\n" << std::string(78, '-') << "\n";

    for (double rho : {0.2, 0.4, 0.6, 0.8})
    {
        double lam = rho * mu;
        auto cfg = mm1_config(lam, mu);

        // Type-erased MDP for DynaPlex framework calls (GetPolicy, GetDCL, etc.)
        auto mdp = dp.GetMDP(cfg);

        // ---- policies ----
        auto random = mdp->GetPolicy("random");
        auto fifo   = mdp->GetPolicy("FIFO policy");

        VarGroup rvi_cfg;
        rvi_cfg.Add("id",      std::string("RVI_optimal"));
        rvi_cfg.Add("rel_tol", 0.01);
        rvi_cfg.Add("silent",  int64_t(1));
        auto rvi = mdp->GetPolicy(rvi_cfg);

        // DCL: train NN from random starting policy
        VarGroup nn_arch;
        nn_arch.Add("type",          std::string("mlp"));
        nn_arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

        VarGroup dcl_cfg;
        dcl_cfg.Add("N",           int64_t(10000));
        dcl_cfg.Add("M",           int64_t(100));
        dcl_cfg.Add("H",           int64_t(50));
        dcl_cfg.Add("num_gens",    int64_t(1));
        dcl_cfg.Add("silent",      true);
        dcl_cfg.Add("nn_architecture", nn_arch);

        auto dcl = dp.GetDCL(mdp, random, dcl_cfg);
        dcl.TrainPolicy();
        auto nn = dcl.GetPolicies().back();

        // Concrete MDP for EvaluatePolicyRaw — created AFTER DCL
        qm::MDP raw_mdp(cfg);

        // ---- evaluate with EvaluatePolicyRaw ----
        auto eval = [&](DynaPlex::Policy pol) {
            return qm::EvaluatePolicyRaw(raw_mdp, pol,
                /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
        };

        auto r_random = eval(random);
        auto r_fifo   = eval(fifo);
        auto r_rvi    = eval(rvi);
        auto r_nn     = eval(nn);

        // ---- theoretical value: rho^2 / Lambda (FIFO, binary cost, D=0) ----
        double theory   = (rho * rho) / raw_mdp.uniformization_rate;
        double fifo_err = (r_fifo.mean_cost_per_event - theory) / theory * 100.0;
        double nn_ratio = (r_rvi.mean_cost_per_event > 1e-12)
                        ? r_nn.mean_cost_per_event / r_rvi.mean_cost_per_event
                        : 1.0;

        dp.System() << std::fixed
                    << std::setw(5)  << std::setprecision(1) << rho
                    << std::setw(11) << std::setprecision(6) << r_random.mean_cost_per_event
                    << std::setw(11) << std::setprecision(6) << r_fifo.mean_cost_per_event
                    << std::setw(11) << std::setprecision(6) << r_rvi.mean_cost_per_event
                    << std::setw(11) << std::setprecision(6) << r_nn.mean_cost_per_event
                    << std::setw(9)  << std::setprecision(4) << nn_ratio
                    << std::setw(11) << std::setprecision(6) << theory
                    << std::setw(8)  << std::setprecision(2) << fifo_err << "%"
                    << "\n";
    }

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
