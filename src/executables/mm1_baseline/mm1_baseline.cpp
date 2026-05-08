// mm1_baseline.cpp
//
// Minimal M/M/1 sanity-check: single server, single job type,
// due_date=0, cost_rate=1, QL reward.
//
// With D=0 every waiting job contributes its full age as cost.
// The time-average of the sum of ages is:
//
//   E[cost / real-time] = rho^2 / (mu*(1-rho)^2)
//                       = E[L_q] * E[W_q | job waits]
//
// This is NOT E[L_q] = rho^2/(1-rho) (off by factor 1/(mu*(1-rho))).
//
// g* from RVI is the same quantity scaled by tick_rate/uniformization_rate.
// It must equal mean_cost_per_step_gic from simulation (< ~2% gap).

#include <iostream>
#include <iomanip>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
using RawMDP = DynaPlex::Models::queue_mdp::MDP;

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
    cfg.Add("reward_type",     int64_t(1));       // QL: cost = sum of ages
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

    dp.System() << "\n=== mm1_baseline: D=0, cost_rate=1, QL reward ===\n";
    dp.System() << "  g* (RVI) must match sim_gic  (<~2%)\n";
    dp.System() << "  E[L_q] = rho^2/(1-rho)  is NOT equal to g*\n\n";

    dp.System() << std::left
                << std::setw(6)  << "rho"
                << std::setw(12) << "E[L_q]"
                << std::setw(12) << "g*(RVI)"
                << std::setw(14) << "sim_gic"
                << std::setw(10) << "g*/E[L_q]"
                << "\n" << std::string(54, '-') << "\n";

    for (double rho : {0.2, 0.4, 0.6, 0.8})
    {
        double lam = rho * mu;
        RawMDP mdp(mm1_config(lam, mu));

        auto sol = mdp.runRVI(0.01);

        auto fifo_fn = [](const RawMDP::State&) -> int64_t { return 1; };
        auto raw = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
            mdp, fifo_fn, /*n_traj=*/200, /*steps=*/100000, /*warmup=*/10000);

        double Lq    = rho * rho / (1.0 - rho);
        double ratio = sol.g_star / Lq;

        dp.System() << std::fixed
                    << std::setw(6)  << std::setprecision(1) << rho
                    << std::setw(12) << std::setprecision(6) << Lq
                    << std::setw(12) << std::setprecision(6) << sol.g_star
                    << std::setw(14) << std::setprecision(6) << raw.mean_cost_per_step_gic
                    << std::setw(10) << std::setprecision(4) << ratio
                    << "\n";
    }

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
