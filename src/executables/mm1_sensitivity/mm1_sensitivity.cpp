// mm1_sensitivity.cpp
//
// DCL hyperparameter grid search on the strategic-idleness problem:
//   2 types, 1 pool of 2 servers, rho=0.6, c1=1, c2=20, tick_rate=5
//   Grid: D x N x H x M x arch  (108 runs)
//   Metric: physical_cost_rate = mean_cost_per_event * Lambda  (tick_rate-invariant)
//
// RVI (optimal) and FIFO are evaluated once per D.
// DCL training statistics are printed in full (silent not set).
// Summary table sorted by NN/RVI ratio within each D group.

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>
#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// 2-type, 2-server config with variable due time and cost asymmetry.
// due is in real-time seconds; constructor multiplies by tick_rate.
static VarGroup sens_config(double c2, double due, double tick_rate)
{
    VarGroup srv;
    srv.Add("servers",      int64_t(2));
    srv.Add("can_serve",    VarGroup::Int64Vec{0, 1});   // fully flexible
    srv.Add("service_rate", 1.0);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{0.6, 0.6});   // rho=0.6
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0, c2});
    cfg.Add("due_times",       VarGroup::DoubleVec{due, due});
    cfg.Add("server_type_0",   srv);
    return cfg;
}

struct RunResult {
    double      D;
    int64_t     N, H, M;
    std::string arch_name;
    double      rate_fifo, rate_rvi, rate_nn;
    double      nn_rvi;
};

int main()
{
    auto& dp = DynaPlexProvider::Get();

    dp.System() << "\n=== mm1_sensitivity: DCL Hyperparameter Grid Search ===\n";
    dp.System() << "  Fixed : 2 types, 2 servers, rho=0.6, c1=1, c2=20, tick_rate=5\n";
    dp.System() << "  Grid  : D={0,2,5} x N={5K,20K,100K} x H={50,100,200}"
                   " x M={400,1600} x arch={shallow,deep}  -> 108 runs\n";
    dp.System() << "  Metric: physical_cost_rate = mean_cost_per_event * Lambda\n";

    const double tick_rate = 5.0;
    const double c2        = 20.0;

    const std::vector<double>  D_vals = {0.0, 2.0, 5.0};
    const std::vector<int64_t> N_vals = {5000, 20000, 100000};
    const std::vector<int64_t> H_vals = {50, 100, 200};
    const std::vector<int64_t> M_vals = {400, 1600};

    struct Arch { std::string name; VarGroup::Int64Vec layers; };
    const std::vector<Arch> archs = {
        {"shallow", VarGroup::Int64Vec{128, 64, 2}},
        {"deep",    VarGroup::Int64Vec{128, 64, 32, 2}}
    };

    auto eval_par = [&](qm::MDP& raw_mdp, DynaPlex::Policy pol) {
        return qm::EvaluatePolicyRawParallel(raw_mdp, pol,
            /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
    };

    std::vector<RunResult> results;
    int run_idx = 0;
    const int total_runs = static_cast<int>(D_vals.size() * N_vals.size() *
                                            H_vals.size() * M_vals.size() * archs.size());

    for (double D : D_vals)
    {
        dp.System() << "\n\n============================================================\n";
        dp.System() << "  D = " << std::fixed << std::setprecision(0) << D
                    << "  (FIL threshold = " << static_cast<int>(D * tick_rate) << " ticks)\n";
        dp.System() << "============================================================\n";

        auto cfg         = sens_config(c2, D, tick_rate);
        auto mdp         = dp.GetMDP(cfg);
        qm::MDP raw_mdp(cfg);
        const double Lambda = raw_mdp.uniformization_rate;

        // ---- FIFO baseline (once per D) ----
        auto fifo    = mdp->GetPolicy("FIFO policy");
        double rate_fifo = eval_par(raw_mdp, fifo).mean_cost_per_event * Lambda;

        // ---- RVI optimal (once per D) ----
        VarGroup rvi_cfg;
        rvi_cfg.Add("id",      std::string("RVI_optimal"));
        rvi_cfg.Add("rel_tol", 0.01);
        rvi_cfg.Add("silent",  int64_t(1));
        auto rvi = mdp->GetPolicy(rvi_cfg);
        double rate_rvi  = eval_par(raw_mdp, rvi).mean_cost_per_event * Lambda;

        double fifo_rvi_base = (rate_rvi > 1e-15) ? rate_fifo / rate_rvi : 1.0;
        dp.System() << "\n  Baselines:  FIFO=" << std::setprecision(6) << rate_fifo
                    << "   RVI=" << rate_rvi
                    << "   FIFO/RVI=" << std::setprecision(4) << fifo_rvi_base << "\n";

        // ---- Grid search ----
        for (int64_t N : N_vals)
        for (int64_t H : H_vals)
        for (int64_t M : M_vals)
        for (const auto& arch : archs)
        {
            ++run_idx;
            dp.System() << "\n--- Run " << run_idx << "/" << total_runs
                        << "  D=" << std::setprecision(0) << D
                        << "  N=" << N << "  H=" << H
                        << "  M=" << M << "  arch=" << arch.name << " ---\n";

            VarGroup nn_arch;
            nn_arch.Add("type",          std::string("mlp"));
            nn_arch.Add("hidden_layers", arch.layers);

            VarGroup dcl_cfg;
            dcl_cfg.Add("N",               N);
            dcl_cfg.Add("M",               M);
            dcl_cfg.Add("H",               H);
            dcl_cfg.Add("num_gens",        int64_t(1));
            dcl_cfg.Add("nn_architecture", nn_arch);
            // silent NOT set -> DynaPlex prints full training statistics

            auto dcl = dp.GetDCL(mdp, fifo, dcl_cfg);
            dcl.TrainPolicy();
            auto nn  = dcl.GetPolicies().back();

            double rate_nn  = eval_par(raw_mdp, nn).mean_cost_per_event * Lambda;
            double nn_ratio = (rate_rvi > 1e-15) ? rate_nn / rate_rvi : 1.0;

            dp.System() << "  Result:  NN=" << std::setprecision(6) << rate_nn
                        << "  NN/RVI=" << std::setprecision(4) << nn_ratio
                        << "  (FIFO/RVI=" << fifo_rvi_base << ")\n";

            results.push_back({D, N, H, M, arch.name,
                               rate_fifo, rate_rvi, rate_nn, nn_ratio});
        }
    }

    // ---- Summary: sorted by NN/RVI within each D group ----
    std::stable_sort(results.begin(), results.end(),
        [](const RunResult& a, const RunResult& b) {
            if (a.D != b.D) return a.D < b.D;
            return a.nn_rvi < b.nn_rvi;
        });

    dp.System() << "\n\n=== SUMMARY (sorted by NN/RVI, lower = better) ===\n";
    dp.System() << std::left
                << std::setw(4)  << "D"
                << std::setw(8)  << "N"
                << std::setw(6)  << "H"
                << std::setw(7)  << "M"
                << std::setw(10) << "arch"
                << std::right
                << std::setw(9)  << "NN/RVI"
                << std::setw(10) << "FIFO/RVI"
                << std::setw(12) << "NN_rate"
                << std::setw(12) << "FIFO_rate"
                << std::setw(12) << "RVI_rate"
                << "\n" << std::string(90, '-') << "\n";

    double prev_D = -1.0;
    for (const auto& r : results)
    {
        if (r.D != prev_D) {
            dp.System() << "\n D=" << std::fixed << std::setprecision(0) << r.D
                        << "  (threshold=" << static_cast<int>(r.D * tick_rate) << " ticks)\n";
            prev_D = r.D;
        }
        double fifo_rvi = (r.rate_rvi > 1e-15) ? r.rate_fifo / r.rate_rvi : 1.0;
        dp.System() << std::fixed
                    << std::right
                    << std::setw(4)  << std::setprecision(0) << r.D
                    << std::setw(8)  << r.N
                    << std::setw(6)  << r.H
                    << std::setw(7)  << r.M
                    << "  " << std::left << std::setw(8) << r.arch_name
                    << std::right
                    << std::setw(9)  << std::setprecision(4) << r.nn_rvi
                    << std::setw(10) << std::setprecision(4) << fifo_rvi
                    << std::setw(12) << std::setprecision(6) << r.rate_nn
                    << std::setw(12) << std::setprecision(6) << r.rate_fifo
                    << std::setw(12) << std::setprecision(6) << r.rate_rvi
                    << "\n";
    }

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
