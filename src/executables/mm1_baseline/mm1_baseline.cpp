// mm1_baseline.cpp
//
// M/M/1 comparison: Random vs FIFO vs RVI (optimal) vs Neural Network (DCL).
// Runs for multiple tick_rates to verify it is a pure granularity parameter.
//
// Evaluates with EvaluatePolicyRawParallel, then RESCALES:
//   physical_cost_rate = mean_cost_per_event * Lambda    (cost per unit real-time)
// This rescaled metric is tick_rate-invariant.  Theory for M/M/1 with
// binary cost 1{wait>0}, D=0:  physical_cost_rate = rho^2.
//
// NN trained via DCL from a FIFO starting policy (better signal than random).

#include <iostream>
#include <iomanip>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

static VarGroup mm1_config(double lam, double mu, double tick_rate)
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
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     int64_t(0));       // binary cost: 1{FIL > D}
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0});   // real-time units; constructor divides by tick_rate
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0});   // real-time seconds; constructor multiplies by tick_rate
    cfg.Add("server_type_0",   srv);
    return cfg;
}

static void print_header(DynaPlex::DynaPlexProvider& dp)
{
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
}

// ============================================================
// Experiment 2 helpers — 2 job types, 1 pool of 2 servers
// ============================================================

// Config: rho=0.6, lambda1=lambda2=0.6, mu=1, D=0 (cost on any wait)
// c1=1 fixed; c2 and tick_rate are the parameters.
static VarGroup si_config(double c2, double tick_rate)
{
    VarGroup srv;
    srv.Add("servers",      int64_t(2));
    srv.Add("can_serve",    VarGroup::Int64Vec{0, 1});   // fully flexible
    srv.Add("service_rate", 1.0);                         // same rate both types

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{0.6, 0.6});  // rho = 1.2/2 = 0.6
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0, c2});   // real-time units
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0, 0.0});  // D=0: cost on any wait
    cfg.Add("server_type_0",   srv);
    return cfg;
}

int main()
{
    auto& dp = DynaPlexProvider::Get();
    const double mu = 1.0;

    dp.System() << "\n=== mm1_baseline: Random / FIFO / RVI / NN ===\n";
    dp.System() << "  reward_type=0 (binary cost 1{wait>0}), D=0, mu=1\n";
    dp.System() << "  Displayed metric: physical cost rate = mean_cost_per_event * Lambda\n";
    dp.System() << "  Theory (M/M/1): rho^2  (tick_rate-invariant)\n";
    dp.System() << "  NN trained via DCL from FIFO policy (N=10K, M=100, H=50)\n";

    for (double tick_rate : {1.0, 2.0, 10.0})
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0) << tick_rate
                    << "  (Lambda = tick_rate + lambda + mu) ---\n\n";
        print_header(dp);

        for (double rho : {0.2, 0.4, 0.6, 0.8})
        {
            double lam = rho * mu;
            auto cfg = mm1_config(lam, mu, tick_rate);

            // Type-erased MDP for DynaPlex framework calls
            auto mdp = dp.GetMDP(cfg);

            // ---- policies ----
            auto random = mdp->GetPolicy("random");
            auto fifo   = mdp->GetPolicy("FIFO policy");

            VarGroup rvi_cfg;
            rvi_cfg.Add("id",      std::string("RVI_optimal"));
            rvi_cfg.Add("rel_tol", 0.01);
            rvi_cfg.Add("silent",  int64_t(1));
            auto rvi = mdp->GetPolicy(rvi_cfg);

            // DCL: train NN from FIFO starting policy (better signal than random)
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

            auto dcl = dp.GetDCL(mdp, fifo, dcl_cfg);
            dcl.TrainPolicy();
            auto nn = dcl.GetPolicies().back();

            // Concrete MDP for evaluation — created AFTER DCL
            qm::MDP raw_mdp(cfg);

            // ---- parallel evaluation ----
            auto eval = [&](DynaPlex::Policy pol) {
                return qm::EvaluatePolicyRawParallel(raw_mdp, pol,
                    /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
            };

            auto r_random = eval(random);
            auto r_fifo   = eval(fifo);
            auto r_rvi    = eval(rvi);
            auto r_nn     = eval(nn);

            // Rescale: physical cost rate = mean_cost_per_event * Lambda (tick_rate-invariant).
            // Theory for M/M/1 with binary cost 1{wait>0}, D=0: physical_cost_rate = rho^2.
            const double Lambda = raw_mdp.uniformization_rate;
            double rate_random = r_random.mean_cost_per_event * Lambda;
            double rate_fifo   = r_fifo.mean_cost_per_event   * Lambda;
            double rate_rvi    = r_rvi.mean_cost_per_event    * Lambda;
            double rate_nn     = r_nn.mean_cost_per_event     * Lambda;

            double theory   = rho * rho;
            double fifo_err = (theory > 1e-15)
                            ? (rate_fifo - theory) / theory * 100.0
                            : 0.0;
            double nn_ratio = (rate_rvi > 1e-15)
                            ? rate_nn / rate_rvi
                            : 1.0;

            dp.System() << std::fixed
                        << std::setw(5)  << std::setprecision(1) << rho
                        << std::setw(11) << std::setprecision(6) << rate_random
                        << std::setw(11) << std::setprecision(6) << rate_fifo
                        << std::setw(11) << std::setprecision(6) << rate_rvi
                        << std::setw(11) << std::setprecision(6) << rate_nn
                        << std::setw(9)  << std::setprecision(4) << nn_ratio
                        << std::setw(11) << std::setprecision(6) << theory
                        << std::setw(8)  << std::setprecision(2) << fifo_err << "%"
                        << "\n";
        }
    }

    // ================================================================
    // Experiment 2: Strategic Idleness
    // 2 job types, 1 pool of 2 fully-flexible servers, rho=0.6, D=7
    // c1=1 fixed; vary c2 in {1,2,5,10,20}; tick_rate=5 (test phase)
    // ================================================================
    dp.System() << "\n\n=== Exp 2: Strategic Idleness (2 types, 2 servers, rho=0.6, D=0) ===\n";
    dp.System() << "  c1=1 fixed, mu=1, D=0, rho=0.6 (lambda1=lambda2=0.6)\n";
    dp.System() << "  Displayed metric: physical cost rate = mean_cost_per_event * Lambda\n";
    dp.System() << "  NN: DCL from FIFO policy (N=20K, M=1600, H=100, num_gens=3)\n";
    dp.System() << "  RVI: optimal solver (rel_tol=0.01, silent)\n";

    // State saved for heatmaps (tick_rate=5, c2=20)
    bool             have_saved2  = false;
    VarGroup         saved_cfg2;
    DynaPlex::Policy saved_fifo_si;
    DynaPlex::Policy saved_rvi_si;
    DynaPlex::Policy saved_nn_si;

    double tr2 = 5.0;  // Test with tick_rate=5 only
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0)
                    << tr2 << " ---\n\n";
        dp.System() << std::left
                    << std::setw(6)  << "c2"
                    << std::setw(11) << "FIFO"
                    << std::setw(11) << "RVI"
                    << std::setw(11) << "NN"
                    << std::setw(9)  << "NN/RVI"
                    << "\n" << std::string(48, '-') << "\n";

        for (double c2 : {1.0, 2.0, 5.0, 10.0, 20.0})
        {
            auto cfg2    = si_config(c2, tr2);
            auto mdp2    = dp.GetMDP(cfg2);
            qm::MDP raw_mdp2(cfg2);

            auto fifo2   = mdp2->GetPolicy("FIFO policy");

            // RVI: optimal solver
            VarGroup rvi_cfg2;
            rvi_cfg2.Add("id",      std::string("RVI_optimal"));
            rvi_cfg2.Add("rel_tol", 0.01);
            rvi_cfg2.Add("silent",  int64_t(1));
            auto rvi2 = mdp2->GetPolicy(rvi_cfg2);

            // DCL: train NN from FIFO starting policy (better signal than random)
            VarGroup nn_arch2;
            nn_arch2.Add("type",          std::string("mlp"));
            nn_arch2.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

            VarGroup dcl_cfg2;
            dcl_cfg2.Add("N",               int64_t(20000));
            dcl_cfg2.Add("M",               int64_t(1600));  // high M key for asymmetric problems
            dcl_cfg2.Add("H",               int64_t(100));   // shorter H → smaller Q values → faster training
            dcl_cfg2.Add("num_gens",        int64_t(3));
            dcl_cfg2.Add("silent",          true);
            dcl_cfg2.Add("nn_architecture", nn_arch2);

            auto dcl2 = dp.GetDCL(mdp2, fifo2, dcl_cfg2);
            dcl2.TrainPolicy();
            auto nn2 = dcl2.GetPolicies().back();

            auto eval2 = [&](DynaPlex::Policy pol) {
                return qm::EvaluatePolicyRawParallel(raw_mdp2, pol,
                    /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
            };

            auto r_fifo2 = eval2(fifo2);
            auto r_rvi2  = eval2(rvi2);
            auto r_nn2   = eval2(nn2);

            // Rescale to physical cost rate (tick_rate-invariant)
            const double Lambda2 = raw_mdp2.uniformization_rate;
            double rate_fifo = r_fifo2.mean_cost_per_event * Lambda2;
            double rate_rvi  = r_rvi2.mean_cost_per_event  * Lambda2;
            double rate_nn   = r_nn2.mean_cost_per_event   * Lambda2;

            double ratio2 = (rate_rvi > 1e-15) ? rate_nn / rate_rvi : 1.0;

            dp.System() << std::fixed << std::right
                        << std::setw(5)  << std::setprecision(0) << c2  << " "
                        << std::setw(11) << std::setprecision(6) << rate_fifo
                        << std::setw(11) << std::setprecision(6) << rate_rvi
                        << std::setw(11) << std::setprecision(6) << rate_nn
                        << std::setw(9)  << std::setprecision(4) << ratio2
                        << "\n";

            // Save tick_rate=5, c2=20 for heatmaps
            if (std::abs(c2 - 20.0) < 1e-9)
            {
                saved_cfg2    = cfg2;
                saved_fifo_si = fifo2;
                saved_rvi_si  = rvi2;
                saved_nn_si   = nn2;
                have_saved2   = true;
            }
        }
    }

    // ---- Heatmaps for tick_rate=5, c2=20 ----
    if (have_saved2)
    {
        constexpr int HEAT_MAX = 12;
        auto mdp_hm = dp.GetMDP(saved_cfg2);

        dp.System() << "\n\n=== Heatmaps: tick_rate=5, c2=20, rho=0.6, D=0 ===\n";
        dp.System() << "(simulation-based; canonical: 1 server busy on type-0; 1 job of each type waiting)\n";

        dp.System() << "\n--- FIFO Policy ---\n";
        qm::PrintPolicyHeatmap(mdp_hm, saved_fifo_si, HEAT_MAX,
                               /*n_warmup=*/10000, /*n_samples=*/100000);

        dp.System() << "\n--- RVI (Optimal) Policy ---\n";
        qm::PrintPolicyHeatmap(mdp_hm, saved_rvi_si, HEAT_MAX,
                               /*n_warmup=*/10000, /*n_samples=*/100000);

        dp.System() << "\n--- NN (DCL, N=20K, M=200) Policy ---\n";
        qm::PrintPolicyHeatmap(mdp_hm, saved_nn_si, HEAT_MAX,
                               /*n_warmup=*/10000, /*n_samples=*/100000);
    }

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
