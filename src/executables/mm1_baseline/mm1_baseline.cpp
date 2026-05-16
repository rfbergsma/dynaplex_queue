// mm1_baseline.cpp
//
// Three experiments for the paper:
//
//   Experiment 1 (sec:mm1)  — M/M/1 validation.
//     Single job type, single server, D=0, mu=1.
//     Metric: physical cost rate = mean_cost_per_event * Lambda  (tick_rate-invariant).
//     Theory: rho^2 / Lambda.  Compared: Random / FIFO / RVI / NN.
//
//   Experiment 2 (sec:exp2) — Two servers, two job types.
//     Configs: simple (symmetric) and simple_asym (asymmetric costs + deadlines).
//     Metric: mean cost per epoch (PolicyComparer, 100 traj x 500K periods).
//
//   Experiment 3 (sec:exp3) — Three servers, three job types, partial flexibility.
//     Config: medium (circular skill sets).
//     Same metric / evaluation as Exp 2.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// ============================================================
// Experiment 1 helpers
// ============================================================

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
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0});
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0});
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
// Experiment 2 / 3 helper — runs one named config end-to-end
// ============================================================

static void run_config_experiment(
    DynaPlex::DynaPlexProvider& dp,
    const std::string& label,
    const std::string& json_file,
    bool   use_rel_tol,      // true  → RVI with rel_tol=0.01
    int64_t rvi_M_fixed,     // used only when use_rel_tol==false
    int64_t N, int64_t H, int64_t M_dcl,
    int64_t reward_type       = int64_t(0),  // 0=binary FIL lateness, 1=queue lateness
    int64_t num_gens          = int64_t(1),
    bool    print_heatmap     = false,       // 2-job configs only
    int64_t early_stop        = int64_t(0))  // 0=disabled; >0 sets early_stopping_patience
{
    // ---- load config & MDP ----
    auto path       = dp.FilePath({"mdp_config_examples", "queue_mdp"}, json_file);
    auto mdp_config = VarGroup::LoadFromFile(path);
    mdp_config.Set("reward_type", reward_type);  // override JSON value
    auto mdp        = dp.GetMDP(mdp_config);

    // ---- policies ----
    auto fifo = mdp->GetPolicy("FIFO policy");

    VarGroup rvi_cfg;
    if (use_rel_tol)
        rvi_cfg = VarGroup{ {"id", std::string("RVI_optimal")}, {"rel_tol", 0.01},
                            {"silent", int64_t(1)} };
    else
        rvi_cfg = VarGroup{ {"id", std::string("RVI_optimal")}, {"M", rvi_M_fixed},
                            {"silent", int64_t(1)} };
    auto rvi = mdp->GetPolicy(rvi_cfg);

    // ---- DCL ----
    VarGroup nn_arch;
    nn_arch.Add("type",          std::string("mlp"));
    nn_arch.Add("hidden_layers", VarGroup::Int64Vec{128, 64, 2});

    VarGroup dcl_cfg;
    dcl_cfg.Add("N",               N);
    dcl_cfg.Add("M",               M_dcl);
    dcl_cfg.Add("H",               H);
    dcl_cfg.Add("num_gens",        num_gens);
    dcl_cfg.Add("silent",          true);
    dcl_cfg.Add("nn_architecture", nn_arch);
    if (early_stop > 0) {
        VarGroup nn_training;
        nn_training.Add("early_stopping_patience", early_stop);
        dcl_cfg.Add("nn_training", nn_training);
    }

    auto dcl = dp.GetDCL(mdp, fifo, dcl_cfg);
    dcl.TrainPolicy();
    auto nn = dcl.GetPolicies().back();

    // ---- evaluate (PolicyComparer) ----
    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(100));
    eval_cfg.Add("periods_per_trajectory",  int64_t(500000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);
    auto res = comparer.Compare({fifo, rvi, nn});

    double fifo_mean = 0.0, rvi_mean = 0.0, nn_mean = 0.0;
    res[0].Get("mean", fifo_mean);
    res[1].Get("mean", rvi_mean);
    res[2].Get("mean", nn_mean);

    double nn_rvi   = (rvi_mean  > 1e-12) ? nn_mean   / rvi_mean  : 1.0;
    double fifo_rvi = (rvi_mean  > 1e-12) ? fifo_mean / rvi_mean  : 1.0;
    double fifo_gap = (fifo_rvi - 1.0) * 100.0;

    dp.System() << std::fixed
                << std::left  << std::setw(14) << label
                << std::right << std::setprecision(6)
                << std::setw(12) << fifo_mean
                << std::setw(12) << rvi_mean
                << std::setw(12) << nn_mean
                << std::setprecision(4)
                << std::setw(10) << nn_rvi
                << std::setw(10) << fifo_rvi
                << std::setprecision(1)
                << std::setw(9)  << fifo_gap << "%"
                << "\n";

    // ---- optional: print policy heatmaps (2-job configs only) ----
    if (print_heatmap) {
        dp.System() << "\n  [" << label << "] FIFO policy heatmap"
                    << " (FIL_0=row, FIL_1=col; 0=serve type0, 1=serve type1, .=skip):\n";
        qm::PrintPolicyHeatmap(mdp, fifo, /*max_fil=*/12);

        dp.System() << "\n  [" << label << "] RVI optimal policy heatmap:\n";
        qm::PrintPolicyHeatmap(mdp, rvi, /*max_fil=*/12);

        dp.System() << "\n  [" << label << "] NN policy heatmap:\n";
        qm::PrintPolicyHeatmap(mdp, nn, /*max_fil=*/12);
        dp.System() << "\n";
    }
}

// ============================================================
// main
// ============================================================

int main()
{
    auto& dp = DynaPlexProvider::Get();
    const double mu = 1.0;

    // ---- Run-control flags: set to false to skip sections ----
    const bool run_exp1 = true;   // M/M/1 validation (fast)
    const bool run_exp2 = true;   // 2-server 2-job configs + heatmaps
    const bool run_exp3 = true;   // 3-server 3-job medium config
    const bool run_exp4 = false;  // reward_type=1, num_gens=3 (slow)

    // ----------------------------------------------------------
    // Experiment 1: M/M/1 validation
    // ----------------------------------------------------------
  if (run_exp1) {
    dp.System() << "\n=== Experiment 1: M/M/1 validation ===\n";
    dp.System() << "  reward_type=0 (binary cost 1{wait>0}), D=0, mu=1\n";
    dp.System() << "  Metric: cost per arrival = (mean_cost_per_event * Lambda) / lambda\n";
    dp.System() << "  Theory: rho  (= P(customer waits) for M/M/1, tick_rate-invariant)\n";
    dp.System() << "  DCL: N=10K, M=400, H=50, num_gens=1, arch={64,32,2}\n";

    for (double tick_rate : {1.0, 2.0, 10.0})
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0)
                    << tick_rate << " ---\n\n";
        print_header(dp);

        for (double rho : {0.2, 0.4, 0.6, 0.8})
        {
            double lam = rho * mu;
            auto cfg   = mm1_config(lam, mu, tick_rate);
            auto mdp   = dp.GetMDP(cfg);

            auto random = mdp->GetPolicy("random");
            auto fifo   = mdp->GetPolicy("FIFO policy");

            VarGroup rvi_cfg;
            rvi_cfg.Add("id",      std::string("RVI_optimal"));
            rvi_cfg.Add("rel_tol", 0.01);
            rvi_cfg.Add("silent",  int64_t(1));
            auto rvi = mdp->GetPolicy(rvi_cfg);

            VarGroup nn_arch;
            nn_arch.Add("type",          std::string("mlp"));
            nn_arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

            VarGroup dcl_cfg;
            dcl_cfg.Add("N",               int64_t(10000));
            dcl_cfg.Add("M",               int64_t(400));
            dcl_cfg.Add("H",               int64_t(50));
            dcl_cfg.Add("num_gens",        int64_t(1));
            dcl_cfg.Add("silent",          true);
            dcl_cfg.Add("nn_architecture", nn_arch);

            auto dcl = dp.GetDCL(mdp, fifo, dcl_cfg);
            dcl.TrainPolicy();
            auto nn = dcl.GetPolicies().back();

            qm::MDP raw_mdp(cfg);
            auto eval = [&](DynaPlex::Policy pol) {
                return qm::EvaluatePolicyRawParallel(raw_mdp, pol,
                    /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
            };

            auto r_random = eval(random);
            auto r_fifo   = eval(fifo);
            auto r_rvi    = eval(rvi);
            auto r_nn     = eval(nn);

            const double Lambda = raw_mdp.uniformization_rate;
            double rate_random = r_random.mean_cost_per_event * Lambda / lam;
            double rate_fifo   = r_fifo.mean_cost_per_event   * Lambda / lam;
            double rate_rvi    = r_rvi.mean_cost_per_event    * Lambda / lam;
            double rate_nn     = r_nn.mean_cost_per_event     * Lambda / lam;

            double theory   = rho;
            double fifo_err = (theory > 1e-15)
                            ? (rate_fifo - theory) / theory * 100.0 : 0.0;
            double nn_ratio = (rate_rvi > 1e-15) ? rate_nn / rate_rvi : 1.0;

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
  } // end run_exp1

    // ----------------------------------------------------------
    // Experiment 2: two servers, two job types
    // ----------------------------------------------------------
  if (run_exp2) {
    dp.System() << "\n\n=== Experiment 2: Two servers, two job types ===\n";
    dp.System() << "  simple     : k=2, n=2, fully flexible, symmetric  (c=[100,100], D=[5,5])\n";
    dp.System() << "  simple_asym: k=2, n=2, fully flexible, asymmetric (c=[100,300], D=[6,3])\n";
    dp.System() << "  Metric: mean cost per epoch (PolicyComparer, 100 traj x 500K periods)\n";
    dp.System() << "  DCL: N=20K, M=1600, H=100, num_gens=1, arch={128,64,2}\n\n";

    dp.System() << std::left
                << std::setw(14) << "Config"
                << std::right
                << std::setw(12) << "FIFO"
                << std::setw(12) << "RVI"
                << std::setw(12) << "NN"
                << std::setw(10) << "NN/RVI"
                << std::setw(10) << "FIFO/RVI"
                << std::setw(10) << "FIFO_gap"
                << "\n" << std::string(80, '-') << "\n";

    run_config_experiment(dp, "simple",
        "mdp_config_simple.json",      /*use_rel_tol=*/true,  /*rvi_M=*/0,
        /*N=*/20000, /*H=*/100, /*M_dcl=*/1600,
        /*reward_type=*/int64_t(0), /*num_gens=*/int64_t(1), /*print_heatmap=*/true,
        /*early_stop=*/int64_t(3));

    run_config_experiment(dp, "simple_asym",
        "mdp_config_simple_asym.json", /*use_rel_tol=*/true,  /*rvi_M=*/0,
        /*N=*/20000, /*H=*/100, /*M_dcl=*/1600,
        /*reward_type=*/int64_t(0), /*num_gens=*/int64_t(1), /*print_heatmap=*/true,
        /*early_stop=*/int64_t(3));
  } // end run_exp2

    // ----------------------------------------------------------
    // Experiment 3: three servers, three job types, partial flexibility
    // ----------------------------------------------------------
  if (run_exp3) {
    dp.System() << "\n\n=== Experiment 3: Three servers, three job types, partial flexibility ===\n";
    dp.System() << "  medium: k=3, n=3, circular skill sets ({0,1},{1,2},{2,0})\n";
    dp.System() << "          c=[100,100,100], D=[6,6,6], rho~0.71\n";
    dp.System() << "  RVI: fixed truncation M=28 (state space too large for auto rel_tol)\n";
    dp.System() << "  DCL: N=20K, M=1600, H=100, num_gens=1, arch={128,64,2}\n\n";

    dp.System() << std::left
                << std::setw(14) << "Config"
                << std::right
                << std::setw(12) << "FIFO"
                << std::setw(12) << "RVI"
                << std::setw(12) << "NN"
                << std::setw(10) << "NN/RVI"
                << std::setw(10) << "FIFO/RVI"
                << std::setw(10) << "FIFO_gap"
                << "\n" << std::string(80, '-') << "\n";

    run_config_experiment(dp, "medium",
        "mdp_config_1.json",           /*use_rel_tol=*/false, /*rvi_M=*/28,
        /*N=*/20000, /*H=*/100, /*M_dcl=*/400,
        /*reward_type=*/int64_t(0), /*num_gens=*/int64_t(1), /*print_heatmap=*/false,
        /*early_stop=*/int64_t(3));
  } // end run_exp3

    // ----------------------------------------------------------
    // Experiment 4: queue-lateness reward (reward_type=1), num_gens=3
    // ----------------------------------------------------------
  if (run_exp4) {
    dp.System() << "\n\n=== Experiment 4: Queue-lateness reward (reward_type=1), num_gens=3 ===\n";
    dp.System() << "  Same configs as Exp 2/3; reward_type=1 (cost proportional to FIL-D per tick)\n";
    dp.System() << "  Hypothesis: richer cost signal + multi-gen bootstrapping improves medium NN.\n";
    dp.System() << "  DCL: N=20K, M=1600, H=100, num_gens=3, arch={128,64,2}\n\n";

    dp.System() << std::left
                << std::setw(14) << "Config"
                << std::right
                << std::setw(12) << "FIFO"
                << std::setw(12) << "RVI"
                << std::setw(12) << "NN"
                << std::setw(10) << "NN/RVI"
                << std::setw(10) << "FIFO/RVI"
                << std::setw(10) << "FIFO_gap"
                << "\n" << std::string(80, '-') << "\n";

    run_config_experiment(dp, "simple",
        "mdp_config_simple.json",      /*use_rel_tol=*/true,  /*rvi_M=*/0,
        /*N=*/int64_t(20000), /*H=*/int64_t(100), /*M_dcl=*/int64_t(1600),
        /*reward_type=*/int64_t(1), /*num_gens=*/int64_t(3));

    run_config_experiment(dp, "simple_asym",
        "mdp_config_simple_asym.json", /*use_rel_tol=*/true,  /*rvi_M=*/0,
        /*N=*/int64_t(20000), /*H=*/int64_t(100), /*M_dcl=*/int64_t(1600),
        /*reward_type=*/int64_t(1), /*num_gens=*/int64_t(3));

    run_config_experiment(dp, "medium",
        "mdp_config_1.json",           /*use_rel_tol=*/false, /*rvi_M=*/28,
        /*N=*/int64_t(20000), /*H=*/int64_t(100), /*M_dcl=*/int64_t(1600),
        /*reward_type=*/int64_t(1), /*num_gens=*/int64_t(3));
  } // end run_exp4

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
