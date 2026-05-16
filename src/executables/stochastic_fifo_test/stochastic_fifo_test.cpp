#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// run_experiment
//
// For a single MDP config:
//   1. Evaluate FIFO and RVI as benchmarks.
//   2. For each epsilon in thresholds:
//        a. Evaluate StochasticFIFO(eps) as a standalone policy.
//        b. Train DCL with StochasticFIFO(eps) as the base policy.
//        c. Evaluate the trained NN.
//   3. Print a summary table.
// ---------------------------------------------------------------------------
static void run_experiment(
    DynaPlex::DynaPlexProvider& dp,
    const std::string&          label,
    const std::string&          config_file,
    const std::vector<double>&  thresholds,
    bool                        use_rel_tol,
    int64_t                     rvi_M_fixed)
{
    // --- Load MDP ---
    auto path       = dp.FilePath({"mdp_config_examples", "queue_mdp"}, config_file);
    auto mdp_config = VarGroup::LoadFromFile(path);
    auto mdp        = dp.GetMDP(mdp_config);

    // --- Evaluation settings ---
    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(100));
    eval_cfg.Add("periods_per_trajectory", int64_t(500000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    // --- DCL settings (shared across all runs) ---
    VarGroup nn_arch{
        {"type", "mlp"},
        {"hidden_layers", VarGroup::Int64Vec{64, 32, 2}}
    };
    VarGroup nn_training;
    nn_training.Add("early_stopping_patience", int64_t(3));

    auto make_dcl_config = [&]() {
        VarGroup dcl;
        dcl.Add("N",             int64_t(20000));
        dcl.Add("M",             int64_t(400));
        dcl.Add("H",             int64_t(100));
        dcl.Add("num_gens",      int64_t(1));
        dcl.Add("silent",        true);
        dcl.Add("nn_architecture", nn_arch);
        dcl.Add("nn_training",     nn_training);
        return dcl;
    };

    // --- Benchmarks: FIFO and RVI ---
    auto fifo = mdp->GetPolicy("FIFO policy");

    VarGroup rvi_cfg;
    rvi_cfg.Add("id",     std::string("RVI_optimal"));
    rvi_cfg.Add("silent", int64_t(1));
    if (use_rel_tol)
        rvi_cfg.Add("rel_tol", 0.01);
    else
        rvi_cfg.Add("M", rvi_M_fixed);
    auto rvi = mdp->GetPolicy(rvi_cfg);

    auto bench = comparer.Compare({fifo, rvi});
    double fifo_mean = 0.0, rvi_mean = 0.0;
    bench[0].Get("mean", fifo_mean);
    bench[1].Get("mean", rvi_mean);

    // --- Header ---
    dp.System() << "\n=== " << label << " ===\n";
    dp.System() << "  Config: " << config_file << "\n";
    dp.System() << "  FIFO = " << std::fixed << std::setprecision(4) << fifo_mean
                << "  |  RVI = " << rvi_mean
                << "  |  FIFO/RVI = " << std::setprecision(4) << fifo_mean / rvi_mean
                << "  (" << std::setprecision(1) << (fifo_mean / rvi_mean - 1.0) * 100.0
                << "% gap)\n\n";

    dp.System() << std::left
        << std::setw(32) << "Policy / Base"
        << std::setw(14) << "Direct mean"
        << std::setw(12) << "Direct/RVI"
        << std::setw(14) << "NN mean"
        << std::setw(10) << "NN/RVI"
        << "\n" << std::string(82, '-') << "\n";

    auto print_row = [&](const std::string& base_label,
                         double             base_mean,
                         double             nn_mean)
    {
        dp.System() << std::left  << std::setw(32) << base_label
                    << std::fixed << std::setprecision(4)
                    << std::setw(14) << base_mean
                    << std::setw(12) << base_mean / rvi_mean
                    << std::setw(14) << nn_mean
                    << std::setw(10) << nn_mean / rvi_mean
                    << "\n";
    };

    // --- FIFO base (plain DCL, no stochastic) ---
    {
        auto dcl = dp.GetDCL(mdp, fifo, make_dcl_config());
        dcl.TrainPolicy();
        auto nn  = dcl.GetPolicies().back();
        auto res = comparer.Compare({nn});
        double nn_mean = 0.0;
        res[0].Get("mean", nn_mean);
        print_row("FIFO (eps=0.0)", fifo_mean, nn_mean);
    }

    // --- StochasticFIFO variants ---
    for (double eps : thresholds) {
        VarGroup stoch_cfg;
        stoch_cfg.Add("id",        std::string("stochastic_FIFO"));
        stoch_cfg.Add("threshold", eps);
        auto stoch = mdp->GetPolicy(stoch_cfg);

        // Evaluate base policy directly
        auto base_res = comparer.Compare({stoch});
        double base_mean = 0.0;
        base_res[0].Get("mean", base_mean);

        // Train DCL from this base
        auto dcl = dp.GetDCL(mdp, stoch, make_dcl_config());
        dcl.TrainPolicy();
        auto nn  = dcl.GetPolicies().back();
        auto res = comparer.Compare({nn});
        double nn_mean = 0.0;
        res[0].Get("mean", nn_mean);

        std::ostringstream lbl;
        lbl << "StochFIFO(eps=" << std::fixed << std::setprecision(2) << eps << ")";
        print_row(lbl.str(), base_mean, nn_mean);
    }

    dp.System() << "\n";
}

// ---------------------------------------------------------------------------
int main()
{
    auto& dp = DynaPlexProvider::Get();

    dp.System() << "\n=== Stochastic FIFO Exploration Test ===\n";
    dp.System() << "Two configs: asymmetric-cost (fully flexible) and asymmetric-flexibility\n";
    dp.System() << "For each: FIFO base vs StochasticFIFO(eps) base for DCL training\n";
    dp.System() << "Goal: does injecting eps% random skips improve NN/RVI?\n\n";

    // Thresholds to test: 0.05, 0.10, 0.20, 0.30
    std::vector<double> thresholds = {0.05, 0.10, 0.20, 0.30};

    // --- Config 1: fully flexible, asymmetric costs ---
    run_experiment(dp,
        "asym_cost_2s: fully flexible, cost=[100,300], D=[6,3]",
        "mdp_config_asym_cost_2s.json",
        thresholds,
        /*use_rel_tol=*/true, /*rvi_M_fixed=*/0);

    // --- Config 2: specialist + generalist servers ---
    run_experiment(dp,
        "flex_2s: specialist server_0={0}, generalist server_1={0,1}",
        "mdp_config_flex_2s.json",
        thresholds,
        /*use_rel_tol=*/true, /*rvi_M_fixed=*/0);

    dp.System() << "=== DONE ===\n";
    return 0;
}
