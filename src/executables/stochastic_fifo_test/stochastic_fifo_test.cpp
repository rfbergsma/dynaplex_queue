#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// Shared DCL config builder
// ---------------------------------------------------------------------------
static VarGroup make_dcl(int64_t N, int64_t M, int64_t H, int64_t num_gens,
                         const VarGroup& nn_arch)
{
    VarGroup nn_training;
    nn_training.Add("early_stopping_patience", int64_t(3));

    VarGroup dcl;
    dcl.Add("N",               N);
    dcl.Add("M",               M);
    dcl.Add("H",               H);
    dcl.Add("num_gens",        num_gens);
    dcl.Add("silent",          true);
    dcl.Add("nn_architecture", nn_arch);
    dcl.Add("nn_training",     nn_training);
    return dcl;
}

// ---------------------------------------------------------------------------
// run_one_config
//
// Runs one (MDP config, tick_rate) block:
//  - Evaluates FIFO and RVI as benchmarks (RVI skipped if !compute_rvi).
//  - FIFO base: trains num_gens_fifo gens, reports every gen.
//  - For each eps in epsilons: trains num_gens gens, reports every gen.
//
// H should already be set to 100 * tick_rate by the caller.
// ---------------------------------------------------------------------------
static void run_one_config(
    DynaPlex::DynaPlexProvider& dp,
    const std::string&           section_label,
    VarGroup                     mdp_config,     // tick_rate already overridden
    int64_t                      N,
    int64_t                      M,
    int64_t                      H,
    int64_t                      num_gens,       // for stochastic base
    int64_t                      num_gens_fifo,  // for FIFO base (usually 1)
    const std::vector<double>&   epsilons,
    bool                         compute_rvi,
    const VarGroup&              nn_arch)
{
    auto mdp = dp.GetMDP(mdp_config);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(100));
    eval_cfg.Add("periods_per_trajectory", int64_t(500000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    // --- Benchmarks ---
    auto fifo = mdp->GetPolicy("FIFO policy");
    double fifo_mean = 0.0, rvi_mean = 0.0;

    if (compute_rvi) {
        VarGroup rvi_cfg;
        rvi_cfg.Add("id",      std::string("RVI_optimal"));
        rvi_cfg.Add("silent",  int64_t(1));
        rvi_cfg.Add("rel_tol", 0.01);
        auto rvi    = mdp->GetPolicy(rvi_cfg);
        auto bench  = comparer.Compare({fifo, rvi});
        bench[0].Get("mean", fifo_mean);
        bench[1].Get("mean", rvi_mean);
    } else {
        auto bench = comparer.Compare({fifo});
        bench[0].Get("mean", fifo_mean);
        rvi_mean = fifo_mean;  // normalise against FIFO when no RVI
    }

    const double   norm       = compute_rvi ? rvi_mean : fifo_mean;
    const char*    norm_label = compute_rvi ? "NN/RVI"  : "NN/FIFO";
    const char*    base_label = compute_rvi ? "Base/RVI": "Base/FIFO";

    // --- Section header ---
    dp.System() << "\n" << section_label << "\n";
    dp.System() << std::string(section_label.size(), '-') << "\n";
    dp.System() << "  FIFO = " << std::fixed << std::setprecision(4) << fifo_mean;
    if (compute_rvi)
        dp.System() << "  |  RVI = " << rvi_mean
                    << "  |  FIFO/RVI = " << fifo_mean / rvi_mean
                    << "  (" << std::setprecision(1)
                    << (fifo_mean / rvi_mean - 1.0) * 100.0 << "% gap)";
    dp.System() << "\n\n";

    // --- Table header ---
    dp.System() << std::left
        << std::setw(28) << "Base policy"
        << std::setw(11) << base_label
        << std::setw(5)  << "Gen"
        << std::setw(12) << "NN mean"
        << std::setw(10) << norm_label
        << "\n" << std::string(66, '-') << "\n";

    // Helper: train DCL, evaluate every generation, print rows.
    auto train_and_print = [&](const std::string&      base_name,
                                double                  base_direct_mean,
                                int64_t                 n_gens,
                                const DynaPlex::Policy& base_policy)
    {
        auto dcl      = dp.GetDCL(mdp, base_policy, make_dcl(N, M, H, n_gens, nn_arch));
        dcl.TrainPolicy();
        auto policies = dcl.GetPolicies();
        // policies[0] = base, policies[1..n_gens] = NN generations

        // Batch-evaluate all NN generations in one Compare call
        std::vector<DynaPlex::Policy> nn_gens;
        for (int64_t g = 1; g <= n_gens; ++g)
            nn_gens.push_back(policies[(size_t)g]);
        auto res = comparer.Compare(nn_gens);

        for (int64_t g = 1; g <= n_gens; ++g) {
            double nn_mean = 0.0;
            res[(size_t)(g - 1)].Get("mean", nn_mean);

            // Only print base name and base/norm on the first gen row
            if (g == 1) {
                dp.System() << std::left  << std::setw(28) << base_name
                            << std::fixed << std::setprecision(4)
                            << std::setw(11) << base_direct_mean / norm
                            << std::setw(5)  << g
                            << std::setw(12) << nn_mean
                            << std::setw(10) << nn_mean / norm
                            << "\n";
            } else {
                dp.System() << std::left
                            << std::setw(28) << ""   // blank: same base
                            << std::setw(11) << ""   // blank: same base ratio
                            << std::fixed << std::setprecision(4)
                            << std::setw(5)  << g
                            << std::setw(12) << nn_mean
                            << std::setw(10) << nn_mean / norm
                            << "\n";
            }
        }
    };

    // FIFO base
    train_and_print("FIFO (eps=0.00)", fifo_mean, num_gens_fifo, fifo);

    // Stochastic FIFO variants
    for (double eps : epsilons) {
        VarGroup stoch_cfg;
        stoch_cfg.Add("id",        std::string("stochastic_FIFO"));
        stoch_cfg.Add("threshold", eps);
        auto stoch = mdp->GetPolicy(stoch_cfg);

        // Evaluate the base policy directly (once)
        auto base_res = comparer.Compare({stoch});
        double base_mean = 0.0;
        base_res[0].Get("mean", base_mean);

        std::ostringstream lbl;
        lbl << "StochFIFO(eps=" << std::fixed << std::setprecision(2) << eps << ")";
        train_and_print(lbl.str(), base_mean, num_gens, stoch);
    }

    dp.System() << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    auto& dp = DynaPlexProvider::Get();

    dp.System() << "\n=== Stochastic FIFO Exploration: tick_rate x generations x large system ===\n\n";

    // -----------------------------------------------------------------------
    // Shared architectures
    // -----------------------------------------------------------------------
    VarGroup nn_arch_small{
        {"type", "mlp"},
        {"hidden_layers", VarGroup::Int64Vec{64, 32, 2}}
    };
    VarGroup nn_arch_large{
        {"type", "mlp"},
        {"hidden_layers", VarGroup::Int64Vec{256, 128, 64, 2}}
    };

    // -----------------------------------------------------------------------
    // Part 1: Small configs — sweep tick_rate in {1, 3, 6}
    //   H scales with tick_rate (H = 100 * tick_rate) to maintain consistent
    //   real-time trajectory coverage.
    //   FIFO base: num_gens=1 (reference).
    //   StochFIFO(eps=0.30): num_gens=3, every gen reported.
    // -----------------------------------------------------------------------
    dp.System() << "=== Part 1: Small configs (2-server, 2-job) ===\n";
    dp.System() << "  H = 100 x tick_rate  |  N=20000  M=400  num_gens(stoch)=3\n";
    dp.System() << "  RVI computed for each tick_rate (configs are small)\n\n";

    struct SmallCfg {
        std::string name;
        std::string file;
    };
    std::vector<SmallCfg> small_configs = {
        {"asym_cost_2s [cost=[100,300], D=[6,3], fully flexible]",
         "mdp_config_asym_cost_2s.json"},
        {"flex_2s     [specialist server_0={0}, generalist server_1={0,1}]",
         "mdp_config_flex_2s.json"}
    };

    std::vector<double> tick_rates_small = {1.0, 3.0, 6.0};
    std::vector<double> epsilons_small   = {0.30};   // best from previous run

    for (auto& sc : small_configs) {
        auto base_path   = dp.FilePath({"mdp_config_examples", "queue_mdp"}, sc.file);
        auto base_config = VarGroup::LoadFromFile(base_path);

        for (double tr : tick_rates_small) {
            int64_t H = int64_t(100.0 * tr);

            // Override tick_rate in the loaded config
            VarGroup cfg = base_config;
            cfg.Set("tick_rate", tr);

            std::ostringstream lbl;
            lbl << sc.name << "  |  tick_rate=" << std::fixed << std::setprecision(0)
                << tr << "  H=" << H;

            run_one_config(dp,
                lbl.str(),
                cfg,
                /*N=*/       int64_t(20000),
                /*M=*/       int64_t(400),
                /*H=*/       H,
                /*num_gens=*/int64_t(3),
                /*num_gens_fifo=*/int64_t(1),
                epsilons_small,
                /*compute_rvi=*/true,
                nn_arch_small);
        }
    }

    // -----------------------------------------------------------------------
    // Part 2: Large system — 6 job types, 5 server types, chain skill structure
    //   tick_rate=1 (low, so H=100 is sufficient)
    //   N=100000 (5x), M=1600 (4x), num_gens=3
    //   No RVI (state space too large)
    //   Tests FIFO, StochFIFO(0.20), StochFIFO(0.30)
    // -----------------------------------------------------------------------
    dp.System() << "=== Part 2: Large system (6 job types, 5 server types) ===\n";
    dp.System() << "  Chain skill structure: server_k serves jobs {k, k+1}\n";
    dp.System() << "  Simple->complex gradient: mu=[1.2..0.7], cost=[50..200], D=[8..3]\n";
    dp.System() << "  tick_rate=1  H=100  N=100000  M=1600  num_gens=3  No RVI\n\n";

    {
        auto large_path   = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                                         "mdp_config_large_6j5s.json");
        auto large_config = VarGroup::LoadFromFile(large_path);

        std::vector<double> epsilons_large = {0.20, 0.30};

        run_one_config(dp,
            "large_6j5s [k=5, n=6, chain flexibility, tick_rate=1]",
            large_config,
            /*N=*/             int64_t(100000),
            /*M=*/             int64_t(1600),
            /*H=*/             int64_t(100),
            /*num_gens=*/      int64_t(3),
            /*num_gens_fifo=*/ int64_t(1),
            epsilons_large,
            /*compute_rvi=*/   false,
            nn_arch_large);
    }

    dp.System() << "=== DONE ===\n";
    return 0;
}
