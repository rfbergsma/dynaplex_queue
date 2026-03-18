#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

int main() {
    auto& dp = DynaPlexProvider::Get();

    // MDP configs to test
    // use_rel_tol=true  -> rel_tol=0.01 (auto M, accurate for small 2x2 state spaces)
    // use_rel_tol=false -> fixed rvi_M   (medium 3x3 is too large for auto M selection)
    struct ConfigEntry {
        std::string name;
        std::string json;
        int64_t     rvi_M;       // used only when use_rel_tol==false
        bool        use_rel_tol;
    };
    std::vector<ConfigEntry> configs = {
        {"simple",      "mdp_config_simple.json",      0,  true },
        {"simple_asym", "mdp_config_simple_asym.json", 0,  true },
        {"medium",      "mdp_config_1.json",           28, false},
    };

    // Hyperparameter grid
    std::vector<int64_t> Ns     = { 5000, 20000 };
    std::vector<int64_t> Hs     = { 100, 500  };
    std::vector<int64_t> M_dcls = { 400, 1600 };

    // Fixed DCL settings
    VarGroup nn_architecture{
        {"type", "mlp"},
        {"hidden_layers", VarGroup::Int64Vec{128, 64, 2}}
    };
    VarGroup nn_training{ {"early_stopping_patience", 3} };

    // Evaluation settings
    VarGroup test_config;
    test_config.Add("number_of_trajectories", 100);
    test_config.Add("periods_per_trajectory", 10000);

    // Accumulated results for summary table
    struct Result {
        std::string config;
        int64_t N, H, M_dcl;
        double fifo_mean, nn_mean, rvi_mean, nn_over_rvi;
    };
    std::vector<Result> all_results;

    // Header
    dp.System() << "\n=== Analysis 1: DCL Hyperparameter Sensitivity ===\n\n";
    dp.System() << std::left
        << std::setw(12) << "Config"
        << std::setw(8)  << "N"
        << std::setw(7)  << "H"
        << std::setw(9)  << "M_dcl"
        << std::setw(12) << "FIFO_mean"
        << std::setw(12) << "NN_mean"
        << std::setw(12) << "RVI_mean"
        << std::setw(10) << "NN/FIFO"
        << std::setw(10) << "NN/RVI"
        << "\n";
    dp.System() << std::string(92, '-') << "\n";

    for (auto& cfg : configs) {
        auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"}, cfg.json);
        auto mdp_config = VarGroup::LoadFromFile(path);
        auto mdp = dp.GetMDP(mdp_config);
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto comparer = dp.GetPolicyComparer(mdp, test_config);

        // --- Evaluate FIFO baseline once per config ---
        auto fifo_res = comparer.Compare({fifo});
        double fifo_mean = 0.0;
        fifo_res[0].Get("mean", fifo_mean);

        // --- Build RVI config and evaluate once per config (gold standard) ---
        VarGroup rvi_config;
        if (cfg.use_rel_tol) {
            rvi_config = VarGroup{ {"id", std::string("RVI_optimal")}, {"rel_tol", 0.01} };
            dp.System() << "\n-- Config: " << cfg.name
                        << "  (FIFO mean = " << std::fixed << std::setprecision(4) << fifo_mean
                        << ", running RVI with rel_tol=0.01) --\n";
        } else {
            rvi_config = VarGroup{ {"id", std::string("RVI_optimal")}, {"M", cfg.rvi_M} };
            dp.System() << "\n-- Config: " << cfg.name
                        << "  (FIFO mean = " << std::fixed << std::setprecision(4) << fifo_mean
                        << ", running RVI with M=" << cfg.rvi_M << ") --\n";
        }

        auto rvi_policy = mdp->GetPolicy(rvi_config);
        auto rvi_res = comparer.Compare({rvi_policy});
        double rvi_mean = 0.0;
        rvi_res[0].Get("mean", rvi_mean);

        dp.System() << "   RVI_optimal mean = " << std::fixed << std::setprecision(4)
                    << rvi_mean << "  (gap vs FIFO: "
                    << std::setprecision(1) << (rvi_mean / fifo_mean - 1.0) * 100.0
                    << "%)\n";

        // --- Hyperparameter grid ---
        for (int64_t N : Ns)
        for (int64_t H : Hs)
        for (int64_t M_dcl : M_dcls) {
            VarGroup dcl_config{
                {"N",                    N},
                {"num_gens",             int64_t(1)},
                {"M",                    M_dcl},
                {"nn_architecture",      nn_architecture},
                {"nn_training",          nn_training},
                {"retrain_lastgen_only", false},
                {"H",                    H}
            };

            auto dcl = dp.GetDCL(mdp, fifo, dcl_config);
            dcl.TrainPolicy();

            // GetPolicies returns [gen0=FIFO, gen1=NN]; evaluate only NN
            auto policies = dcl.GetPolicies();
            auto nn_policy = policies.back();   // gen-1 NN
            auto res = comparer.Compare({nn_policy});
            double nn_mean = 0.0;
            res[0].Get("mean", nn_mean);

            double nn_over_fifo = (fifo_mean > 1e-12) ? nn_mean / fifo_mean : 0.0;
            double nn_over_rvi  = (rvi_mean  > 1e-12) ? nn_mean / rvi_mean  : 0.0;

            dp.System() << std::left
                << std::setw(12) << cfg.name
                << std::setw(8)  << N
                << std::setw(7)  << H
                << std::setw(9)  << M_dcl
                << std::fixed << std::setprecision(4)
                << std::setw(12) << fifo_mean
                << std::setw(12) << nn_mean
                << std::setw(12) << rvi_mean
                << std::setprecision(4)
                << std::setw(10) << nn_over_fifo
                << std::setw(10) << nn_over_rvi
                << "\n";

            all_results.push_back({cfg.name, N, H, M_dcl,
                                   fifo_mean, nn_mean, rvi_mean, nn_over_rvi});
        }
    }

    // -----------------------------------------------------------------------
    // Summary table — best and worst NN/RVI per config
    // -----------------------------------------------------------------------
    dp.System() << "\n\n";
    dp.System() << std::string(100, '=') << "\n";
    dp.System() << "=== Summary: Best & Worst NN/RVI per Config ===\n";
    dp.System() << std::string(100, '=') << "\n\n";
    dp.System() << std::left
        << std::setw(12) << "Config"
        << std::setw(12) << "FIFO_mean"
        << std::setw(12) << "RVI_mean"
        << std::setw(11) << "FIFO_gap%"
        << std::setw(13) << "Best_NN/RVI"
        << std::setw(28) << "Best (N / H / M_dcl)"
        << std::setw(13) << "Worst_NN/RVI"
        << "\n";
    dp.System() << std::string(101, '-') << "\n";

    for (auto& cfg : configs) {
        double best_ratio  = std::numeric_limits<double>::max();
        double worst_ratio = std::numeric_limits<double>::lowest();
        int64_t best_N = 0, best_H = 0, best_M = 0;
        double fifo_mean = 0.0, rvi_mean = 0.0;

        for (auto& r : all_results) {
            if (r.config != cfg.name) continue;
            fifo_mean = r.fifo_mean;
            rvi_mean  = r.rvi_mean;
            if (r.nn_over_rvi < best_ratio) {
                best_ratio = r.nn_over_rvi;
                best_N = r.N; best_H = r.H; best_M = r.M_dcl;
            }
            if (r.nn_over_rvi > worst_ratio) worst_ratio = r.nn_over_rvi;
        }

        double fifo_gap_pct = (rvi_mean > 1e-12)
            ? (fifo_mean / rvi_mean - 1.0) * 100.0 : 0.0;

        std::ostringstream combo;
        combo << "(" << best_N << " / " << best_H << " / " << best_M << ")";

        dp.System() << std::left
            << std::setw(12) << cfg.name
            << std::fixed << std::setprecision(4)
            << std::setw(12) << fifo_mean
            << std::setw(12) << rvi_mean
            << std::setprecision(1)
            << std::setw(11) << fifo_gap_pct
            << std::setprecision(4)
            << std::setw(13) << best_ratio
            << std::setw(28) << combo.str()
            << std::setw(13) << worst_ratio
            << "\n";
    }

    dp.System() << "\n=== Done ===\n";
    return 0;
}
