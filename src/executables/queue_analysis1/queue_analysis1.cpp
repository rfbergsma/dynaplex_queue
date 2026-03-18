#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

int main() {
    auto& dp = DynaPlexProvider::Get();

    // MDP configs to test
    // rvi_M: truncation horizon passed to RVI_optimal
    //   simple/asym: M=35 — 2x2 state space tractable
    //   medium:      M=25 — smaller to keep 3x3 state space (~250K states) tractable
    struct ConfigEntry { std::string name; std::string json; int64_t rvi_M; };
    std::vector<ConfigEntry> configs = {
        {"simple",      "mdp_config_simple.json",      35},
        {"simple_asym", "mdp_config_simple_asym.json", 35},
        {"medium",      "mdp_config_1.json",           25},
    };

    // Hyperparameter grid
    std::vector<int64_t> Ns     = { 500, 2000 };
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

    // Header
    dp.System() << "\n=== Analysis 1: DCL Hyperparameter Sensitivity ===\n\n";
    dp.System() << std::left
        << std::setw(10) << "Config"
        << std::setw(7)  << "N"
        << std::setw(7)  << "H"
        << std::setw(9)  << "M_dcl"
        << std::setw(14) << "FIFO_mean"
        << std::setw(14) << "NN_mean"
        << std::setw(14) << "RVI_mean"
        << std::setw(10) << "NN/FIFO"
        << std::setw(10) << "NN/RVI"
        << "\n";
    dp.System() << std::string(91, '-') << "\n";

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

        // --- Evaluate RVI_optimal once per config (gold standard) ---
        dp.System() << "\n-- Config: " << cfg.name
                    << "  (FIFO mean = " << std::fixed << std::setprecision(6) << fifo_mean
                    << ", running RVI with M=" << cfg.rvi_M << ") --\n";

        VarGroup rvi_config{ {"id", std::string("RVI_optimal")}, {"M", cfg.rvi_M} };
        auto rvi_policy = mdp->GetPolicy(rvi_config);
        auto rvi_res = comparer.Compare({rvi_policy});
        double rvi_mean = 0.0;
        rvi_res[0].Get("mean", rvi_mean);

        dp.System() << "   RVI_optimal mean = " << std::fixed << std::setprecision(6)
                    << rvi_mean << "  (gap vs FIFO: "
                    << std::setprecision(2) << (rvi_mean / fifo_mean - 1.0) * 100.0
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
                << std::setw(10) << cfg.name
                << std::setw(7)  << N
                << std::setw(7)  << H
                << std::setw(9)  << M_dcl
                << std::fixed << std::setprecision(6)
                << std::setw(14) << fifo_mean
                << std::setw(14) << nn_mean
                << std::setw(14) << rvi_mean
                << std::setprecision(4)
                << std::setw(10) << nn_over_fifo
                << std::setw(10) << nn_over_rvi
                << "\n";
        }
    }

    dp.System() << "\n=== Done ===\n";
    return 0;
}
