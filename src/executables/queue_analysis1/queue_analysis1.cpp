#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

int main() {
    auto& dp = DynaPlexProvider::Get();

    // Two MDP configs: simple (2x2, RVI-feasible) and medium (3x3)
    struct ConfigEntry { std::string name; std::string json; };
    std::vector<ConfigEntry> configs = {
        {"simple", "mdp_config_simple.json"},
        {"medium", "mdp_config_1.json"},
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
        << std::setw(10) << "NN/FIFO"
        << "\n";
    dp.System() << std::string(71, '-') << "\n";

    for (auto& cfg : configs) {
        auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"}, cfg.json);
        auto mdp_config = VarGroup::LoadFromFile(path);
        auto mdp = dp.GetMDP(mdp_config);
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto comparer = dp.GetPolicyComparer(mdp, test_config);

        // Evaluate FIFO baseline once per config
        auto fifo_res = comparer.Compare({fifo});
        double fifo_mean = 0.0;
        fifo_res[0].Get("mean", fifo_mean);

        dp.System() << "\n-- Config: " << cfg.name << "  (FIFO mean = "
                    << std::fixed << std::setprecision(6) << fifo_mean << ") --\n";

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

            double ratio = (fifo_mean > 1e-12) ? nn_mean / fifo_mean : 0.0;

            dp.System() << std::left
                << std::setw(10) << cfg.name
                << std::setw(7)  << N
                << std::setw(7)  << H
                << std::setw(9)  << M_dcl
                << std::fixed << std::setprecision(6)
                << std::setw(14) << fifo_mean
                << std::setw(14) << nn_mean
                << std::setprecision(4)
                << std::setw(10) << ratio
                << "\n";
        }
    }

    dp.System() << "\n=== Done ===\n";
    return 0;
}
