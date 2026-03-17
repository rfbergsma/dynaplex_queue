#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <numeric>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

// -----------------------------------------------------------------------
// Scenario definition
// -----------------------------------------------------------------------
struct Scenario {
    std::string name;
    VarGroup::DoubleVec arrival_rates;
    double tick_rate;
    VarGroup::DoubleVec due_times;
    double mu0;
    double mu1;
};

static VarGroup build_mdp_config(const Scenario& s) {
    VarGroup srv0;
    srv0.Add("servers",      int64_t(1));
    srv0.Add("can_serve",    VarGroup::Int64Vec{0, 1});
    srv0.Add("service_rate", s.mu0);

    VarGroup srv1;
    srv1.Add("servers",      int64_t(1));
    srv1.Add("can_serve",    VarGroup::Int64Vec{0, 1});
    srv1.Add("service_rate", s.mu1);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(2));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       s.tick_rate);
    cfg.Add("arrival_rates",   s.arrival_rates);
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0, 1.0});
    cfg.Add("due_times",       s.due_times);
    cfg.Add("server_type_0",   srv0);
    cfg.Add("server_type_1",   srv1);
    return cfg;
}

int main() {
    auto& dp = DynaPlexProvider::Get();

    // -----------------------------------------------------------------------
    // 8 named scenarios (all 2x2: n_jobs=2, k_servers=2, servers=1 each)
    // rho = sum(lambda) / (mu0 + mu1)
    // -----------------------------------------------------------------------
    std::vector<Scenario> scenarios = {
        // name,                arrival_rates,       tick, due_times,   mu0,  mu1
        {"low_rho_tight",    {0.175, 0.175},  1.0, {3.0, 3.0},  0.35, 0.35},
        {"low_rho_loose",    {0.175, 0.175},  1.0, {8.0, 8.0},  0.35, 0.35},
        {"med_rho_tight",    {0.250, 0.250},  1.0, {3.0, 3.0},  0.35, 0.35},
        {"med_rho_loose",    {0.250, 0.250},  1.0, {8.0, 8.0},  0.35, 0.35},
        {"high_rho_tight",   {0.285, 0.285},  1.0, {3.0, 3.0},  0.35, 0.35},
        {"high_rho_loose",   {0.285, 0.285},  1.0, {8.0, 8.0},  0.35, 0.35},
        {"slow_tick_med",    {0.250, 0.250},  0.5, {5.0, 5.0},  0.35, 0.35},
        {"asymmetric_srvr",  {0.250, 0.250},  1.0, {5.0, 5.0},  0.50, 0.20},
    };

    // Fixed DCL settings (modest, since we run 8 times)
    VarGroup nn_architecture{
        {"type", "mlp"},
        {"hidden_layers", VarGroup::Int64Vec{128, 64, 2}}
    };
    VarGroup nn_training{ {"early_stopping_patience", 3} };

    VarGroup dcl_config{
        {"N",                    int64_t(1000)},
        {"num_gens",             int64_t(1)},
        {"M",                    int64_t(800)},
        {"nn_architecture",      nn_architecture},
        {"nn_training",          nn_training},
        {"retrain_lastgen_only", false},
        {"H",                    int64_t(200)}
    };

    // Evaluation settings
    VarGroup test_config;
    test_config.Add("number_of_trajectories", 100);
    test_config.Add("periods_per_trajectory", 10000);

    // RVI config (fixed M=35 — tractable for 2x2 at rho up to ~0.81)
    VarGroup rvi_config{ {"id", std::string("RVI_optimal")}, {"M", int64_t(35)} };

    // -----------------------------------------------------------------------
    // Header
    // -----------------------------------------------------------------------
    dp.System() << "\n=== Analysis 2: Policy Comparison Across Problem Settings ===\n\n";
    dp.System() << std::left
        << std::setw(20) << "Scenario"
        << std::setw(7)  << "rho"
        << std::setw(7)  << "tick"
        << std::setw(7)  << "due"
        << std::setw(12) << "FIFO_mean"
        << std::setw(12) << "NN_mean"
        << std::setw(12) << "RVI_mean"
        << std::setw(10) << "NN/RVI"
        << "\n";
    dp.System() << std::string(87, '-') << "\n";

    for (auto& s : scenarios) {
        // Compute rho for display
        double sum_lambda = std::accumulate(s.arrival_rates.begin(), s.arrival_rates.end(), 0.0);
        double sum_mu     = s.mu0 + s.mu1;
        double rho        = sum_lambda / sum_mu;
        double due_t      = s.due_times[0];

        // Build MDP
        auto mdp  = dp.GetMDP(build_mdp_config(s));
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto comparer = dp.GetPolicyComparer(mdp, test_config);

        // Train DCL
        dp.System() << "  [Training DCL for scenario: " << s.name << "]\n";
        auto dcl = dp.GetDCL(mdp, fifo, dcl_config);
        dcl.TrainPolicy();

        // Collect policies: FIFO (gen-0), NN (gen-1), RVI_optimal
        auto all_policies = dcl.GetPolicies();          // [FIFO, NN]
        all_policies.push_back(mdp->GetPolicy(rvi_config));  // append RVI

        // Compare all three
        auto res = comparer.Compare(all_policies);

        double fifo_mean = 0.0, nn_mean = 0.0, rvi_mean = 0.0;
        res[0].Get("mean", fifo_mean);  // gen-0 = FIFO
        res[1].Get("mean", nn_mean);    // gen-1 = NN
        res[2].Get("mean", rvi_mean);   // RVI_optimal

        double nn_over_rvi = (rvi_mean > 1e-12) ? nn_mean / rvi_mean : 0.0;

        dp.System() << std::left
            << std::setw(20) << s.name
            << std::fixed << std::setprecision(3)
            << std::setw(7)  << rho
            << std::setw(7)  << s.tick_rate
            << std::setw(7)  << due_t
            << std::setprecision(6)
            << std::setw(12) << fifo_mean
            << std::setw(12) << nn_mean
            << std::setw(12) << rvi_mean
            << std::setprecision(4)
            << std::setw(10) << nn_over_rvi
            << "\n";
    }

    dp.System() << "\n=== Done ===\n";
    return 0;
}
