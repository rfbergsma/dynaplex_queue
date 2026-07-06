// queue_dcl_probe.cpp
//
// Single-cell DCL playground for fast local iteration.  Runs ONE experiment with
// ONE base policy for NUM_GENS generations, at the FULL hyperparameters used in the
// paper (N=20K, M=400, H=300) so results are trustworthy (not down-scaled).
//
// Edit the KNOBS block, rebuild just this target (~30s), run (~few min/gen).
// No heatmaps, no CSV, no generation guards/EG wrapping — just the diagnostic row:
//
//   Gen  NN*L      NN/RVI   TrLoss   VaLoss   Gap      In
//
// where Gap = VaLoss - TrLoss (classic overfitting signal) and In = #NN input features.
//
// Use it to ablate the label features (LABELS = "none"/"cmu"/"rfq"/"fifo"/combos),
// flip the action sort, swap the base policy, and compare training losses across
// conditions — all without the 30-min full Snellius run.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// ============================================================
//  KNOBS  — edit these, rebuild only queue_dcl_probe
// ============================================================
constexpr int           EXPERIMENT  = 2;              // 2 (fully flexible) or 3 (specialist+generalist)
constexpr int64_t       REWARD_TYPE = 0;              // 0 = binary (FIL>D); 1 = queue-lateness (denser ramp past deadline)
constexpr bool          PRINT_HEATMAP = true;         // print the trained policy's heatmap (use with 1 seed)
static const std::string BASE        = "FIFO policy"; // "FIFO policy" / "reverse_fifo" / "stochastic_FIFO" / "cmu"
static const std::string ACTION_SORT = "fifo";        // "fifo" (descending) / "reverse_fifo" (ascending)
static const std::string LABELS      = "all";         // "all" / "none" / "fifo" / "cmu" / "rfq" / e.g. "cmu+rfq"
constexpr int           NUM_GENS    = 1;              // generations from BASE (raw NN passed forward, no EG)
constexpr double        STOCH_EPS   = 0.30;           // only used when BASE == "stochastic_FIFO"

static const std::string METHOD      = "ppo";         // "dcl" (base+gens) or "ppo" (on-policy)

// Seed sweep: one independent training run per seed (measures run-to-run variance).
// DCL/PPO key is "rng_seed".  Compare the spread of NN/RVI across seeds: that is the
// signal — DCL on Exp3 ranged ~1.0 .. 18.5; the question is whether PPO is tighter.
static const std::vector<int64_t> SEEDS = {1};

// PPO hyperparameters (used when METHOD == "ppo").
constexpr int64_t       PPO_NUM_ENVS      = 16;
constexpr int64_t       PPO_ROLLOUT_STEPS = 256;
constexpr int64_t       PPO_NUM_UPDATES   = 300;
constexpr int64_t       PPO_EPOCHS        = 4;
constexpr int64_t       PPO_MINI_BATCH    = 256;
constexpr double        PPO_LR            = 3e-4;
constexpr double        PPO_GAE_GAMMA     = 0.99;
constexpr double        PPO_ENTROPY_COEF  = 0.01;

// Full hyperparameters (do NOT down-scale: smaller N/M/H is too noisy).
constexpr int64_t       N           = 20000;
constexpr int64_t       M           = 400;
constexpr double        TICK_RATE   = 3.0;
constexpr int64_t       BASE_H      = 100;            // H = BASE_H * TICK_RATE = 300
// ============================================================

// Mirrors make_specialist_generalist_config() in mm1_baseline.cpp (Experiment 3).
static VarGroup exp3_config()
{
    VarGroup srv0;  // specialist: type 0 only
    srv0.Add("servers",       int64_t(1));
    srv0.Add("can_serve",     VarGroup::Int64Vec{0});
    srv0.Add("service_rates", VarGroup::DoubleVec{1.0});

    VarGroup srv1;  // generalist: both types
    srv1.Add("servers",       int64_t(1));
    srv1.Add("can_serve",     VarGroup::Int64Vec{0, 1});
    srv1.Add("service_rates", VarGroup::DoubleVec{1.0, 1.0});

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(2));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       1.0);                  // overridden below
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{0.8, 0.2});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{100.0, 300.0});
    cfg.Add("due_times",       VarGroup::DoubleVec{1.0, 1.0});
    cfg.Add("server_type_0",   srv0);
    cfg.Add("server_type_1",   srv1);
    return cfg;
}

int main()
{
    auto& dp = DynaPlexProvider::Get();

    // --- Build the MDP config for the chosen experiment ---
    VarGroup cfg;
    if (EXPERIMENT == 2) {
        auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                                 "mdp_config_asym_cost_2s.json");
        cfg = VarGroup::LoadFromFile(path);
        cfg.Set("due_times", VarGroup::DoubleVec{3.0, 3.0});   // symmetric deadlines D=[3,3]
    } else {
        cfg = exp3_config();
    }
    cfg.Set("tick_rate",     TICK_RATE);
    cfg.Set("action_sort",   ACTION_SORT);
    cfg.Set("action_labels", LABELS);
    cfg.Set("reward_type",   REWARD_TYPE);

    const int64_t H = int64_t(BASE_H * TICK_RATE);

    // Lambda from the raw MDP (for physical cost rate = mean * Lambda).
    qm::MDP raw_mdp(cfg);
    const double Lambda = raw_mdp.uniformization_rate;

    auto mdp = dp.GetMDP(cfg);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(100));
    eval_cfg.Add("periods_per_trajectory",  int64_t(500000));
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    // --- Benchmarks: FIFO and RVI ---
    auto fifo = mdp->GetPolicy("FIFO policy");
    VarGroup rvi_cfg{{"id", std::string("RVI_optimal")}, {"rel_tol", 0.01}, {"silent", int64_t(1)}};
    auto rvi = mdp->GetPolicy(rvi_cfg);

    auto bench = comparer.Compare({fifo, rvi});
    double fifo_mean = 0.0, rvi_mean = 0.0;
    bench[0].Get("mean", fifo_mean);
    bench[1].Get("mean", rvi_mean);
    const double norm = rvi_mean;

    // --- Base policy ---
    DynaPlex::Policy base;
    if (BASE == "stochastic_FIFO") {
        VarGroup c{{"id", std::string("stochastic_FIFO")}, {"threshold", STOCH_EPS}};
        base = mdp->GetPolicy(c);
    } else {
        base = mdp->GetPolicy(BASE);
    }
    double base_mean = 0.0;
    comparer.Compare({base})[0].Get("mean", base_mean);

    // --- NN architecture (same as paper) ---
    VarGroup nn_arch;
    nn_arch.Add("type",          std::string("mlp"));
    nn_arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

    // --- Header ---
    std::cout << "\n================ queue_dcl_probe ================\n";
    std::cout << "method=" << METHOD
              << "  Exp" << EXPERIMENT
              << "  base=" << (METHOD == "ppo" ? std::string("(none)") : BASE)
              << "  sort=" << ACTION_SORT
              << "  labels=" << LABELS
              << "  reward_type=" << REWARD_TYPE << "\n";
    std::cout << "N=" << N << "  M=" << M << "  H=" << H
              << "  tick_rate=" << TICK_RATE << "  Lambda=" << std::fixed << std::setprecision(3) << Lambda << "\n";
    std::cout << std::fixed << std::setprecision(4)
              << "FIFO*L=" << fifo_mean * Lambda
              << "  RVI*L=" << rvi_mean * Lambda
              << "  FIFO/RVI=" << fifo_mean / norm
              << "  Base/RVI=" << base_mean / norm << "\n";
    std::cout << std::left
              << std::setw(6)  << "Seed"
              << std::setw(5)  << "Gen"
              << std::setw(12) << "NN*Lambda"
              << std::setw(10) << "NN/RVI"
              << std::setw(10) << "TrLoss"
              << std::setw(10) << "VaLoss"
              << std::setw(10) << "Gap"
              << std::setw(6)  << "In"
              << "\n" << std::string(69, '-') << "\n";

    // --- Seed sweep: independent training run per seed ---
    std::vector<double> ratios;
    for (int64_t seed : SEEDS) {
        if (METHOD == "ppo") {
            VarGroup ppo_cfg;
            ppo_cfg.Add("num_envs",          PPO_NUM_ENVS);
            ppo_cfg.Add("rollout_steps",     PPO_ROLLOUT_STEPS);
            ppo_cfg.Add("num_updates",       PPO_NUM_UPDATES);
            ppo_cfg.Add("epochs_per_update", PPO_EPOCHS);
            ppo_cfg.Add("mini_batch_size",   PPO_MINI_BATCH);
            ppo_cfg.Add("learning_rate",     PPO_LR);
            ppo_cfg.Add("gae_gamma",         PPO_GAE_GAMMA);
            ppo_cfg.Add("entropy_coef",      PPO_ENTROPY_COEF);
            ppo_cfg.Add("rng_seed",          seed);
            ppo_cfg.Add("silent",            false);   // show training trace for diagnosis
            VarGroup parch; parch.Add("hidden_layers", VarGroup::Int64Vec{64, 32});
            ppo_cfg.Add("nn_architecture", parch);

            auto ppo = dp.GetPPO(mdp, nullptr, ppo_cfg);
            ppo.TrainPolicy();
            auto nn = ppo.GetPolicy();
            auto nn_stoch = ppo.GetStochasticPolicy();

            double nn_mean = 0.0, stoch_mean = 0.0;
            try {
                std::cout << "  [eval] comparing argmax + stochastic PPO policies...\n" << std::flush;
                auto res = comparer.Compare({nn, nn_stoch});
                res[0].Get("mean", nn_mean);
                res[1].Get("mean", stoch_mean);
            } catch (const std::exception& e) {
                std::cout << "  [eval EXCEPTION] " << e.what() << "\n" << std::flush;
                continue;
            }
            const double nn_ratio = nn_mean / norm;
            ratios.push_back(nn_ratio);

            std::cout << std::left << std::fixed << std::setprecision(4)
                      << std::setw(6)  << seed
                      << std::setw(5)  << "-"
                      << std::setw(12) << nn_mean * Lambda
                      << std::setw(10) << nn_ratio
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(6)  << "-"
                      << "\n" << std::flush;
            std::cout << "      [stoch] NN*Lambda=" << stoch_mean * Lambda
                      << "  NN/RVI=" << stoch_mean / norm
                      << "   (stoch << argmax = extraction mismatch)\n" << std::flush;
            if (PRINT_HEATMAP) {
                std::cout << "\n  PPO policy heatmap [Exp" << EXPERIMENT
                          << ", reward=" << REWARD_TYPE << ", seed=" << seed << "]:\n";
                qm::PrintPolicyHeatmap(mdp, nn, 12);
                std::cout << std::flush;
            }
            continue;
        }

        DynaPlex::Policy current_base = base;
        for (int g = 1; g <= NUM_GENS; ++g) {
            VarGroup dcl;
            dcl.Add("N",               N);
            dcl.Add("M",               M);
            dcl.Add("H",               H);
            dcl.Add("num_gens",        int64_t(1));
            dcl.Add("silent",          true);
            dcl.Add("rng_seed",        seed);
            dcl.Add("nn_architecture", nn_arch);
            VarGroup nn_training;
            nn_training.Add("early_stopping_patience", int64_t(3));
            dcl.Add("nn_training", nn_training);

            auto dclo = dp.GetDCL(mdp, current_base, dcl);
            dclo.TrainPolicy();
            auto nn = dclo.GetPolicies()[(size_t)1];

            double nn_mean = 0.0;
            comparer.Compare({nn})[0].Get("mean", nn_mean);
            const double nn_ratio = nn_mean / norm;
            if (g == NUM_GENS) ratios.push_back(nn_ratio);

            double  t_loss = 0.0, v_loss = 0.0;
            int64_t num_inputs = 0;
            auto ncfg = nn->GetConfig();
            ncfg.Get("saved_training_loss",   t_loss);
            ncfg.Get("saved_validation_loss", v_loss);
            if (ncfg.HasKey("num_inputs")) ncfg.Get("num_inputs", num_inputs);

            std::cout << std::left << std::fixed << std::setprecision(4)
                      << std::setw(6)  << seed
                      << std::setw(5)  << g
                      << std::setw(12) << nn_mean * Lambda
                      << std::setw(10) << nn_ratio
                      << std::setw(10) << t_loss
                      << std::setw(10) << v_loss
                      << std::setw(10) << (v_loss - t_loss)
                      << std::setw(6)  << num_inputs
                      << "\n" << std::flush;

            current_base = nn;  // raw NN forward (only matters if NUM_GENS>1)
        }
    }

    // --- Summary stats over seeds (final-gen NN/RVI) ---
    if (!ratios.empty()) {
        double mn = ratios[0], mx = ratios[0], sum = 0.0;
        for (double r : ratios) { mn = std::min(mn, r); mx = std::max(mx, r); sum += r; }
        const double mean = sum / ratios.size();
        double var = 0.0;
        for (double r : ratios) var += (r - mean) * (r - mean);
        const double sd = ratios.size() > 1 ? std::sqrt(var / (ratios.size() - 1)) : 0.0;
        std::cout << std::string(69, '-') << "\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "NN/RVI over " << ratios.size() << " seeds:  "
                  << "min=" << mn << "  max=" << mx
                  << "  mean=" << mean << "  sd=" << sd << "\n";
    }

    std::cout << "================================================\n";
    return 0;
}
