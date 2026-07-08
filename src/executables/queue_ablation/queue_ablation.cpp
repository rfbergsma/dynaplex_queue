// queue_ablation.cpp
//
// Controlled ablation matrix for the tuned PPO recipe (paper evidence runs).
// Starts from the frozen recipe (gamma=0.997, entropy=0.03, temp_min=0.1,
// rollout=512, resets=16, reward_type=2) and flips ONE ingredient per cell.
//
// Usage (one cell x one seed per process, for slurm backfill):
//   queue_ablation cell=base seeds=1,2,3
//   queue_ablation cell=dper seeds=4 updates=6000 eval_periods=500000
//
// Cells:
//   base     - the full recipe, nothing removed (control)
//   dper     - restore the semi-MDP discounting bug (gamma^max(1,dp))     [mech 1]
//   vnorm    - no value-target normalization                              [mech 2]
//   resets0  - never reset environments                                   [mech 3]
//   shape0   - reward_type=0: true cost, no shaping                       [shaping]
//   anneal0  - no guarded temperature annealing                           [mech 4 mitigation]
//   skipbias - skip_all action + pessimistic init bias -3                 [P4 confirmation]
//   comboall - skip_all + macro features + bias -3                        [P4 confirmation]
//
// Reference for Exp2 (tick=3): RVI*Lambda = 40.50, FIFO*Lambda = 70.46.
// Within-5% threshold: NN*Lambda <= 42.53.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;

int main(int argc, char** argv)
{
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1);
    }
    auto S = [&](const std::string& k, const std::string& d) { auto it = kv.find(k); return it == kv.end() ? d : it->second; };
    auto I = [&](const std::string& k, int64_t d)            { auto it = kv.find(k); return it == kv.end() ? d : (int64_t)std::atoll(it->second.c_str()); };

    const std::string CELL         = S("cell", "base");
    const int64_t     UPDATES      = I("updates", 6000);
    const int64_t     EVAL_TRAJ    = I("eval_traj", 100);
    const int64_t     EVAL_PERIODS = I("eval_periods", 500000);
    const double      RVI_REF      = std::atof(S("rvi_ref", "40.50").c_str());  // NN*Lambda scale

    std::vector<int64_t> SEEDS;
    {
        std::stringstream ss(S("seeds", "1,2,3,4,5,6,7,8"));
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) SEEDS.push_back(std::atoll(tok.c_str()));
    }

    // ---- per-cell deltas from the recipe ----
    int64_t reward_type = 2;
    bool    skip_all = false, macro_feat = false;
    bool    dper_clamp = true, value_norm = true, temp_anneal = true;
    int64_t resets = 16;
    double  skip_bias = 0.0;

    if      (CELL == "base")     { /* the full recipe */ }
    else if (CELL == "dper")     { dper_clamp = false; }
    else if (CELL == "vnorm")    { value_norm = false; }
    else if (CELL == "resets0")  { resets = 0; }
    else if (CELL == "shape0")   { reward_type = 0; }
    else if (CELL == "anneal0")  { temp_anneal = false; }
    else if (CELL == "skipbias") { skip_all = true; skip_bias = -3.0; }
    else if (CELL == "comboall") { skip_all = true; macro_feat = true; skip_bias = -3.0; }
    else { std::cout << "unknown cell '" << CELL << "'\n"; return 1; }

    auto& dp = DynaPlexProvider::Get();

    // ---- Exp2 MDP (tick=3, symmetric deadlines D=[3,3]) ----
    auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"}, "mdp_config_asym_cost_2s.json");
    VarGroup cfg = VarGroup::LoadFromFile(path);
    cfg.Set("due_times",     VarGroup::DoubleVec{3.0, 3.0});
    cfg.Set("tick_rate",     3.0);
    cfg.Set("action_sort",   std::string("fifo"));
    cfg.Set("action_labels", std::string("all"));
    cfg.Set("reward_type",   reward_type);
    if (skip_all)   cfg.Set("enable_skip_all", true);
    if (macro_feat) cfg.Set("macro_features",  true);

    DynaPlex::Models::queue_mdp::MDP raw_mdp(cfg);
    const double Lambda = raw_mdp.uniformization_rate;
    auto mdp = dp.GetMDP(cfg);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", EVAL_TRAJ);
    eval_cfg.Add("periods_per_trajectory",  EVAL_PERIODS);
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    std::cout << "\n================ queue_ablation ================\n";
    std::cout << "cell=" << CELL
              << "  reward_type=" << reward_type
              << "  dper_clamp=" << dper_clamp
              << "  value_norm=" << value_norm
              << "  resets=" << resets
              << "  temp_anneal=" << temp_anneal
              << "  skip_all=" << skip_all
              << "  macro_feat=" << macro_feat
              << "  skip_bias=" << skip_bias << "\n";
    std::cout << "updates=" << UPDATES << "  Lambda=" << std::fixed << std::setprecision(3) << Lambda
              << "  rvi_ref=" << std::setprecision(2) << RVI_REF << "\n";
    std::cout << std::left << std::setw(6) << "Seed"
              << std::setw(14) << "argmax*L"
              << std::setw(14) << "stoch*L"
              << std::setw(10) << "arg/RVI"
              << std::setw(8)  << "in5%"
              << "\n" << std::string(52, '-') << "\n";

    std::vector<double> ratios;
    int64_t n_in5 = 0;
    for (int64_t seed : SEEDS) {
        VarGroup ppo_cfg;
        ppo_cfg.Add("num_envs",          (int64_t)16);
        ppo_cfg.Add("rollout_steps",     (int64_t)512);
        ppo_cfg.Add("num_updates",       UPDATES);
        ppo_cfg.Add("epochs_per_update", (int64_t)4);
        ppo_cfg.Add("mini_batch_size",   (int64_t)256);
        ppo_cfg.Add("learning_rate",     3e-4);
        ppo_cfg.Add("gae_gamma",         0.997);
        ppo_cfg.Add("gae_lambda",        0.95);
        ppo_cfg.Add("entropy_coef",      0.03);
        ppo_cfg.Add("temp_anneal",       temp_anneal);
        ppo_cfg.Add("temp_min",          0.1);
        ppo_cfg.Add("env_reset_every",   resets);
        ppo_cfg.Add("dper_clamp",        dper_clamp);
        ppo_cfg.Add("value_norm",        value_norm);
        ppo_cfg.Add("skip_all_bias",     skip_bias);
        ppo_cfg.Add("rng_seed",          seed);
        ppo_cfg.Add("silent",            false);
        VarGroup parch; parch.Add("hidden_layers", VarGroup::Int64Vec{64, 32});
        ppo_cfg.Add("nn_architecture", parch);

        auto ppo = dp.GetPPO(mdp, nullptr, ppo_cfg);
        ppo.TrainPolicy();

        double m_arg = 0.0, m_sto = 0.0;
        try {
            auto res = comparer.Compare({ ppo.GetPolicy(), ppo.GetStochasticPolicy() });
            res[0].Get("mean", m_arg);
            res[1].Get("mean", m_sto);
        } catch (const std::exception& e) {
            std::cout << seed << "  [eval EXCEPTION] " << e.what() << "\n" << std::flush;
            continue;
        }
        const double aL = m_arg * Lambda, sL = m_sto * Lambda;
        const double ratio = aL / RVI_REF;
        const bool in5 = ratio <= 1.05;
        if (in5) ++n_in5;
        ratios.push_back(ratio);

        std::cout << std::left << std::fixed << std::setprecision(4)
                  << std::setw(6)  << seed
                  << std::setw(14) << aL
                  << std::setw(14) << sL
                  << std::setw(10) << ratio
                  << std::setw(8)  << (in5 ? "yes" : "-")
                  << "\n" << std::flush;
    }

    if (!ratios.empty()) {
        std::vector<double> sorted = ratios;
        std::sort(sorted.begin(), sorted.end());
        const double median = sorted[sorted.size() / 2];
        const double worst  = sorted.back();
        std::cout << std::string(52, '-') << "\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "cell=" << CELL << "  seeds=" << ratios.size()
                  << "  in5%=" << n_in5 << "/" << ratios.size()
                  << "  median=" << median
                  << "  worst=" << worst << "\n";
    }
    std::cout << "================================================\n";
    return 0;
}
