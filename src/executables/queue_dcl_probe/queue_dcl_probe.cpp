// queue_dcl_probe.cpp
//
// Single-cell DCL/PPO playground for fast local iteration.  Runs ONE experiment
// with ONE method for a sweep of seeds, at full paper hyperparameters by default.
//
// All knobs are runtime key=value arguments (defaults below) — no rebuild needed,
// and multiple configurations can run in PARALLEL as separate processes:
//
//   queue_dcl_probe exp=3 reward=2 seeds=1,2,3
//   queue_dcl_probe exp=2 reward=2 seeds=1 heatmap=1 temp_min=0.10 updates=600
//   queue_dcl_probe exp=3 method=dcl reward=2 seeds=1,2,3
//
// Keys (default):
//   exp(2) reward(2) method(ppo) seeds(2,3) heatmap(0) gens(1)
//   base(FIFO policy) sort(fifo) labels(all)
//   updates(300) envs(16) rollout(256) epochs(4) minibatch(256) lr(3e-4)
//   gamma(0.99) lambda(0.95) entropy(0.01) avg(0) rho_step(1.0)
//   temp_anneal(1) temp_min(0.25) resets(16)
//   N(20000) M(400) tick(3.0) base_h(100) eval_traj(100) eval_periods(50000)
//
// Output per seed: argmax row (NN*Lambda, NN/RVI) + [stoch] line; summary stats.

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <random>
#include <thread>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// Hand-coded policy that always takes one fixed action, for eval-path integrity
// checks (const_eval=1): always-0 = skip chain, always-2 = skip-all.  The two are
// value-degenerate by design, so their evaluated costs must match EXACTLY — and
// must equal the cost of a trained net whose argmax collapsed to always-idle.
class ConstActionPolicy : public DynaPlex::PolicyInterface {
    int64_t action;
    DynaPlex::VarGroup config;
public:
    explicit ConstActionPolicy(int64_t action) : action(action) {
        config.Add("id", std::string("ConstAction"));
        config.Add("action", action);
    }
    std::string TypeIdentifier() const override { return "ConstAction"; }
    const DynaPlex::VarGroup& GetConfig() const override { return config; }
    void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override {
        for (auto& t : trajectories) t.NextAction = action;
    }
};

// Hand-coded stochastic policy sampling from fixed action probabilities
// (mix_eval=p0,p1,p2).  Distinguishes "the training rollout's terrible
// rew/period under mixed play is REAL dynamics" from "trainer-side accounting
// bug": the eval harness measures the same behavior independently.
// Disallowed sampled actions fall back to 0 (always allowed).
class MixActionPolicy : public DynaPlex::PolicyInterface {
    DynaPlex::MDP mdp;
    std::vector<double> cum;   // cumulative probabilities over actions 0..A-1
    DynaPlex::VarGroup config;
public:
    MixActionPolicy(DynaPlex::MDP mdp, const std::vector<double>& probs)
        : mdp(mdp)
    {
        double sum = 0.0;
        for (double p : probs) sum += p;
        double c = 0.0;
        for (double p : probs) { c += p / sum; cum.push_back(c); }
        config.Add("id", std::string("MixAction"));
    }
    std::string TypeIdentifier() const override { return "MixAction"; }
    const DynaPlex::VarGroup& GetConfig() const override { return config; }
    void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override {
        thread_local std::mt19937 rng(
            0x9e3779b9u ^ (unsigned)std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for (auto& t : trajectories) {
            const double u = unif(rng);
            int64_t a = 0;
            for (size_t i = 0; i < cum.size(); ++i)
                if (u <= cum[i]) { a = (int64_t)i; break; }
            if (a != 0 && !mdp->IsAllowedAction(t.GetState(), a)) a = 0;
            t.NextAction = a;
        }
    }
};

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

int main(int argc, char** argv)
{
    // ---------------- runtime knobs (key=value args) ----------------
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq != std::string::npos) kv[a.substr(0, eq)] = a.substr(eq + 1);
    }
    auto S = [&](const std::string& k, const std::string& d) { auto it = kv.find(k); return it == kv.end() ? d : it->second; };
    auto I = [&](const std::string& k, int64_t d)            { auto it = kv.find(k); return it == kv.end() ? d : (int64_t)std::atoll(it->second.c_str()); };
    auto D = [&](const std::string& k, double d)             { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); };

    const int         EXPERIMENT   = (int)I("exp", 2);
    const int64_t     REWARD_TYPE  = I("reward", 2);
    const std::string METHOD       = S("method", "ppo");
    const bool        PRINT_HEATMAP= I("heatmap", 0) != 0;
    const std::string BASE         = S("base", "FIFO policy");
    const std::string ACTION_SORT  = S("sort", "fifo");
    const std::string LABELS       = S("labels", "all");
    const int         NUM_GENS     = (int)I("gens", 1);
    const double      STOCH_EPS    = D("stoch_eps", 0.30);

    std::vector<int64_t> SEEDS;
    {
        std::stringstream ss(S("seeds", "2,3"));
        std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) SEEDS.push_back(std::atoll(tok.c_str()));
    }

    const int64_t PPO_NUM_ENVS      = I("envs", 16);
    const int64_t PPO_ROLLOUT_STEPS = I("rollout", 256);
    const int64_t PPO_NUM_UPDATES   = I("updates", 300);
    const int64_t PPO_EPOCHS        = I("epochs", 4);
    const int64_t PPO_MINI_BATCH    = I("minibatch", 256);
    const double  PPO_LR            = D("lr", 3e-4);
    const double  PPO_GAE_GAMMA     = D("gamma", 0.99);
    const double  PPO_GAE_LAMBDA    = D("lambda", 0.95);
    const double  PPO_ENTROPY_COEF  = D("entropy", 0.01);
    const bool    PPO_AVG_REWARD    = I("avg", 0) != 0;
    const double  PPO_RHO_STEP      = D("rho_step", 1.0);
    const bool    PPO_TEMP_ANNEAL   = I("temp_anneal", 1) != 0;
    const double  PPO_TEMP_MIN      = D("temp_min", 0.25);
    const int64_t PPO_RESETS        = I("resets", 16);
    const bool    PPO_DISTILL       = I("distill", 0) != 0;   // DCL gen-1 from the stochastic PPO policy

    const int64_t N            = I("N", 20000);
    const int64_t M            = I("M", 400);
    const double  TICK_RATE    = D("tick", 3.0);
    const int64_t BASE_H       = I("base_h", 100);
    const int64_t EVAL_TRAJ    = I("eval_traj", 100);
    const int64_t EVAL_PERIODS = I("eval_periods", 50000);
    // bench=1 full (FIFO+RVI); bench=2 FIFO only (for cells where RVI is
    // intractable, e.g. exp=6); bench=0 none (RVI solve costs 30+ min per
    // process — ratios computed offline against known references).
    const int64_t BENCH_MODE   = I("bench", 1);
    // -----------------------------------------------------------------

    auto& dp = DynaPlexProvider::Get();

    // --- Build the MDP config for the chosen experiment ---
    VarGroup cfg;
    if (EXPERIMENT == 2) {
        auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                                 "mdp_config_asym_cost_2s.json");
        cfg = VarGroup::LoadFromFile(path);
        cfg.Set("due_times", VarGroup::DoubleVec{3.0, 3.0});   // symmetric deadlines D=[3,3]
    } else if (EXPERIMENT == 6) {
        // large instance: 6 job types, 5 chain-skill servers.  RVI intractable
        // here — use bench=2 (FIFO-only baseline).
        auto path = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                                 "mdp_config_large_6j5s.json");
        cfg = VarGroup::LoadFromFile(path);
    } else {
        cfg = exp3_config();
    }
    cfg.Set("tick_rate",     TICK_RATE);
    cfg.Set("action_sort",   ACTION_SORT);
    cfg.Set("action_labels", LABELS);
    cfg.Set("reward_type",   REWARD_TYPE);
    // salt: extra config key that changes the MDP identity hash (and nothing
    // else).  Gives concurrent DCL jobs their own sample-cache directories —
    // without it, same-config jobs race on samples_gen0.json (corrupt JSON,
    // SIGABRT) or silently REUSE each other's samples (non-independent seeds).
    const int64_t SALT = I("salt", 0);
    if (SALT != 0) cfg.Add("sample_salt", SALT);
    // 2x2 experiment knobs: skip_all adds action 2 (skip all remaining candidates);
    // macro_feat adds remaining-queue + idle-cost summary features.
    if (I("skip_all", 0) != 0)   cfg.Set("enable_skip_all", true);
    if (I("macro_feat", 0) != 0) cfg.Set("macro_features", true);
    // mode=pe: per-event action space (each idle capacity unit picks a type or
    // idles; valid_actions = n_jobs+1).  Default: the candidate-queue space.
    if (S("mode", "") == "pe")   cfg.Set("action_mode", std::string("per_event"));

    const int64_t H = int64_t((double)BASE_H * TICK_RATE);

    // Lambda from the raw MDP (for physical cost rate = mean * Lambda).
    qm::MDP raw_mdp(cfg);
    const double Lambda = raw_mdp.uniformization_rate;

    auto mdp = dp.GetMDP(cfg);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", EVAL_TRAJ);
    eval_cfg.Add("periods_per_trajectory",  EVAL_PERIODS);
    auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

    // --- mix_eval=p0,p1,p2: evaluate a fixed-probability random policy and exit ---
    if (kv.count("mix_eval")) {
        std::vector<double> probs;
        {
            std::stringstream ss(S("mix_eval", "1,1,1"));
            std::string tok;
            while (std::getline(ss, tok, ',')) if (!tok.empty()) probs.push_back(std::atof(tok.c_str()));
        }
        std::vector<DynaPlex::Policy> mpols = {
            std::make_shared<MixActionPolicy>(mdp, probs),
            mdp->GetPolicy("FIFO policy") };   // same-process reference
        std::cout << "\n================ mix_eval ================\n";
        std::cout << "  probs = " << S("mix_eval", "1,1,1") << "\n";
        auto res = comparer.Compare(mpols);
        for (size_t r = 0; r < mpols.size(); ++r) {
            double m = 0.0; res[r].Get("mean", m);
            std::cout << std::fixed << std::setprecision(4)
                      << "  [" << (r == 0 ? "mix " : "FIFO") << "] mean/period=" << m
                      << "  NN*Lambda=" << m * Lambda << "\n" << std::flush;
        }
        std::cout << "==========================================\n";
        return 0;
    }

    // --- const_eval=1: evaluate hand-coded constant policies and exit ---
    // Integrity check for the skip-all action + eval path, no training involved.
    if (I("const_eval", 0) != 0) {
        std::vector<std::string> cnames = { "always-0 (skip chain)" };
        std::vector<DynaPlex::Policy> cpols = { std::make_shared<ConstActionPolicy>(0) };
        if (I("skip_all", 0) != 0) {
            cnames.push_back("always-2 (skip-all)");
            cpols.push_back(std::make_shared<ConstActionPolicy>(2));
        }
        std::cout << "\n================ const_eval ================\n";
        auto res = comparer.Compare(cpols);
        for (size_t r = 0; r < cpols.size(); ++r) {
            double m = 0.0; res[r].Get("mean", m);
            std::cout << "  [" << std::left << std::setw(22) << cnames[r] << "] "
                      << std::fixed << std::setprecision(4)
                      << "NN*Lambda=" << m * Lambda << "\n" << std::flush;
        }
        std::cout << "============================================\n";
        return 0;
    }

    // --- Benchmarks (bench=1: FIFO+RVI; bench=2: FIFO only; bench=0: none) ---
    double fifo_mean = 1.0, rvi_mean = 1.0, base_mean = 1.0;
    if (BENCH_MODE == 1) {
        auto fifo = mdp->GetPolicy("FIFO policy");
        // rvi_tol: between-M convergence tolerance of the auto-truncation loop.
        // The 0.01 default is too loose for queue-lateness rewards (cost keeps
        // growing with FIL depth, so g* still drifts at the default stop).
        VarGroup rvi_cfg{{"id", std::string("RVI_optimal")}, {"rel_tol", D("rvi_tol", 0.01)}, {"silent", int64_t(1)}};
        auto rvi = mdp->GetPolicy(rvi_cfg);

        auto bench = comparer.Compare({fifo, rvi});
        bench[0].Get("mean", fifo_mean);
        bench[1].Get("mean", rvi_mean);
    } else if (BENCH_MODE == 2) {
        auto fifo = mdp->GetPolicy("FIFO policy");
        comparer.Compare({fifo})[0].Get("mean", fifo_mean);
        rvi_mean = fifo_mean;   // norm against FIFO: NN/RVI column reads as NN/FIFO
    }
    const double norm = rvi_mean;

    // --- Base policy (always constructed: DCL needs it) ---
    DynaPlex::Policy base;
    if (BASE == "stochastic_FIFO") {
        VarGroup c{{"id", std::string("stochastic_FIFO")}, {"threshold", STOCH_EPS}};
        base = mdp->GetPolicy(c);
    } else {
        base = mdp->GetPolicy(BASE);
    }
    if (BENCH_MODE == 1)
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
    if (METHOD == "ppo")
        std::cout << "updates=" << PPO_NUM_UPDATES
                  << "  temp_anneal=" << (PPO_TEMP_ANNEAL ? 1 : 0)
                  << "  temp_min=" << PPO_TEMP_MIN
                  << "  resets=" << PPO_RESETS
                  << "  avg=" << (PPO_AVG_REWARD ? 1 : 0)
                  << "  lambda=" << PPO_GAE_LAMBDA << "\n";
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
            ppo_cfg.Add("gae_lambda",        PPO_GAE_LAMBDA);
            ppo_cfg.Add("entropy_coef",      PPO_ENTROPY_COEF);
            ppo_cfg.Add("average_reward",    PPO_AVG_REWARD);
            ppo_cfg.Add("rho_step",          PPO_RHO_STEP);
            ppo_cfg.Add("temp_anneal",       PPO_TEMP_ANNEAL);
            ppo_cfg.Add("temp_min",          PPO_TEMP_MIN);
            ppo_cfg.Add("env_reset_every",   PPO_RESETS);
            ppo_cfg.Add("skip_all_bias",     D("skip_bias", 0.0));
            ppo_cfg.Add("value_norm",        I("vnorm", 1) != 0);
            ppo_cfg.Add("dper_clamp",        I("dper", 1) != 0);
            ppo_cfg.Add("guard_tol_sigma",   D("gtol_sigma", 0.0));
            ppo_cfg.Add("guard_robust",      I("grobust", 0) != 0);
            ppo_cfg.Add("guard_leak",        D("gleak", 0.0));
            ppo_cfg.Add("rng_seed",          seed);
            ppo_cfg.Add("silent",            false);   // show training trace for diagnosis
            VarGroup parch; parch.Add("hidden_layers", VarGroup::Int64Vec{64, 32});
            ppo_cfg.Add("nn_architecture", parch);

            auto ppo = dp.GetPPO(mdp, nullptr, ppo_cfg);
            ppo.TrainPolicy();
            auto nn = ppo.GetPolicy();
            auto nn_stoch = ppo.GetStochasticPolicy();

            // Readout battery: one training run, many extraction strategies.
            std::vector<std::string> rnames = {
                "argmax", "stoch T=1", "stoch T=0.1",
                "bias 0.25", "bias 0.5", "bias 1.0",
                "advhead", "advhead b.25" };
            std::vector<DynaPlex::Policy> rpols = {
                nn,
                nn_stoch,
                ppo.GetReadoutPolicy(0.1, 0.0, false),
                ppo.GetReadoutPolicy(0.0, 0.25, false),
                ppo.GetReadoutPolicy(0.0, 0.5,  false),
                ppo.GetReadoutPolicy(0.0, 1.0,  false),
                ppo.GetReadoutPolicy(0.0, 0.0,  true),
                ppo.GetReadoutPolicy(0.0, 0.25, true) };

            double nn_mean = 0.0;
            try {
                std::cout << "  [eval] readout battery (" << rpols.size() << " policies)...\n" << std::flush;
                auto res = comparer.Compare(rpols);
                res[0].Get("mean", nn_mean);
                for (size_t r = 0; r < rpols.size(); ++r) {
                    double m = 0.0; res[r].Get("mean", m);
                    std::cout << "      [" << std::left << std::setw(12) << rnames[r] << "] "
                              << std::fixed << std::setprecision(4)
                              << "NN*Lambda=" << std::setw(10) << m * Lambda
                              << "  NN/RVI=" << m / norm << "\n" << std::flush;
                }
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
            if (PRINT_HEATMAP) {
                std::cout << "\n  PPO policy heatmap [Exp" << EXPERIMENT
                          << ", reward=" << REWARD_TYPE << ", seed=" << seed << "]:\n";
                qm::PrintPolicyHeatmap(mdp, nn, 12);
                std::cout << std::flush;
            }

            // --- optional distillation: one DCL generation from the stochastic
            // PPO policy.  DCL's Q-rollouts + classifier read out the policy's
            // BEHAVIOR (near-optimal) instead of its logit signs, producing a
            // deterministic policy and sidestepping argmax extraction entirely.
            if (PPO_DISTILL) {
                std::cout << "  [distill] DCL gen-1 from stochastic PPO policy...\n" << std::flush;
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

                auto dclo = dp.GetDCL(mdp, nn_stoch, dcl);
                dclo.TrainPolicy();
                auto nn_d = dclo.GetPolicies()[(size_t)1];

                double d_mean = 0.0;
                comparer.Compare({nn_d})[0].Get("mean", d_mean);
                std::cout << "      [distilled] NN*Lambda=" << std::fixed << std::setprecision(4)
                          << d_mean * Lambda
                          << "  NN/RVI=" << d_mean / norm
                          << "   (deterministic classifier from stoch behavior)\n" << std::flush;
                if (PRINT_HEATMAP) {
                    std::cout << "\n  distilled policy heatmap:\n";
                    qm::PrintPolicyHeatmap(mdp, nn_d, 12);
                    std::cout << std::flush;
                }
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

            if (PRINT_HEATMAP) {
                std::cout << "\n  DCL policy heatmap [Exp" << EXPERIMENT
                          << ", reward=" << REWARD_TYPE << ", seed=" << seed << "]:\n";
                qm::PrintPolicyHeatmap(mdp, nn, 12);
                std::cout << std::flush;
            }

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
