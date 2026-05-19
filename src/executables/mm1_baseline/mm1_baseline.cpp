// mm1_baseline.cpp
//
// Experiments for the paper:
//
//   Experiment 1 (sec:mm1)  — M/M/1 validation.
//     Single job type, single server, D=0, mu=1.
//     Metric: physical cost rate = mean_cost_per_event * Lambda  (tick_rate-invariant).
//     Theory: rho^2.  Compared: Random / FIFO / RVI / NN.
//
//   Experiment 2 (sec:exp2) — Two servers, two job types, asymmetric costs.
//     Fully flexible (both servers serve both types), c=[100,300], D=[6,3].
//     Base: FIFO (1 gen reference) + StochFIFO(0.30) x 3 gens.
//     EpsilonGreedyWrapper wraps the NN between generations (stochastic NN policy).
//     tick_rate=3; values * Lambda = physical cost rate.
//
//   Experiment 3 (sec:exp3) — Two servers, heterogeneous skill sets.
//     Server 0: specialist (type 0 only).  Server 1: generalist (types 0+1).
//     lam=[0.80, 0.60], c=[100,300], D=[5,3].
//     Same training setup as Experiment 2.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <vector>
#include <memory>
#include <span>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// ============================================================
// EpsilonGreedyWrapper
//
// Wraps any Policy with epsilon-greedy exploration:
//   - With probability epsilon: forces action = 0 (skip).
//   - Otherwise: delegates to the inner policy.
// Used between DCL generations to prevent the NN from becoming
// fully deterministic and killing data diversity.
// ============================================================
class EpsilonGreedyWrapper : public DynaPlex::PolicyInterface
{
    DynaPlex::Policy   inner_;
    double             epsilon_;
    DynaPlex::VarGroup config_;
public:
    EpsilonGreedyWrapper(DynaPlex::Policy inner, double epsilon)
        : inner_(std::move(inner)), epsilon_(epsilon)
    {
        config_.Add("id",      std::string("epsilon_greedy_wrapper"));
        config_.Add("epsilon", epsilon_);
    }
    std::string TypeIdentifier() const override { return "epsilon_greedy_wrapper"; }
    const DynaPlex::VarGroup& GetConfig() const override { return config_; }
    void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override
    {
        inner_->SetAction(trajectories);
        for (auto& traj : trajectories)
            if (traj.RNGProvider.GetPolicyRNG().genUniform() < epsilon_)
                traj.NextAction = 0;
    }
};

static DynaPlex::Policy make_epsilon_greedy(DynaPlex::Policy p, double eps)
{
    return std::make_shared<EpsilonGreedyWrapper>(std::move(p), eps);
}

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

static void print_header_exp1(DynaPlex::DynaPlexProvider& dp)
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
// Shared DCL config builder
// (always num_gens=1 per call; the caller loops for multi-gen)
// ============================================================
static VarGroup make_dcl_cfg(int64_t N, int64_t M, int64_t H,
                              const VarGroup& nn_arch,
                              int64_t early_stopping_patience = 3)
{
    VarGroup nn_training;
    nn_training.Add("early_stopping_patience", early_stopping_patience);

    VarGroup dcl;
    dcl.Add("N",               N);
    dcl.Add("M",               M);
    dcl.Add("H",               H);
    dcl.Add("num_gens",        int64_t(1));
    dcl.Add("silent",          false);
    dcl.Add("nn_architecture", nn_arch);
    dcl.Add("nn_training",     nn_training);
    return dcl;
}

// ============================================================
// Experiment 3 config: specialist (type 0) + generalist (types 0+1)
// lam=[0.80, 0.60], c=[100, 300], D=[5, 3], tick_rate overridden externally.
// Total load = 1.40 / capacity 2.0 = rho_total = 0.70.
// Routing tension: type 1 can ONLY use server 1; optimal policy
// reserves server 1 for type 1 when type 1 is waiting.
// ============================================================
static VarGroup make_specialist_generalist_config()
{
    // Server 0: specialist — serves only job type 0
    VarGroup srv0;
    srv0.Add("servers",       int64_t(1));
    srv0.Add("can_serve",     VarGroup::Int64Vec{0});
    srv0.Add("service_rates", VarGroup::DoubleVec{1.0});

    // Server 1: generalist — serves both job types
    VarGroup srv1;
    srv1.Add("servers",       int64_t(1));
    srv1.Add("can_serve",     VarGroup::Int64Vec{0, 1});
    srv1.Add("service_rates", VarGroup::DoubleVec{1.0, 1.0});

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(2));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       1.0);          // overridden by tick_rate loop
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{0.85, 0.70});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{100.0, 300.0});
    cfg.Add("due_times",       VarGroup::DoubleVec{2.0, 1.0});
    cfg.Add("server_type_0",   srv0);
    cfg.Add("server_type_1",   srv1);
    return cfg;
}

// ============================================================
// run_stoch_fifo_experiment
//
// Runs Experiments 2 and 3, looping over tick_rates.
// For each tick_rate:
//   - Overrides tick_rate in mdp_config_base and scales H = base_H * tick_rate.
//   - Obtains Lambda from the raw MDP (uniformization_rate).
//   - Evaluates FIFO and RVI; prints heatmaps on the FIRST tick_rate only.
//   - Reports all absolute costs as  value * Lambda  (physical cost rate per
//     unit real time — tick-rate invariant, so values repeat across tick rates).
//   - Trains DCL from FIFO base (1 gen reference) and each StochFIFO base
//     (num_gens gens), wrapping the NN with EpsilonGreedyWrapper between gens.
//   - Prints final NN heatmap after each variant's last generation (first
//     tick_rate only).
// ============================================================
static void run_stoch_fifo_experiment(
    DynaPlex::DynaPlexProvider& dp,
    const std::string&          section_label,
    VarGroup                    mdp_config_base,   // tick_rate overridden per iteration
    int64_t N, int64_t M, int64_t base_H,         // H = base_H * tick_rate
    int64_t num_gens,
    bool    use_rel_tol,
    int64_t rvi_M_fixed,
    const VarGroup&             nn_arch,
    const std::vector<double>&  stoch_epsilons,
    const std::vector<double>&  tick_rates,        // e.g. {1.0, 3.0}
    double                      eg_epsilon     = 0.10,
    bool                        print_heatmaps = true)
{
    // Print outer section header once
    dp.System() << "\n" << section_label << "\n"
                << std::string(section_label.size(), '-') << "\n";

    for (size_t ti = 0; ti < tick_rates.size(); ++ti)
    {
        const double tr        = tick_rates[ti];
        const bool   first_tr  = (ti == 0);
        const int64_t H        = int64_t(base_H * tr);

        // Override tick_rate in config
        VarGroup mdp_config = mdp_config_base;
        mdp_config.Set("tick_rate", tr);

        // Get Lambda from the raw (non-type-erased) MDP
        qm::MDP raw_mdp(mdp_config);
        const double Lambda = raw_mdp.uniformization_rate;

        auto mdp = dp.GetMDP(mdp_config);

        VarGroup eval_cfg;
        eval_cfg.Add("number_of_trajectories", int64_t(100));
        eval_cfg.Add("periods_per_trajectory",  int64_t(500000));
        auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);

        // --- Benchmarks ---
        auto fifo = mdp->GetPolicy("FIFO policy");

        VarGroup rvi_cfg;
        if (use_rel_tol)
            rvi_cfg = VarGroup{{"id",     std::string("RVI_optimal")},
                               {"rel_tol", 0.01},
                               {"silent",  int64_t(1)}};
        else
            rvi_cfg = VarGroup{{"id",    std::string("RVI_optimal")},
                               {"M",      rvi_M_fixed},
                               {"silent", int64_t(1)}};
        auto rvi = mdp->GetPolicy(rvi_cfg);

        auto bench = comparer.Compare({fifo, rvi});
        double fifo_mean = 0.0, rvi_mean = 0.0;
        bench[0].Get("mean", fifo_mean);
        bench[1].Get("mean", rvi_mean);

        // Physical cost rates (per unit real time)
        const double fifo_phys = fifo_mean * Lambda;
        const double rvi_phys  = rvi_mean  * Lambda;
        const double norm      = rvi_mean;   // normalise ratios by raw RVI (Lambda cancels)

        // --- Tick-rate sub-header ---
        dp.System() << "\n  [tick_rate=" << std::fixed << std::setprecision(0) << tr
                    << "  Lambda=" << std::setprecision(3) << Lambda
                    << "  H=" << H << "]\n";
        dp.System() << "  FIFO*Λ = " << std::fixed << std::setprecision(4) << fifo_phys
                    << "  |  RVI*Λ = " << rvi_phys
                    << "  |  FIFO/RVI = " << fifo_mean / rvi_mean
                    << "  (" << std::setprecision(1)
                    << (fifo_mean / rvi_mean - 1.0) * 100.0 << "% gap)"
                    << "  |  eg_eps=" << std::setprecision(2) << eg_epsilon << "\n\n";

        // --- Heatmaps for FIFO and RVI (first tick_rate only) ---
        if (print_heatmaps && first_tr) {
            dp.System() << "  FIFO policy  (FIL_0=row, FIL_1=col; 0=skip, 1=serve type0, 2=serve type1):\n";
            qm::PrintPolicyHeatmap(mdp, fifo, 12);
            dp.System() << "\n  RVI optimal policy:\n";
            qm::PrintPolicyHeatmap(mdp, rvi, 12);
            dp.System() << "\n";
        }

        // --- Table header ---
        dp.System() << std::left
            << std::setw(28) << "Base policy"
            << std::setw(11) << "Base/RVI"
            << std::setw(5)  << "Gen"
            << std::setw(12) << "NN*Lambda"
            << std::setw(10) << "NN/RVI"
            << "\n" << std::string(66, '-') << "\n";

        // --- Training loop ---
        auto train_and_print = [&](const std::string&      base_name,
                                    double                  base_direct_mean,
                                    int64_t                 n_gens,
                                    const DynaPlex::Policy& base_policy)
        {
            DynaPlex::Policy current_base = base_policy;

            for (int64_t g = 1; g <= n_gens; ++g) {
                auto dcl = dp.GetDCL(mdp, current_base,
                                      make_dcl_cfg(N, M, H, nn_arch));
                dcl.TrainPolicy();
                auto nn_g = dcl.GetPolicies()[(size_t)1];

                double nn_mean = 0.0;
                comparer.Compare({nn_g})[0].Get("mean", nn_mean);
                const double nn_phys = nn_mean * Lambda;

                if (g == 1)
                    dp.System() << std::left  << std::setw(28) << base_name
                                << std::fixed << std::setprecision(4)
                                << std::setw(11) << base_direct_mean / norm
                                << std::setw(5)  << g
                                << std::setw(12) << nn_phys
                                << std::setw(10) << nn_mean / norm << "\n";
                else
                    dp.System() << std::left
                                << std::setw(28) << "" << std::setw(11) << ""
                                << std::fixed << std::setprecision(4)
                                << std::setw(5)  << g
                                << std::setw(12) << nn_phys
                                << std::setw(10) << nn_mean / norm << "\n";

                if (g < n_gens) {
                    // Overfitting guard: skip EG wrap if val-train gap > 0.01
                    double t_loss = 0.0, v_loss = 0.0;
                    auto cfg = nn_g->GetConfig();
                    cfg.Get("saved_training_loss",   t_loss);
                    cfg.Get("saved_validation_loss", v_loss);
                    double gap = v_loss - t_loss;
                    if (gap > 0.01) {
                        dp.System() << "  [overfit guard] gen " << g
                                    << "  gap=" << std::fixed << std::setprecision(4)
                                    << gap << "  -> passing raw NN\n";
                        current_base = nn_g;
                    } else {
                        current_base = make_epsilon_greedy(nn_g, eg_epsilon);
                    }
                } else if (g == n_gens && print_heatmaps && first_tr) {
                    // NN heatmap: first tick_rate only, after final generation
                    dp.System() << "\n  NN policy heatmap [" << base_name
                                << ", gen " << g << "]:\n";
                    qm::PrintPolicyHeatmap(mdp, nn_g, 12);
                    dp.System() << "\n";
                }
            }
        };

        // FIFO base: 1 generation (reference point only)
        train_and_print("FIFO (eps=0.00)", fifo_mean, /*n_gens=*/1, fifo);

        // Stochastic FIFO variants: num_gens generations each
        for (double eps : stoch_epsilons) {
            VarGroup stoch_cfg;
            stoch_cfg.Add("id",        std::string("stochastic_FIFO"));
            stoch_cfg.Add("threshold", eps);
            auto stoch = mdp->GetPolicy(stoch_cfg);

            double base_mean = 0.0;
            comparer.Compare({stoch})[0].Get("mean", base_mean);

            std::ostringstream lbl;
            lbl << "StochFIFO(eps="
                << std::fixed << std::setprecision(2) << eps << ")";
            train_and_print(lbl.str(), base_mean, num_gens, stoch);
        }

    } // end tick_rate loop

    dp.System() << "\n";
}

// ============================================================
// run_config_experiment  (retained for Experiment 4, disabled by default)
// ============================================================
static void run_config_experiment(
    DynaPlex::DynaPlexProvider& dp,
    const std::string& label,
    const std::string& json_file,
    bool   use_rel_tol,
    int64_t rvi_M_fixed,
    int64_t N, int64_t H, int64_t M_dcl,
    int64_t reward_type       = int64_t(0),
    int64_t num_gens          = int64_t(1),
    bool    print_heatmap     = false,
    int64_t early_stop        = int64_t(0))
{
    auto path       = dp.FilePath({"mdp_config_examples", "queue_mdp"}, json_file);
    auto mdp_config = VarGroup::LoadFromFile(path);
    mdp_config.Set("reward_type", reward_type);
    auto mdp        = dp.GetMDP(mdp_config);

    auto fifo = mdp->GetPolicy("FIFO policy");

    VarGroup rvi_cfg;
    if (use_rel_tol)
        rvi_cfg = VarGroup{{"id", std::string("RVI_optimal")}, {"rel_tol", 0.01},
                           {"silent", int64_t(1)}};
    else
        rvi_cfg = VarGroup{{"id", std::string("RVI_optimal")}, {"M", rvi_M_fixed},
                           {"silent", int64_t(1)}};
    auto rvi = mdp->GetPolicy(rvi_cfg);

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

    if (print_heatmap) {
        dp.System() << "\n  [" << label << "] FIFO policy heatmap"
                    << " (FIL_0=row, FIL_1=col; 0=skip, 1=type0, 2=type1):\n";
        qm::PrintPolicyHeatmap(mdp, fifo, /*max_fil=*/12);
        dp.System() << "\n  [" << label << "] RVI optimal policy heatmap:\n";
        qm::PrintPolicyHeatmap(mdp, rvi,  /*max_fil=*/12);
        dp.System() << "\n  [" << label << "] NN policy heatmap:\n";
        qm::PrintPolicyHeatmap(mdp, nn,   /*max_fil=*/12);
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

    // ---- Run-control flags ----
    const bool run_exp1      = true;
    const bool run_exp2      = true;
    const bool run_exp3_grid = true;   // fast FIFO/RVI gap scan (no NN training)
    const bool run_exp3      = true;   // full DCL training with chosen config
    const bool run_exp4      = false;  // queue-lateness reward, num_gens=3 (slow)

    // Shared NN architecture for Experiments 2 and 3.
    // {64,32,2} matches stochastic_fifo_test and reliably converges.
    // {128,64,2} collapses to 50% accuracy and learns nothing.
    VarGroup nn_arch;
    nn_arch.Add("type",          std::string("mlp"));
    nn_arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

    // tick_rate=3 for Experiments 2 and 3.
    // At tick_rate=3 the trajectory length H=300 gives enough routing decisions per
    // sample for stable multi-generation training.  (tick_rate=1 with H=100 produces
    // too few routing events per trajectory, leading to Gen 3 instability.)
    const std::vector<double> tick_rates_exp = {3.0};

    // ----------------------------------------------------------
    // Experiment 1: M/M/1 validation
    // ----------------------------------------------------------
  if (run_exp1) {
    dp.System() << "\n=== Experiment 1: M/M/1 validation ===\n";
    dp.System() << "  reward_type=0 (binary cost 1{wait>0}), D=0, mu=1\n";
    dp.System() << "  Metric: physical cost rate = mean_cost_per_event * Lambda  (tick_rate-invariant)\n";
    dp.System() << "  Theory: rho^2  (= lambda * P(customer waits) for M/M/1)\n";
    dp.System() << "  DCL: N=10K, M=400, H=50, num_gens=1, arch={64,32,2}\n";

    VarGroup nn_arch_small;
    nn_arch_small.Add("type",          std::string("mlp"));
    nn_arch_small.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

    for (double tick_rate : {1.0, 2.0, 10.0})
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0)
                    << tick_rate << " ---\n\n";
        print_header_exp1(dp);

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

            VarGroup dcl_cfg;
            dcl_cfg.Add("N",               int64_t(10000));
            dcl_cfg.Add("M",               int64_t(400));
            dcl_cfg.Add("H",               int64_t(50));
            dcl_cfg.Add("num_gens",        int64_t(1));
            dcl_cfg.Add("silent",          true);
            dcl_cfg.Add("nn_architecture", nn_arch_small);

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
            double rate_random = r_random.mean_cost_per_event * Lambda;
            double rate_fifo   = r_fifo.mean_cost_per_event   * Lambda;
            double rate_rvi    = r_rvi.mean_cost_per_event    * Lambda;
            double rate_nn     = r_nn.mean_cost_per_event     * Lambda;

            double theory   = rho * rho;
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
    // Experiment 2: two servers, two job types, asymmetric costs
    // Fully flexible — both servers can serve both job types.
    // Reported for tick_rates = {1, 3}; values * Lambda are
    // physical cost rates and should be tick-rate invariant.
    // ----------------------------------------------------------
  if (run_exp2) {
    dp.System() << "\n\n=== Experiment 2: Asymmetric Costs (two servers, fully flexible) ===\n";
    dp.System() << "  Config: asym_cost_2s  (k=2, n=2, fully flexible, lam=[0.25,0.25], c=[100,300], D=[6,3])\n";
    dp.System() << "  Equal arrivals but type 1 costs 3x more with tighter deadline.\n";
    dp.System() << "  FIFO ignores costs → large gap. StochFIFO+EG reliably solves to near-optimality.\n";
    dp.System() << "  tick_rate=3  H=300  N=20K  M=400  num_gens=3  eg_eps=0.10\n";
    dp.System() << "  Base: FIFO (1 gen reference) + StochFIFO(0.30) x 3 gens\n";
    dp.System() << "  StochFIFO skips each candidate with P=0.30 (exploration);\n";
    dp.System() << "  between gens the NN is wrapped with EpsilonGreedy(0.10) (stochastic NN).\n";

    auto path2 = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                              "mdp_config_asym_cost_2s.json");
    auto cfg2  = VarGroup::LoadFromFile(path2);

    run_stoch_fifo_experiment(dp,
        "asym_cost_2s  [k=2, n=2, fully flexible, lam=[0.25,0.25], c=[100,300], D=[6,3]]",
        cfg2,
        /*N=*/       int64_t(20000),
        /*M=*/       int64_t(400),
        /*base_H=*/  int64_t(100),
        /*num_gens=*/int64_t(3),
        /*rel_tol=*/ true,
        /*rvi_M=*/   int64_t(0),
        nn_arch,
        /*epsilons=*/std::vector<double>{0.30},
        tick_rates_exp,
        /*eg_eps=*/  0.10,
        /*heatmaps=*/true);
  } // end run_exp2

    // ----------------------------------------------------------
    // Experiment 3 grid search: find (lam1, D1) with largest
    // FIFO/RVI gap for the specialist-generalist structure.
    // Fixed: lam0=0.6, D0=6, c=[100,300], mu=[1,1], tick_rate=3
    // No NN training — just FIFO vs RVI for each cell.
    // ----------------------------------------------------------
  if (run_exp3_grid) {
    dp.System() << "\n\n=== Experiment 3 grid search: specialist+generalist ===\n";
    dp.System() << "  Server 0: specialist (type 0 only)  Server 1: generalist (both)\n";
    dp.System() << "  Fixed: lam0=0.6  D0=6.0  c=[100,300]  mu=[1,1]  tick_rate=3\n";
    dp.System() << "  Cells show FIFO/RVI gap (%) — pick the largest for full training.\n\n";

    const double   grid_lam0 = 0.6;
    const double   grid_D0   = 6.0;
    const double   grid_tr   = 3.0;
    const std::vector<double> lam1_vals = {0.4, 0.6, 0.8};
    const std::vector<double> D1_vals   = {0.3, 0.7, 1.5};

    // Evaluation config — fewer periods for speed
    VarGroup grid_eval;
    grid_eval.Add("number_of_trajectories", int64_t(50));
    grid_eval.Add("periods_per_trajectory",  int64_t(200000));

    // Table header
    dp.System() << std::left << std::setw(12) << "  lam1 \\ D1";
    for (double d1 : D1_vals)
        dp.System() << std::right << std::fixed << std::setprecision(1)
                    << std::setw(12) << ("D1=" + std::to_string(d1).substr(0,3));
    dp.System() << "\n  " << std::string(12 + 12*(int)D1_vals.size(), '-') << "\n";

    for (double lam1 : lam1_vals) {
        dp.System() << "  lam1=" << std::fixed << std::setprecision(1) << lam1 << "   |";

        for (double D1 : D1_vals) {
            // Build specialist-generalist config for this cell
            VarGroup srv0g, srv1g;
            srv0g.Add("servers",       int64_t(1));
            srv0g.Add("can_serve",     VarGroup::Int64Vec{0});
            srv0g.Add("service_rates", VarGroup::DoubleVec{1.0});
            srv1g.Add("servers",       int64_t(1));
            srv1g.Add("can_serve",     VarGroup::Int64Vec{0, 1});
            srv1g.Add("service_rates", VarGroup::DoubleVec{1.0, 1.0});

            VarGroup gcfg;
            gcfg.Add("id",              std::string("queue_mdp"));
            gcfg.Add("discount_factor", 1.0);
            gcfg.Add("k_servers",       int64_t(2));
            gcfg.Add("n_jobs",          int64_t(2));
            gcfg.Add("tick_rate",       grid_tr);
            gcfg.Add("reward_type",     int64_t(0));
            gcfg.Add("arrival_rates",   VarGroup::DoubleVec{grid_lam0, lam1});
            gcfg.Add("cost_rates",      VarGroup::DoubleVec{100.0, 300.0});
            gcfg.Add("due_times",       VarGroup::DoubleVec{grid_D0, D1});
            gcfg.Add("server_type_0",   srv0g);
            gcfg.Add("server_type_1",   srv1g);

            auto gmdp = dp.GetMDP(gcfg);
            auto gcomp = dp.GetPolicyComparer(gmdp, grid_eval);

            auto gfifo = gmdp->GetPolicy("FIFO policy");
            VarGroup grvi_cfg;
            grvi_cfg.Add("id",      std::string("RVI_optimal"));
            grvi_cfg.Add("silent",  int64_t(1));
            grvi_cfg.Add("rel_tol", 0.01);
            auto grvi = gmdp->GetPolicy(grvi_cfg);

            auto gres = gcomp.Compare({gfifo, grvi});
            double gfifo_mean = 0.0, grvi_mean = 0.0;
            gres[0].Get("mean", gfifo_mean);
            gres[1].Get("mean", grvi_mean);
            double gap_pct = (grvi_mean > 1e-15)
                           ? (gfifo_mean / grvi_mean - 1.0) * 100.0 : 0.0;

            dp.System() << std::fixed << std::setprecision(1)
                        << std::setw(10) << gap_pct << "%  ";
        }
        dp.System() << "\n";
    }
    dp.System() << "\n";
  } // end run_exp3_grid

    // ----------------------------------------------------------
    // Experiment 3: two servers, heterogeneous skill sets
    // Server 0 is a specialist (type 0 only); server 1 is a
    // generalist (types 0+1).  lam=[0.80,0.60] raises load so
    // that FIFO's failure to reserve server 1 for type 1 is
    // costly.  Same tick_rate sweep as Experiment 2.
    // ----------------------------------------------------------
  if (run_exp3) {
    dp.System() << "\n\n=== Experiment 3: Heterogeneous skill sets (specialist + generalist) ===\n";
    dp.System() << "  Server 0: specialist — serves only job type 0.\n";
    dp.System() << "  Server 1: generalist — serves both job types.\n";
    dp.System() << "  lam=[0.85, 0.70], c=[100, 300], D=[2, 1]  (rho_total=0.775)\n";
    dp.System() << "  Tight deadlines (D*tick_rate = 6 and 3 ticks) combined with high\n";
    dp.System() << "  utilisation create frequent routing conflicts: FIFO assigns server 1\n";
    dp.System() << "  to type 0 when FIL_0 > FIL_1, repeatedly pushing type 1 past its deadline.\n";
    dp.System() << "  tick_rate=3  H=300  N=20K  M=400  num_gens=3  eg_eps=0.10\n";
    dp.System() << "  Base: FIFO (1 gen reference) + StochFIFO(0.30) x 3 gens\n";

    run_stoch_fifo_experiment(dp,
        "specialist_gen  [srv0=type0_only, srv1=both, lam=[0.85,0.70], c=[100,300], D=[2,1]]",
        make_specialist_generalist_config(),
        /*N=*/       int64_t(20000),
        /*M=*/       int64_t(400),
        /*base_H=*/  int64_t(100),
        /*num_gens=*/int64_t(3),
        /*rel_tol=*/ true,
        /*rvi_M=*/   int64_t(0),
        nn_arch,
        /*epsilons=*/std::vector<double>{0.30},
        tick_rates_exp,
        /*eg_eps=*/  0.10,
        /*heatmaps=*/true);
  } // end run_exp3

    // ----------------------------------------------------------
    // Experiment 4: queue-lateness reward (reward_type=1), num_gens=3
    // Disabled by default — slow; re-enable to compare reward types.
    // ----------------------------------------------------------
  if (run_exp4) {
    dp.System() << "\n\n=== Experiment 4: Queue-lateness reward (reward_type=1), num_gens=3 ===\n";
    dp.System() << "  Same configs as Exp 2/3; reward_type=1 (cost proportional to FIL-D per tick)\n";
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

    run_config_experiment(dp, "simple_asym",
        "mdp_config_simple_asym.json",
        /*rel_tol=*/true,  /*rvi_M=*/int64_t(0),
        /*N=*/int64_t(20000), /*H=*/int64_t(100), /*M=*/int64_t(1600),
        /*reward_type=*/int64_t(1), /*num_gens=*/int64_t(3));
  } // end run_exp4

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
