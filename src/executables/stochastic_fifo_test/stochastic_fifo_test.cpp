#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <span>
#include <memory>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// EpsilonGreedyWrapper
//
// Wraps any DynaPlex::Policy with epsilon-greedy exploration:
//   - With probability epsilon: forces action = 0 (skip this candidate).
//   - Otherwise: delegates to the inner policy.
//
// Mechanics: SetAction first lets the inner policy decide all actions, then
// independently draws from each trajectory's dedicated policy RNG stream and
// overrides with 0 when the draw falls below epsilon.  Using the policy RNG
// (not the event RNG) keeps the override reproducible and independent of the
// MDP event streams.
//
// This wrapper is used as the base policy for DCL generation k+1 when the
// gen-k NN would otherwise be fully deterministic and kill exploration.
// ---------------------------------------------------------------------------
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

    std::string TypeIdentifier() const override
    {
        return "epsilon_greedy_wrapper";
    }

    const DynaPlex::VarGroup& GetConfig() const override
    {
        return config_;
    }

    void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override
    {
        // Step 1: let the inner policy decide all actions
        inner_->SetAction(trajectories);
        // Step 2: override with probability epsilon (per-trajectory policy RNG)
        for (auto& traj : trajectories)
        {
            if (traj.RNGProvider.GetPolicyRNG().genUniform() < epsilon_)
                traj.NextAction = 0;   // force "skip this candidate"
        }
    }
};

static DynaPlex::Policy make_epsilon_greedy(DynaPlex::Policy p, double eps)
{
    return std::make_shared<EpsilonGreedyWrapper>(std::move(p), eps);
}

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
    dcl.Add("silent",          false);   // show training progress
    dcl.Add("nn_architecture", nn_arch);
    dcl.Add("nn_training",     nn_training);
    return dcl;
}

// ---------------------------------------------------------------------------
// run_one_config
//
// Runs one (MDP config, tick_rate) block:
//  - Evaluates FIFO and RVI as benchmarks (RVI skipped if !compute_rvi).
//  - FIFO base:  trains num_gens_fifo generations, reports every gen.
//  - For each eps in epsilons: trains num_gens generations, reports every gen.
//
// Each generation is a separate DCL call with num_gens=1.  Between generations
// the trained NN is wrapped in EpsilonGreedyWrapper(eg_epsilon) so that DCL
// sees a stochastic base in generation k+1 instead of a deterministic NN.
//
// H should already be set to 100 * tick_rate by the caller.
// ---------------------------------------------------------------------------
static void run_one_config(
    DynaPlex::DynaPlexProvider& dp,
    const std::string&           section_label,
    VarGroup                     mdp_config,
    int64_t                      N,
    int64_t                      M,
    int64_t                      H,
    int64_t                      num_gens,
    int64_t                      num_gens_fifo,
    const std::vector<double>&   epsilons,
    bool                         compute_rvi,
    const VarGroup&              nn_arch,
    double                       eg_epsilon = 0.10)
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
        auto rvi   = mdp->GetPolicy(rvi_cfg);
        auto bench = comparer.Compare({fifo, rvi});
        bench[0].Get("mean", fifo_mean);
        bench[1].Get("mean", rvi_mean);
    } else {
        auto bench = comparer.Compare({fifo});
        bench[0].Get("mean", fifo_mean);
        rvi_mean = fifo_mean;
    }

    const double norm       = compute_rvi ? rvi_mean : fifo_mean;
    const char*  norm_label = compute_rvi ? "NN/RVI"  : "NN/FIFO";
    const char*  base_label = compute_rvi ? "Base/RVI": "Base/FIFO";

    // --- Section header ---
    dp.System() << "\n" << section_label << "\n";
    dp.System() << std::string(section_label.size(), '-') << "\n";
    dp.System() << "  FIFO = " << std::fixed << std::setprecision(4) << fifo_mean;
    if (compute_rvi)
        dp.System() << "  |  RVI = " << rvi_mean
                    << "  |  FIFO/RVI = " << fifo_mean / rvi_mean
                    << "  (" << std::setprecision(1)
                    << (fifo_mean / rvi_mean - 1.0) * 100.0 << "% gap)";
    dp.System() << "  |  eg_eps=" << std::setprecision(2) << eg_epsilon << "\n\n";

    // --- Table header ---
    dp.System() << std::left
        << std::setw(28) << "Base policy"
        << std::setw(11) << base_label
        << std::setw(5)  << "Gen"
        << std::setw(12) << "NN mean"
        << std::setw(10) << norm_label
        << "\n" << std::string(66, '-') << "\n";

    // Helper: train one DCL round per generation, wrapping the NN with
    // EpsilonGreedyWrapper between generations to maintain exploration.
    auto train_and_print = [&](const std::string&      base_name,
                                double                  base_direct_mean,
                                int64_t                 n_gens,
                                const DynaPlex::Policy& base_policy)
    {
        DynaPlex::Policy current_base = base_policy;

        for (int64_t g = 1; g <= n_gens; ++g) {
            // One generation of DCL from current_base
            auto dcl  = dp.GetDCL(mdp, current_base, make_dcl(N, M, H, 1, nn_arch));
            dcl.TrainPolicy();
            // GetPolicies(): [0] = base, [1] = trained NN
            auto nn_g = dcl.GetPolicies()[(size_t)1];

            // Evaluate
            double nn_mean = 0.0;
            comparer.Compare({nn_g})[0].Get("mean", nn_mean);

            // Print row
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
                            << std::setw(28) << ""
                            << std::setw(11) << ""
                            << std::fixed << std::setprecision(4)
                            << std::setw(5)  << g
                            << std::setw(12) << nn_mean
                            << std::setw(10) << nn_mean / norm
                            << "\n";
            }

            // Decide base for next generation.
            // If the saved checkpoint shows a large train/val gap the NN overfit
            // (low-variance training data from a near-optimal base).  In that case
            // skip the EG wrap and pass the raw NN — adding random skips on top of
            // an overfit model would only make the next generation's data worse.
            if (g < n_gens) {
                double t_loss = 0.0, v_loss = 0.0;
                auto cfg = nn_g.GetConfig();
                cfg.Get("saved_training_loss",   t_loss);
                cfg.Get("saved_validation_loss", v_loss);
                double overfit_gap = v_loss - t_loss;

                if (overfit_gap > 0.01) {
                    dp.System() << "  [overfit guard] gen " << g
                                << "  gap=" << std::fixed << std::setprecision(4)
                                << overfit_gap
                                << "  → skipping EG wrap, passing raw NN\n";
                    current_base = nn_g;
                } else {
                    current_base = make_epsilon_greedy(nn_g, eg_epsilon);
                }
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
    //   H scales with tick_rate (H = 100 * tick_rate).
    //   FIFO base: num_gens=1 (reference).
    //   StochFIFO(eps=0.30): num_gens=3, every gen reported.
    //   EG wrapper between gens: eg_epsilon=0.10.
    //
    //   Config 1: asym_cost_2s — fully flexible, cost=[100,300], D=[6,3]
    //   Config 2: flex_loaded_2s — specialist+generalist, asymmetric costs,
    //             rho_1=0.70 on the generalist → clear routing tension
    // -----------------------------------------------------------------------
    dp.System() << "=== Part 1: Small configs (2-server, 2-job) ===\n";
    dp.System() << "  H = 100 x tick_rate  |  N=20000  M=400  num_gens(stoch)=3  eg_eps=0.10\n";
    dp.System() << "  RVI computed for each tick_rate (configs are small)\n\n";

    struct SmallCfg { std::string name, file; };
    std::vector<SmallCfg> small_configs = {
        {"asym_cost_2s [cost=[100,300], D=[6,3], fully flexible]",
         "mdp_config_asym_cost_2s.json"},
        {"flex_loaded_2s [specialist+generalist, cost=[100,300], rho1=0.70]",
         "mdp_config_flex_loaded_2s.json"}
    };

    std::vector<double> tick_rates_small = {1.0, 3.0, 6.0};
    std::vector<double> epsilons_small   = {0.30};

    for (auto& sc : small_configs) {
        auto base_path   = dp.FilePath({"mdp_config_examples", "queue_mdp"}, sc.file);
        auto base_config = VarGroup::LoadFromFile(base_path);

        for (double tr : tick_rates_small) {
            int64_t H = int64_t(100.0 * tr);

            VarGroup cfg = base_config;
            cfg.Set("tick_rate", tr);

            std::ostringstream lbl;
            lbl << sc.name << "  |  tick_rate=" << std::fixed
                << std::setprecision(0) << tr << "  H=" << H;

            run_one_config(dp,
                lbl.str(), cfg,
                /*N=*/            int64_t(20000),
                /*M=*/            int64_t(400),
                /*H=*/            H,
                /*num_gens=*/     int64_t(3),
                /*num_gens_fifo=*/int64_t(1),
                epsilons_small,
                /*compute_rvi=*/  true,
                nn_arch_small,
                /*eg_epsilon=*/   0.10);
        }
    }

    // -----------------------------------------------------------------------
    // Part 2: Large system — 6 job types, 5 server types, chain skill structure
    //   tick_rate=1  H=100  N=100000  M=1600  num_gens=3  No RVI  eg_eps=0.10
    // -----------------------------------------------------------------------
    dp.System() << "=== Part 2: Large system (6 job types, 5 server types) ===\n";
    dp.System() << "  Chain skill structure: server_k serves jobs {k, k+1}\n";
    dp.System() << "  Simple->complex gradient: mu=[1.2..0.7], cost=[50..200], D=[8..3]\n";
    dp.System() << "  tick_rate=1  H=100  N=100000  M=1600  num_gens=3  No RVI  eg_eps=0.10\n\n";

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
            nn_arch_large,
            /*eg_epsilon=*/    0.10);
    }

    dp.System() << "=== DONE ===\n";
    return 0;
}
