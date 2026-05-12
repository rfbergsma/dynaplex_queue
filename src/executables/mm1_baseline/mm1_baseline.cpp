// mm1_baseline.cpp
//
// M/M/1 comparison: Random vs FIFO vs RVI (optimal) vs Neural Network (DCL).
// Runs for multiple tick_rates to verify it is a pure granularity parameter.
//
// Evaluates using EvaluatePolicyRawParallel -> mean_cost_per_event
//   = cost / real_event_steps  (FIL-refresh steps NOT in denominator)
//
// Cost is normalised in the MDP constructor: cost per tick = cost_rate / tick_rate,
// so the physical cost rate is tick_rate-independent.  The mean_cost_per_event
// metric still depends on tick_rate through its denominator (tick events are
// included), giving theory = rho^2 / (tick_rate * Lambda).
// Policy quality (NN/RVI, FIFO%err) must be tick_rate-invariant.
//
// NN trained via DCL from a random starting policy (N=10K, M=100, H=50).

#include <iostream>
#include <iomanip>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/retrievestate.h"           // DynaPlex::Erasure::StateAdapter (heatmap)
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

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
    cfg.Add("reward_type",     int64_t(0));       // binary cost: 1{FIL > D}
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0});   // real-time units; constructor divides by tick_rate
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0});   // real-time seconds; constructor multiplies by tick_rate
    cfg.Add("server_type_0",   srv);
    return cfg;
}

static void print_header(DynaPlex::DynaPlexProvider& dp)
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
// Experiment 2 helpers — 2 job types, 1 pool of 2 servers
// ============================================================

// Config: rho=0.6, lambda1=lambda2=0.6, mu=1, D=0
// c1=1 fixed; c2 and tick_rate are the parameters.
static VarGroup si_config(double c2, double tick_rate)
{
    VarGroup srv;
    srv.Add("servers",      int64_t(2));
    srv.Add("can_serve",    VarGroup::Int64Vec{0, 1});   // fully flexible
    srv.Add("service_rate", 1.0);                         // same rate both types

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     int64_t(0));
    cfg.Add("max_queue_depth", int64_t(1));
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{0.6, 0.6});  // rho = 1.2/2 = 0.6
    cfg.Add("cost_rates",      VarGroup::DoubleVec{1.0, c2});   // real-time units
    cfg.Add("due_times",       VarGroup::DoubleVec{0.0, 0.0});
    cfg.Add("server_type_0",   srv);
    return cfg;
}

// Wraps a DynaPlex::Policy into std::function<int64_t(State&)>
// for use with PrintEnumeratedHeatmap / build_heatmap_grid.
// The Trajectory is heap-allocated, seeded once, and reused across calls.
static std::function<int64_t(const qm::MDP::State&)>
make_raw_policy_fn(const DynaPlex::Policy& pol, const qm::MDP& raw_mdp)
{
    auto traj = std::make_shared<DynaPlex::Trajectory>();
    traj->RNGProvider.SeedEventStreams(false, 42);
    traj->Category = DynaPlex::StateCategory::AwaitAction();
    traj->Reset(
        std::make_unique<DynaPlex::Erasure::StateAdapter<qm::MDP::State>>(
            raw_mdp.int_hash, raw_mdp.GetInitialState()));
    return [pol, traj](const qm::MDP::State& s) -> int64_t {
        auto* a = static_cast<DynaPlex::Erasure::StateAdapter<qm::MDP::State>*>(
            traj->GetState().get());
        a->state = s;
        traj->Category = DynaPlex::StateCategory::AwaitAction();
        pol->SetAction(std::span<DynaPlex::Trajectory>(traj.get(), 1));
        return traj->NextAction;
    };
}

// Builds the heatmap grid by enumerating canonical AwaitAction states.
// Canonical state: pool 0 has exactly 1 server busy on type 0;
// both job types have a waiting job with FIL_0=f0, FIL_1=f1.
// Returns grid[f0][f1]:  -2=no valid action,  -1=skip/idle (action=0),
//                         0=serve type 0,       1=serve type 1.
static std::vector<std::vector<int>>
build_heatmap_grid(
    const qm::MDP& mdp,
    std::function<int64_t(const qm::MDP::State&)> fn,
    int max_fil)
{
    std::vector<std::vector<int>> g(
        (size_t)max_fil + 1, std::vector<int>((size_t)max_fil + 1, -2));
    for (int f0 = 0; f0 <= max_fil; ++f0) {
        for (int f1 = 0; f1 <= max_fil; ++f1) {
            qm::MDP::State s;
            s.queue_manager.initialize(mdp.n_jobs, mdp.tick_rate,
                                       mdp.arrival_rates, mdp.max_queue_depth);
            s.queue_manager.set_fil(0, (int64_t)f0);
            s.queue_manager.set_fil(1, (int64_t)f1);
            s.server_manager.initialize(&mdp.server_static_info, mdp.n_jobs);
            s.server_manager.busy_on[0][0] = 1;    // 1 server busy on type 0
            s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
            s.server_manager.set_action_counter(0);
            s.server_manager.update_total_service_rate();
            s.next_fil_job_type = -1;
            s.cat = DynaPlex::StateCategory::AwaitAction();
            if (!s.server_manager.action_queue.empty()) {
                int64_t act = fn(s);
                g[f0][f1] = (act == 0) ? -1
                            : (int)s.server_manager.action_queue[0].job_type;
            }
        }
    }
    return g;
}

// Prints one heatmap: first an ASCII grid (for quick inspection),
// then a self-contained pgfplots \begin{axis}...\end{axis} block.
// Wrap the axis in \begin{tikzpicture}...\end{tikzpicture} in the document.
// Colormap definition to add to preamble or figure:
//   \pgfplotsset{colormap={simap}{
//     color(0)=(blue!60!white) color(1)=(orange!80!red) color(2)=(gray!50)}}
// Encoding: meta 0 = serve type 0, 1 = serve type 1, 2 = skip/idle/no-action.
static void print_heatmap(
    const qm::MDP& mdp,
    std::function<int64_t(const qm::MDP::State&)> fn,
    const std::string& label,
    int max_fil,
    DynaPlex::DynaPlexProvider& dp)
{
    auto g = build_heatmap_grid(mdp, fn, max_fil);
    int N  = max_fil + 1;

    // ---- ASCII ----
    dp.System() << "\n[" << label << "]  (FIL_1 across, FIL_0 down)\n";
    dp.System() << "FIL_0\\FIL_1";
    for (int f1 = 0; f1 <= max_fil; ++f1)
        dp.System() << std::setw(3) << f1;
    dp.System() << "\n";
    for (int f0 = 0; f0 <= max_fil; ++f0) {
        dp.System() << std::setw(10) << f0 << " :";
        for (int f1 = 0; f1 <= max_fil; ++f1) {
            int v = g[f0][f1];
            if      (v == -2) dp.System() << "  *";
            else if (v == -1) dp.System() << "  .";
            else              dp.System() << "  " << v;
        }
        dp.System() << "\n";
    }
    dp.System() << "(0=serve type 0 [blue], 1=serve type 1 [orange], .=skip top, *=no action)\n";

    // ---- LaTeX pgfplots (row-major: x=FIL_1 varies fastest, y=FIL_0 increases up) ----
    dp.System() << "\n% ---- LaTeX heatmap: " << label << " ----\n";
    dp.System() << "% Wrap each block in \\begin{tikzpicture}...\\end{tikzpicture}\n";
    dp.System() << "% Preamble: \\pgfplotsset{colormap={simap}{\n";
    dp.System() << "%   color(0)=(blue!60!white) color(1)=(orange!80!red) color(2)=(gray!50)}}\n";
    dp.System() << "\\begin{axis}[\n";
    dp.System() << "  title={" << label << "},\n";
    dp.System() << "  xlabel={$W^{(1)}$}, ylabel={$W^{(0)}$},\n";
    dp.System() << "  enlargelimits=false, axis on top,\n";
    dp.System() << "  colormap name=simap,\n";
    dp.System() << "  point meta min=0, point meta max=2,\n";
    dp.System() << "  width=5.5cm, height=5.5cm,\n";
    dp.System() << "  xtick={0,2,...," << max_fil << "},\n";
    dp.System() << "  ytick={0,2,...," << max_fil << "}\n";
    dp.System() << "]\n";
    dp.System() << "\\addplot[\n";
    dp.System() << "  matrix plot*, point meta=explicit,\n";
    dp.System() << "  mesh/rows=" << N << ", mesh/cols=" << N << "\n";
    dp.System() << "] coordinates {\n";
    for (int f0 = 0; f0 <= max_fil; ++f0) {
        for (int f1 = 0; f1 <= max_fil; ++f1) {
            int v    = g[f0][f1];
            int meta = (v == 0) ? 0 : (v == 1) ? 1 : 2;
            dp.System() << "(" << f1 << "," << f0 << ") [" << meta << "] ";
        }
        dp.System() << "\n";
    }
    dp.System() << "};\n";
    dp.System() << "\\end{axis}\n";
}

int main()
{
    auto& dp = DynaPlexProvider::Get();
    const double mu = 1.0;

    dp.System() << "\n=== mm1_baseline: Random / FIFO / RVI / NN ===\n";
    dp.System() << "  reward_type=0 (binary cost), D=0, mu=1\n";
    dp.System() << "  cost_rates and due_times are in real-time units (normalised by tick_rate)\n";
    dp.System() << "  Theory: rho^2 / Lambda,  Lambda = tick_rate + lambda + mu\n";
    dp.System() << "  NN trained via DCL from random policy (N=10K, M=100, H=50)\n";

    for (double tick_rate : {1.0, 2.0, 10.0})
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0) << tick_rate
                    << "  (Lambda = tick_rate + rho*mu + mu) ---\n\n";
        print_header(dp);

        for (double rho : {0.2, 0.4, 0.6, 0.8})
        {
            double lam = rho * mu;
            auto cfg = mm1_config(lam, mu, tick_rate);

            // Type-erased MDP for DynaPlex framework calls
            auto mdp = dp.GetMDP(cfg);

            // ---- policies ----
            auto random = mdp->GetPolicy("random");
            auto fifo   = mdp->GetPolicy("FIFO policy");

            VarGroup rvi_cfg;
            rvi_cfg.Add("id",      std::string("RVI_optimal"));
            rvi_cfg.Add("rel_tol", 0.01);
            rvi_cfg.Add("silent",  int64_t(1));
            auto rvi = mdp->GetPolicy(rvi_cfg);

            // DCL: train NN from random starting policy
            VarGroup nn_arch;
            nn_arch.Add("type",          std::string("mlp"));
            nn_arch.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

            VarGroup dcl_cfg;
            dcl_cfg.Add("N",           int64_t(10000));
            dcl_cfg.Add("M",           int64_t(100));
            dcl_cfg.Add("H",           int64_t(50));
            dcl_cfg.Add("num_gens",    int64_t(1));
            dcl_cfg.Add("silent",      true);
            dcl_cfg.Add("nn_architecture", nn_arch);

            auto dcl = dp.GetDCL(mdp, random, dcl_cfg);
            dcl.TrainPolicy();
            auto nn = dcl.GetPolicies().back();

            // Concrete MDP for evaluation — created AFTER DCL
            qm::MDP raw_mdp(cfg);

            // ---- parallel evaluation ----
            auto eval = [&](DynaPlex::Policy pol) {
                return qm::EvaluatePolicyRawParallel(raw_mdp, pol,
                    /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
            };

            auto r_random = eval(random);
            auto r_fifo   = eval(fifo);
            auto r_rvi    = eval(rvi);
            auto r_nn     = eval(nn);

            // Theory: rho^2 / Lambda
            // mean_cost_per_event = (cost/tick) * P(FIL>0) / Lambda.
            // With normalised cost (1/tick_rate per tick) and uniformisation Lambda=tick_rate+lambda+mu,
            // the product simplifies to rho^2/Lambda (verified empirically; tick_rate-independent formula).
            double theory   = (rho * rho) / raw_mdp.uniformization_rate;
            double fifo_err = (theory > 1e-15)
                            ? (r_fifo.mean_cost_per_event - theory) / theory * 100.0
                            : 0.0;
            double nn_ratio = (r_rvi.mean_cost_per_event > 1e-15)
                            ? r_nn.mean_cost_per_event / r_rvi.mean_cost_per_event
                            : 1.0;

            dp.System() << std::fixed
                        << std::setw(5)  << std::setprecision(1) << rho
                        << std::setw(11) << std::setprecision(6) << r_random.mean_cost_per_event
                        << std::setw(11) << std::setprecision(6) << r_fifo.mean_cost_per_event
                        << std::setw(11) << std::setprecision(6) << r_rvi.mean_cost_per_event
                        << std::setw(11) << std::setprecision(6) << r_nn.mean_cost_per_event
                        << std::setw(9)  << std::setprecision(4) << nn_ratio
                        << std::setw(11) << std::setprecision(6) << theory
                        << std::setw(8)  << std::setprecision(2) << fifo_err << "%"
                        << "\n";
        }
    }

    // ================================================================
    // Experiment 2: Strategic Idleness
    // 2 job types, 1 pool of 2 fully-flexible servers, rho=0.6
    // c1=1 fixed; vary c2 in {1,2,5,10,20}; tick_rates {1,5,10}
    // ================================================================
    dp.System() << "\n\n=== Exp 2: Strategic Idleness (2 types, 2 servers, rho=0.6) ===\n";
    dp.System() << "  c1=1 fixed, D=0, mu=1, rho=0.6 (lambda1=lambda2=0.6)\n";
    dp.System() << "  NN: DCL from random policy (N=10K, M=100, H=50, num_gens=3)\n";

    // State saved for heatmaps (tick_rate=5, c2=20)
    bool             have_saved2  = false;
    VarGroup         saved_cfg2;
    DynaPlex::Policy saved_fifo_si;
    DynaPlex::Policy saved_nn_si;

    for (double tr2 : {1.0, 5.0, 10.0})
    {
        dp.System() << "\n--- tick_rate = " << std::fixed << std::setprecision(0)
                    << tr2 << " ---\n\n";
        dp.System() << std::left
                    << std::setw(6)  << "c2"
                    << std::setw(11) << "FIFO"
                    << std::setw(11) << "NN"
                    << std::setw(9)  << "NN/FIFO"
                    << "\n" << std::string(37, '-') << "\n";

        for (double c2 : {1.0, 2.0, 5.0, 10.0, 20.0})
        {
            auto cfg2    = si_config(c2, tr2);
            auto mdp2    = dp.GetMDP(cfg2);
            qm::MDP raw_mdp2(cfg2);

            auto fifo2   = mdp2->GetPolicy("FIFO policy");
            auto random2 = mdp2->GetPolicy("random");

            VarGroup nn_arch2;
            nn_arch2.Add("type",          std::string("mlp"));
            nn_arch2.Add("hidden_layers", VarGroup::Int64Vec{64, 32, 2});

            VarGroup dcl_cfg2;
            dcl_cfg2.Add("N",               int64_t(10000));
            dcl_cfg2.Add("M",               int64_t(100));
            dcl_cfg2.Add("H",               int64_t(50));
            dcl_cfg2.Add("num_gens",        int64_t(3));
            dcl_cfg2.Add("silent",          true);
            dcl_cfg2.Add("nn_architecture", nn_arch2);

            auto dcl2 = dp.GetDCL(mdp2, random2, dcl_cfg2);
            dcl2.TrainPolicy();
            auto nn2 = dcl2.GetPolicies().back();

            auto eval2 = [&](DynaPlex::Policy pol) {
                return qm::EvaluatePolicyRawParallel(raw_mdp2, pol,
                    /*n_traj=*/100, /*steps=*/500000, /*warmup=*/50000);
            };

            auto r_fifo2 = eval2(fifo2);
            auto r_nn2   = eval2(nn2);

            double ratio2 = (r_fifo2.mean_cost_per_event > 1e-15)
                          ? r_nn2.mean_cost_per_event / r_fifo2.mean_cost_per_event
                          : 1.0;

            dp.System() << std::fixed << std::right
                        << std::setw(5)  << std::setprecision(0) << c2  << " "
                        << std::setw(11) << std::setprecision(6) << r_fifo2.mean_cost_per_event
                        << std::setw(11) << std::setprecision(6) << r_nn2.mean_cost_per_event
                        << std::setw(9)  << std::setprecision(4) << ratio2
                        << "\n";

            // Save tick_rate=5, c2=20 for heatmaps
            if (std::abs(tr2 - 5.0) < 1e-9 && std::abs(c2 - 20.0) < 1e-9)
            {
                saved_cfg2    = cfg2;
                saved_fifo_si = fifo2;
                saved_nn_si   = nn2;
                have_saved2   = true;
            }
        }
    }

    // ---- Heatmaps for tick_rate=5, c2=20 ----
    if (have_saved2)
    {
        constexpr int HEAT_MAX = 12;
        qm::MDP hm(saved_cfg2);

        auto fifo_fn = make_raw_policy_fn(saved_fifo_si, hm);
        auto nn_fn   = make_raw_policy_fn(saved_nn_si,   hm);

        dp.System() << "\n\n=== Heatmaps: tick_rate=5, c2=20, rho=0.6 ===\n";
        dp.System() << "(canonical: 1 server busy on type-0; 1 job of each type waiting)\n";

        print_heatmap(hm, fifo_fn, "FIFO  (gamma=5, c2=20)", HEAT_MAX, dp);
        print_heatmap(hm, nn_fn,   "NN    (gamma=5, c2=20)", HEAT_MAX, dp);
    }

    dp.System() << "\n=== DONE ===\n";
    return 0;
}
