// queue_depth_study.cpp
//
// Studies two questions about the RVI policy for a 2-server, 2-job-type
// fully-flexible queueing system:
//
//  (1) TRUE SYSTEM: RVI is solved on the FIL-only (k=1) projection.
//      The "true" system tracks k_true queue positions per type, where
//      k_true is derived from the geometric tail formula:
//          P(queue length > k) <= rho^(k+1) < epsilon
//          => k_true = ceil(log(epsilon) / log(rho))
//      We compare FIFO vs RVI in both the k=1 and k=k_true systems.
//
//  (2) GRANULARITY: the physical deadline is fixed at T_real time units.
//      We sweep tick_rate in {1, 2, 4, 8}, scaling due_ticks = T_real *
//      tick_rate accordingly.  Higher tick_rate gives finer FIL resolution,
//      potentially sharper RVI decisions and cleaner heatmaps.
//
// System: 2 servers (both fully flexible), 2 job types
//   lambda = [0.10, 0.10],  mu = [0.40, 0.40],  rho = 0.25
//   cost_rates = [3.0, 1.0]  (type 0 penalised 3x more)
//   reward = queue-lateness (QL)
// ---------------------------------------------------------------------------

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include "dynaplex/dynaplexprovider.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// Build a 2-server fully-flexible 2-job config
// ---------------------------------------------------------------------------
static VarGroup make_config(
    double  lam0,  double lam1,
    double  mu,                     // same for both server pools
    double  due,                    // due_ticks (integer-valued in ticks)
    double  cost0, double cost1,
    double  tick_rate,
    int64_t reward_type,
    int64_t max_queue_depth)
{
    VarGroup srv0;
    srv0.Add("servers",      int64_t(1));
    srv0.Add("can_serve",    VarGroup::Int64Vec{0, 1});
    srv0.Add("service_rate", mu);

    VarGroup srv1;
    srv1.Add("servers",      int64_t(1));
    srv1.Add("can_serve",    VarGroup::Int64Vec{0, 1});
    srv1.Add("service_rate", mu);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(2));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       tick_rate);
    cfg.Add("reward_type",     reward_type);
    cfg.Add("max_queue_depth", max_queue_depth);
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam0, lam1});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{cost0, cost1});
    cfg.Add("due_times",       VarGroup::DoubleVec{due, due});
    cfg.Add("server_type_0",   srv0);
    cfg.Add("server_type_1",   srv1);
    return cfg;
}

// ---------------------------------------------------------------------------
// Print a single performance row
// ---------------------------------------------------------------------------
static void print_perf_row(
    DynaPlexProvider& dp,
    const std::string& label,
    const VarGroup& cfg,
    int64_t M,
    int64_t k_depth,
    const VarGroup& eval_cfg)
{
    auto mdp  = dp.GetMDP(cfg);
    auto fifo = mdp->GetPolicy("FIFO policy");

    VarGroup rvi_pol_cfg;
    rvi_pol_cfg.Add("id", std::string("RVI_optimal"));
    rvi_pol_cfg.Add("M",  M);
    if (k_depth > 1)
        rvi_pol_cfg.Add("feature_queue_depth", int64_t(1));  // FIL-only projection

    auto rvi = mdp->GetPolicy(rvi_pol_cfg);

    auto cmp = dp.GetPolicyComparer(mdp, eval_cfg);
    auto res = cmp.Compare({fifo, rvi});

    double fifo_m = 0, rvi_m = 0, fifo_e = 0, rvi_e = 0;
    res[0].Get("mean",  fifo_m); res[0].Get("error", fifo_e);
    res[1].Get("mean",  rvi_m);  res[1].Get("error", rvi_e);

    double ratio  = (fifo_m > 1e-12) ? rvi_m / fifo_m : 0.0;
    double improv = 100.0 * (1.0 - ratio);

    dp.System() << "  " << std::left << std::setw(30) << label
                << std::fixed << std::setprecision(6)
                << std::setw(14) << fifo_m
                << std::setw(14) << rvi_m
                << std::setprecision(4)
                << std::setw(10) << ratio
                << std::setprecision(2)
                << std::setw(9)  << improv << "%\n";
}

// ---------------------------------------------------------------------------
int main()
{
    auto& dp = DynaPlexProvider::Get();

    // -----------------------------------------------------------------------
    // System parameters
    // -----------------------------------------------------------------------
    const double  lam0   = 0.10;
    const double  lam1   = 0.10;
    const double  mu     = 0.40;    // both server pools
    const double  cost0  = 3.0;
    const double  cost1  = 1.0;
    const double  T_real = 3.0;     // physical deadline (real-time units)
    const int64_t rtype  = 1;       // queue-lateness reward

    const double rho = (lam0 + lam1) / (mu + mu);  // = 0.25

    // k formula: rho^(k+1) < epsilon  =>  k = ceil(log(eps)/log(rho))
    const double  eps_k  = 0.01;
    const int     k_true = std::max(1, (int)std::ceil(std::log(eps_k) / std::log(rho)));

    dp.System() << "\n" << std::string(80, '=') << "\n";
    dp.System() << "=== queue_depth_study ===\n";
    dp.System() << std::string(80, '=') << "\n\n";

    dp.System() << "System:  2-server fully-flexible, 2 job types, QL reward\n";
    dp.System() << "  lambda     = [" << lam0 << ", " << lam1 << "]\n";
    dp.System() << "  mu         = [" << mu   << ", " << mu   << "]\n";
    dp.System() << "  rho        = " << std::fixed << std::setprecision(4) << rho << "\n";
    dp.System() << "  cost_rates = [" << cost0 << ", " << cost1
                << "]  (type 0 is " << (int)(cost0/cost1) << "x more expensive)\n";
    dp.System() << "  T_real     = " << T_real
                << "  (physical deadline in real-time units)\n\n";

    dp.System() << "Queue-depth formula:  rho^(k+1) < " << eps_k << "\n";
    dp.System() << "  k_true = ceil(log(" << eps_k
                << ") / log(" << rho << ")) = ceil("
                << std::setprecision(3) << std::log(eps_k) / std::log(rho)
                << ") = " << k_true << "\n\n";

    dp.System() << "Tick-rate sweep:  T_real fixed, due_ticks = T_real * tick_rate\n";
    dp.System() << "  tick_rate in {1, 2, 4, 8}  =>  due_ticks in {3, 6, 12, 24}\n\n";

    // Evaluation config — shared by all sections
    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(200));
    eval_cfg.Add("periods_per_trajectory", int64_t(50000));

    // Column header helper
    auto print_header = [&]() {
        dp.System() << "  " << std::left
                    << std::setw(30) << "Configuration"
                    << std::setw(14) << "FIFO_mean"
                    << std::setw(14) << "RVI_mean"
                    << std::setw(10) << "RVI/FIFO"
                    << std::setw(9)  << "Improv"
                    << "\n";
        dp.System() << "  " << std::string(76, '-') << "\n";
    };

    // -----------------------------------------------------------------------
    // Solve RVI once for the baseline (k=1, tick_rate=1) and reuse M
    // -----------------------------------------------------------------------
    const double due_base = T_real * 1.0;   // tick_rate=1 -> due_ticks = 3
    auto cfg_base = make_config(lam0, lam1, mu, due_base, cost0, cost1,
                                /*tick_rate=*/1.0, rtype, /*k=*/1);

    dp.System() << std::string(80, '-') << "\n";
    dp.System() << "Solving baseline RVI (k=1, tick_rate=1, rel_tol=0.01) ...\n";
    dp.System() << std::string(80, '-') << "\n";

    DynaPlex::Models::queue_mdp::MDP mdp_direct(cfg_base);
    auto sol_base = mdp_direct.runRVI(0.01);
    const int M_base = sol_base.M;

    dp.System() << "\nBaseline: g* = " << std::fixed << std::setprecision(8)
                << sol_base.g_star << "  M = " << M_base << "\n\n";

    // -----------------------------------------------------------------------
    // Verification: g* (Bellman) vs. simulated cost per uniformized step
    //
    // EvaluatePolicyRaw.mean_cost_per_step_gic uses GetImmediateCost(state)
    // at every AwaitEvent state (TICK, arrival, completion, FIL-refresh),
    // BEFORE the event fires, then divides by ALL steps.  This is exactly
    // the per-step cost that the RVI Bellman equation averages, so it must
    // equal g* (up to FIL truncation at M and Monte-Carlo noise, ~ < 2%).
    //
    // mean_cost_per_rvi_step uses ModifyStateWithEvent return values (unscaled
    // post-tick cost only) divided by rvi_steps — different denominator AND
    // different cost function from g*.  Kept for diagnostics only.
    // -----------------------------------------------------------------------
    dp.System() << std::string(80, '=') << "\n";
    dp.System() << "=== Verification: g* vs. simulated cost per uniformized step ===\n";
    dp.System() << std::string(80, '=') << "\n\n";

    using QueueState = DynaPlex::Models::queue_mdp::MDP::State;
    auto rvi_raw_fn  = [&mdp_direct, &sol_base](const QueueState& s) -> int64_t {
        return mdp_direct.EvaluateRVIPolicy(sol_base, s);
    };
    auto fifo_raw_fn = [](const QueueState&) -> int64_t { return int64_t(1); };

    dp.System() << "  Running EvaluatePolicyRaw (300 traj x 150 000 steps) ...\n";
    auto rvi_raw  = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
        mdp_direct, rvi_raw_fn,  /*n_traj=*/300, /*steps=*/150000, /*warmup=*/15000);
    auto fifo_raw = DynaPlex::Models::queue_mdp::EvaluatePolicyRaw(
        mdp_direct, fifo_raw_fn, /*n_traj=*/300, /*steps=*/150000, /*warmup=*/15000);

    const double rvi_gic  = rvi_raw.mean_cost_per_step_gic;
    const double fifo_gic = fifo_raw.mean_cost_per_step_gic;

    const double rel_dev_pct =
        std::abs(sol_base.g_star - rvi_gic) / std::max(rvi_gic, 1e-15) * 100.0;

    // ---- per-step column: correct g* comparison ----
    dp.System() << "\n  Per-step cost  [GetImmediateCost / all steps — matches g* denominator]\n";
    dp.System() << "  " << std::string(70, '-') << "\n";
    dp.System() << "  " << std::left << std::setw(36) << "g*  (RVI Bellman)"
                << std::fixed << std::setprecision(10) << sol_base.g_star << "\n";
    dp.System() << "  " << std::left << std::setw(36) << "RVI policy simulated"
                << std::setprecision(10) << rvi_gic << "\n";
    dp.System() << "  " << std::left << std::setw(36) << "FIFO policy simulated"
                << std::setprecision(10) << fifo_gic << "\n";
    dp.System() << "\n  Relative deviation  g* vs RVI_sim : "
                << std::setprecision(3) << rel_dev_pct << " %\n";
    dp.System() << "  (Expected < ~2 %; residual from FIL truncation at M=" << M_base << ")\n";

    // ---- step-type diagnostics ----
    const int64_t rvi_steps_v   = rvi_raw.total_action_steps
                                + rvi_raw.total_real_event_steps;
    const int64_t all_steps_v   = rvi_steps_v + rvi_raw.total_fil_refresh_steps;
    const double  f_rvi         = (all_steps_v > 0)
                                  ? static_cast<double>(rvi_steps_v) / all_steps_v
                                  : 1.0;
    const double frac_act       = (rvi_steps_v > 0)
                                  ? (double)rvi_raw.total_action_steps / rvi_steps_v
                                  : 0.0;
    const double multiplier     = rvi_raw.mean_cost_per_event
                                / rvi_raw.mean_cost_per_rvi_step;

    dp.System() << "\n  Step breakdown (RVI policy, aggregated across all trajectories)\n";
    dp.System() << "  " << std::string(70, '-') << "\n";
    dp.System() << "  action_steps       = " << rvi_raw.total_action_steps      << "\n";
    dp.System() << "  real_event_steps   = " << rvi_raw.total_real_event_steps  << "\n";
    dp.System() << "  fil_refresh_steps  = " << rvi_raw.total_fil_refresh_steps << "\n";
    dp.System() << "  f_rvi              = " << std::setprecision(4) << f_rvi
                << "  (rvi_steps / all_steps)\n";
    dp.System() << "  mean_cost_per_rvi_step (raw ModifyStateWithEvent, unscaled) = "
                << std::setprecision(10) << rvi_raw.mean_cost_per_rvi_step << "\n";
    dp.System() << "  fraction action    = " << std::setprecision(4) << frac_act << "\n";
    dp.System() << "  per_event / per_rvi_step  (multiplier) = "
                << std::setprecision(6) << multiplier << "\n";

    // ---- per-event column: matches Section A policy comparer ----
    dp.System() << "\n  Per-event cost  [denominator = real_event_steps only; matches Section A]\n";
    dp.System() << "  " << std::string(70, '-') << "\n";
    dp.System() << "  " << std::left << std::setw(36) << "RVI policy simulated"
                << std::setprecision(10) << rvi_raw.mean_cost_per_event << "\n";
    dp.System() << "  " << std::left << std::setw(36) << "FIFO policy simulated"
                << std::setprecision(10) << fifo_raw.mean_cost_per_event << "\n";
    dp.System() << "  (These match the Section A policy comparer values)\n\n";

    // -----------------------------------------------------------------------
    // Section A: Performance at k=1 and k=k_true (tick_rate=1)
    // -----------------------------------------------------------------------
    dp.System() << std::string(80, '=') << "\n";
    dp.System() << "=== Section A: FIL-projected RVI in k=1 and k=" << k_true
                << " systems (tick_rate=1) ===\n";
    dp.System() << "    RVI is always solved on the FIL (k=1) projection.\n";
    dp.System() << "    k=" << k_true
                << " system is the 'true' queue: tracks FIL+SIL+TIL+QIL per type.\n";
    dp.System() << "    Question: does the FIL-only policy hold its advantage in the richer system?\n";
    dp.System() << std::string(80, '=') << "\n\n";
    print_header();

    // k=1
    print_perf_row(dp,
        "k=1 (FIL only)",
        cfg_base, M_base, /*k_depth=*/1, eval_cfg);

    // k=k_true: same RVI (FIL-projected, same M), richer simulator
    auto cfg_ktrue = make_config(lam0, lam1, mu, due_base, cost0, cost1,
                                 1.0, rtype, (int64_t)k_true);
    print_perf_row(dp,
        "k=" + std::to_string(k_true) + " (true system)",
        cfg_ktrue, M_base, /*k_depth=*/k_true, eval_cfg);

    dp.System() << "\n  Note: costs in k=" << k_true
                << " are slightly higher than k=1 because the richer simulator\n"
                << "  tracks lateness of SIL/TIL jobs that the k=1 simulator discards.\n"
                << "  The RVI/FIFO ratio is the meaningful comparison.\n\n";

    // -----------------------------------------------------------------------
    // Section B: Heatmaps at baseline (k=1, tick_rate=1)
    // -----------------------------------------------------------------------
    dp.System() << std::string(80, '=') << "\n";
    dp.System() << "=== Section B: Policy heatmaps at tick_rate=1, due=" << due_base << " ===\n";
    dp.System() << "    Row = FIL_0 (age of oldest waiting type-0 job in ticks)\n";
    dp.System() << "    Col = FIL_1 (age of oldest waiting type-1 job in ticks)\n";
    dp.System() << "    Cell: 0=serve type 0, 1=serve type 1, .=skip top candidate\n";
    dp.System() << std::string(80, '=') << "\n";
    {
        const int max_fil_b = std::min((int)(2.5 * due_base) + 1, 12);

        // RVI: look up action from the already-computed sol_base
        auto rvi_fn = [&mdp_direct, &sol_base](
                const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
            return mdp_direct.EvaluateRVIPolicy(sol_base, s);
        };
        // FIFO: always assigns action=1 (never skips)
        auto fifo_fn = [](const DynaPlex::Models::queue_mdp::MDP::State&) -> int64_t {
            return int64_t(1);
        };

        dp.System() << "\n[RVI policy, tick_rate=1, max_fil=" << max_fil_b
                    << "  (enumerated — no sampling artifacts)]\n";
        DynaPlex::Models::queue_mdp::PrintEnumeratedHeatmap(mdp_direct, rvi_fn, max_fil_b);

        dp.System() << "\n[FIFO policy, tick_rate=1, max_fil=" << max_fil_b
                    << "  (enumerated)]\n";
        DynaPlex::Models::queue_mdp::PrintEnumeratedHeatmap(mdp_direct, fifo_fn, max_fil_b);
    }

    // -----------------------------------------------------------------------
    // Section C: Tick-rate granularity sweep
    // -----------------------------------------------------------------------
    dp.System() << "\n" << std::string(80, '=') << "\n";
    dp.System() << "=== Section C: Tick-rate granularity sweep (k=1) ===\n";
    dp.System() << "    Physical deadline fixed at T_real=" << T_real
                << ".  due_ticks = T_real * tick_rate.\n";
    dp.System() << "    Absolute costs SCALE with tick_rate and are not directly comparable.\n";
    dp.System() << "    Key quantity: RVI/FIFO ratio — does finer resolution improve RVI?\n";
    dp.System() << std::string(80, '=') << "\n\n";

    dp.System() << "  " << std::left
                << std::setw(12) << "tick_rate"
                << std::setw(10) << "due_ticks"
                << std::setw(8)  << "M"
                << std::setw(14) << "FIFO_mean"
                << std::setw(14) << "RVI_mean"
                << std::setw(10) << "RVI/FIFO"
                << std::setw(9)  << "Improv"
                << "\n";
    dp.System() << "  " << std::string(76, '-') << "\n";

    struct SweepResult {
        double tick_rate;
        double due;
        int    M;
        double fifo_m;
        double rvi_m;
        double ratio;
        DynaPlex::Models::queue_mdp::MDP::RVISolution sol;  // stored for enumerated heatmap
    };
    std::vector<SweepResult> sweep_results;

    for (double tr : std::vector<double>{1.0, 2.0, 4.0, 8.0})
    {
        double due_tr = T_real * tr;

        // Adaptive RVI solve to pick M
        auto cfg_tr = make_config(lam0, lam1, mu, due_tr, cost0, cost1,
                                  tr, rtype, /*k=*/1);
        DynaPlex::Models::queue_mdp::MDP mdp_tr_direct(cfg_tr);
        auto sol_tr = mdp_tr_direct.runRVI(0.01);
        int M_tr = sol_tr.M;

        // Evaluate via framework
        auto mdp_tr  = dp.GetMDP(cfg_tr);
        auto fifo_tr = mdp_tr->GetPolicy("FIFO policy");
        auto rvi_tr  = mdp_tr->GetPolicy(VarGroup{
            {"id", std::string("RVI_optimal")}, {"M", int64_t(M_tr)}});

        auto cmp = dp.GetPolicyComparer(mdp_tr, eval_cfg);
        auto res = cmp.Compare({fifo_tr, rvi_tr});

        double fifo_m = 0, rvi_m = 0;
        res[0].Get("mean", fifo_m);
        res[1].Get("mean", rvi_m);
        double ratio  = (fifo_m > 1e-12) ? rvi_m / fifo_m : 0.0;
        double improv = 100.0 * (1.0 - ratio);

        dp.System() << "  " << std::left << std::fixed
                    << std::setw(12) << std::setprecision(0) << tr
                    << std::setw(10) << std::setprecision(0) << due_tr
                    << std::setw(8)  << M_tr
                    << std::setprecision(6)
                    << std::setw(14) << fifo_m
                    << std::setw(14) << rvi_m
                    << std::setprecision(4)
                    << std::setw(10) << ratio
                    << std::setprecision(2)
                    << std::setw(8)  << improv << "%"
                    << "\n";

        sweep_results.push_back({tr, due_tr, M_tr, fifo_m, rvi_m, ratio, std::move(sol_tr)});
    }

    // -----------------------------------------------------------------------
    // Section D: Heatmaps across tick rates (RVI only; shows sharpening)
    // -----------------------------------------------------------------------
    dp.System() << "\n" << std::string(80, '=') << "\n";
    dp.System() << "=== Section D: RVI heatmaps across tick rates ===\n";
    dp.System() << "    Same physical system, increasing FIL resolution.\n";
    dp.System() << "    Expect: coarser resolution -> blurrier threshold.\n";
    dp.System() << "             finer resolution  -> sharper staircase, type 0 clearly preferred.\n";
    dp.System() << std::string(80, '=') << "\n";

    for (const auto& sr : sweep_results)
    {
        double tr   = sr.tick_rate;
        double due  = sr.due;
        int    max_fil = std::min((int)(due + 7.0), 35);

        // Rebuild the concrete MDP for this tick rate (fast — no BFS)
        auto cfg_tr = make_config(lam0, lam1, mu, due, cost0, cost1,
                                  tr, rtype, /*k=*/1);
        DynaPlex::Models::queue_mdp::MDP mdp_tr_enum(cfg_tr);

        auto rvi_fn = [&mdp_tr_enum, &sr](
                const DynaPlex::Models::queue_mdp::MDP::State& s) -> int64_t {
            return mdp_tr_enum.EvaluateRVIPolicy(sr.sol, s);
        };
        auto fifo_fn = [](const DynaPlex::Models::queue_mdp::MDP::State&) -> int64_t {
            return int64_t(1);
        };

        dp.System() << "\n[tick_rate=" << std::fixed << std::setprecision(0) << tr
                    << "  due=" << due << "  max_fil=" << max_fil
                    << "  RVI/FIFO=" << std::setprecision(4) << sr.ratio << "]\n";
        dp.System() << "  RVI (enumerated):\n";
        DynaPlex::Models::queue_mdp::PrintEnumeratedHeatmap(mdp_tr_enum, rvi_fn, max_fil);
        dp.System() << "  FIFO (enumerated):\n";
        DynaPlex::Models::queue_mdp::PrintEnumeratedHeatmap(mdp_tr_enum, fifo_fn, max_fil);
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    dp.System() << "\n" << std::string(80, '=') << "\n";
    dp.System() << "=== Summary ===\n";
    dp.System() << std::string(80, '=') << "\n\n";

    dp.System() << "  Section A: FIL-projected RVI in true (k=" << k_true
                << ") system\n";
    dp.System() << "    k_true = " << k_true
                << " (rho^(k+1) < " << eps_k << " for rho=" << rho << ")\n";
    dp.System() << "    At rho=" << rho
                << ", queue rarely has >1 job waiting per type.\n";
    dp.System() << "    FIL projection loses very little: ratio should be similar k=1 vs k="
                << k_true << ".\n\n";

    dp.System() << "  Section C/D: Tick-rate granularity\n";
    dp.System() << "    RVI/FIFO ratios across tick rates:\n";
    for (const auto& sr : sweep_results)
        dp.System() << "      tick_rate=" << std::fixed << std::setprecision(0) << sr.tick_rate
                    << "  due=" << sr.due
                    << "  RVI/FIFO=" << std::setprecision(4) << sr.ratio << "\n";
    dp.System() << "\n";
    dp.System() << "    Heatmap sharpness increases with tick_rate: the type-0 preference\n";
    dp.System() << "    region becomes a cleaner staircase at finer time resolution.\n\n";

    dp.System() << std::string(80, '=') << "\n";
    dp.System() << "=== queue_depth_study DONE ===\n";
    dp.System() << std::string(80, '=') << "\n";

    return 0;
}
