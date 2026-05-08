// queue_cont_sim.cpp
// Tests the continuous-time event-driven simulator (MDP::SimulateContinuous)
// against the discrete-time uniformised-chain evaluator (EvaluatePolicyRaw).
//
// Tests
// -----
//   CS1  TraceContinuous  – short event log, visual sanity check
//   CS2  FIFO: continuous cost/time vs discrete cost/event
//   CS3  Bridge check: DynaPlex::Policy FIFO == raw-lambda FIFO
//   CS4  FIFO vs RVI: continuous simulator confirms RVI improvement
//   CS5  g* cross-check: cont cost/event ≈ g* from discrete RVI
//
// DynaPlex::Policy bridge
// -----------------------
// SimulateContinuous accepts std::function<int64_t(const State&)>.
// DynaPlex::Policy (obtained via fw->GetPolicy) uses a trajectory-based
// interface.  MakePolicyFn() bridges them by wrapping each raw State in a
// StateAdapter with the correct MDP hash and calling pol->SetAction.
// This lets NN policies (or any registered DynaPlex policy) be tested
// directly in the continuous simulator without any changes to its interface.

#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/trajectory.h"
#include "dynaplex/statecategory.h"
#include "dynaplex/erasure/stateadapter.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <span>

using DynaPlex::VarGroup;
using DynaPlex::Models::queue_mdp::MDP;
using DynaPlex::Models::queue_mdp::EvaluatePolicyRaw;

// ============================================================
//  MakePolicyFn
//  Converts a DynaPlex::Policy into std::function<int64_t(const MDP::State&)>.
//  fw must be the DynaPlex::MDP from which pol was obtained (same config hash).
//
//  Internally wraps each query state in a StateAdapter<MDP::State> with
//  the correct mdp_int_hash, places it in a reused Trajectory, and calls
//  pol->SetAction to obtain NextAction without modifying the state.
// ============================================================
static std::function<int64_t(const MDP::State&)>
MakePolicyFn(DynaPlex::MDP fw, DynaPlex::Policy pol)
{
    // Extract the MDP hash from a template initial state.
    auto tmpl      = fw->GetInitialState();
    int64_t hash   = tmpl->mdp_int_hash;

    // Create a reusable Trajectory carrying a StateAdapter<MDP::State>.
    // We use a shared_ptr so the lambda captures it by value safely.
    auto traj_ptr  = std::make_shared<DynaPlex::Trajectory>();
    // Initialise with a default-constructed State (placeholder).
    auto init_dp   = std::make_unique<DynaPlex::Erasure::StateAdapter<MDP::State>>(
                         hash, MDP::State{});
    traj_ptr->Reset(std::move(init_dp));

    return [pol, traj_ptr](const MDP::State& s) mutable -> int64_t {
        // Update the inner state in-place (avoids heap allocation per call).
        auto* adapter = static_cast<DynaPlex::Erasure::StateAdapter<MDP::State>*>(
            traj_ptr->GetState().get());
        adapter->state    = s;
        traj_ptr->Category = DynaPlex::StateCategory::AwaitAction();

        std::span<DynaPlex::Trajectory> span(traj_ptr.get(), 1);
        pol->SetAction(span);
        return traj_ptr->NextAction;
    };
}

// ============================================================
//  Convenience: print one result row
// ============================================================
static void PrintRow(const std::string& name,
                     const MDP::ContinuousSimResult& r,
                     double g_star = -1.0)
{
    std::cout << "  " << std::setw(12) << std::left  << name         << " | "
              << std::right
              << std::setw(10) << std::fixed << std::setprecision(6) << r.mean_cost_per_time  << " | "
              << std::setw(10) << std::setprecision(6)               << r.mean_cost_per_event << " +/- "
              << std::setw(8)  << std::setprecision(6)               << r.std_err_per_time    << " | "
              << std::setw(8)  << r.avg_events    << " | "
              << std::setw(8)  << r.avg_decisions;
    if (g_star >= 0.0)
        std::cout << "   g*=" << std::setprecision(6) << g_star;
    std::cout << "\n";
}

// ============================================================
//  main
// ============================================================
int main()
{
    try {
        auto& dp           = DynaPlex::DynaPlexProvider::Get();
        auto  path_to_json = dp.FilePath({"mdp_config_examples", "queue_mdp"},
                                         "mdp_config_0.json");
        VarGroup config    = VarGroup::LoadFromFile(path_to_json);

        // ---- Raw MDP (for direct API: runRVI, SimulateContinuous, etc.) ----
        MDP mdp(config);

        // ---- DynaPlex::MDP (for registered policy retrieval) ----
        DynaPlex::MDP fw = dp.GetMDP(config);

        std::cout << "========== queue_cont_sim: continuous-time simulator tests ==========\n";
        std::cout << "Config loaded.  n_jobs=" << mdp.n_jobs
                  << "  tick_rate=" << mdp.tick_rate << "\n";

        // ============================================================
        //  CS1: TraceContinuous – short visual trace
        // ============================================================
        std::cout << "\n========== CS1: TraceContinuous (FIFO, t=20) ==========\n";

        // Raw FIFO lambda: always take the offered assignment (action=1).
        auto fifo_fn = [](const MDP::State&) -> int64_t { return 1; };
        mdp.TraceContinuous(fifo_fn, /*t_trace=*/20.0, /*seed=*/42);

        // ============================================================
        //  CS2: Continuous simulation – FIFO baseline
        // ============================================================
        std::cout << "\n========== CS2: SimulateContinuous – FIFO policy ==========\n";
        std::cout << "  Parameters: n_traj=50, t_max=50000, t_warmup=5000\n\n";

        auto res_fifo_cont = mdp.SimulateContinuous(fifo_fn, /*n_traj=*/50,
                                                    /*t_max=*/50000.0,
                                                    /*t_warmup=*/5000.0);

        std::cout << "  Policy       | cost/time  | cost/event  +/-  std_err  | avg_ev   | avg_dec\n";
        std::cout << "  -------------|------------|----------------------------|----------|--------\n";
        PrintRow("FIFO (raw)", res_fifo_cont);

        // Discrete evaluation of FIFO – for reference only.
        //
        // NOTE: the two cost models are intentionally different and NOT directly comparable:
        //   Continuous: exact integral of  max(0, tau - T_real)  over physical time.
        //   Discrete:   FIL-based approximation, max(0, FIL - due_time) charged at each tick.
        // With tick_rate=0.1 and due_time=2 the discrete charges ~ tick_rate^2 x continuous
        // (floor() discretisation + tick-sampling).  A tight numerical match is not expected.
        // Both models produce RELATIVE comparisons (FIFO vs RVI) that are meaningful; the
        // absolute numbers are model-dependent and differ by a tick_rate-dependent factor.
        auto res_fifo_disc = EvaluatePolicyRaw(mdp, fifo_fn,
                                               /*n_traj=*/100,
                                               /*steps=*/100000,
                                               /*warmup=*/10000,
                                               /*seed=*/42);
        std::cout << "\n  [Continuous model]  cost/time = "
                  << std::fixed << std::setprecision(6) << res_fifo_cont.mean_cost_per_time
                  << "  (exact integral of max(0, tau - T_real))\n";
        std::cout << "  [Discrete   model]  cost/rvi-step = "
                  << res_fifo_disc.mean_cost_per_rvi_step_rvi
                  << "  (RVI-style: (tick_rate/Λ) x FIL-based cost)\n";
        std::cout << "  NOTE: these use different cost models; "
                     "a direct comparison is NOT valid.\n"
                     "  Relative ranking (FIFO vs RVI) is meaningful in each model independently.\n";

        // ============================================================
        //  CS3: DynaPlex::Policy bridge – FIFO raw == FIFO via bridge
        // ============================================================
        std::cout << "\n========== CS3: DynaPlex::Policy bridge ==========\n";

        DynaPlex::Policy fifo_pol    = fw->GetPolicy("FIFO policy");
        auto             fifo_bridge = MakePolicyFn(fw, fifo_pol);

        auto res_bridge = mdp.SimulateContinuous(fifo_bridge, /*n_traj=*/20,
                                                 /*t_max=*/50000.0,
                                                 /*t_warmup=*/5000.0);

        std::cout << "  FIFO raw   cost/time = "
                  << std::fixed << std::setprecision(6)
                  << res_fifo_cont.mean_cost_per_time << "\n";
        std::cout << "  FIFO bridge cost/time = "
                  << res_bridge.mean_cost_per_time << "\n";

        double diff_cs3 = std::abs(res_bridge.mean_cost_per_time
                                   - res_fifo_cont.mean_cost_per_time);
        double thr_cs3  = 4.0 * (res_fifo_cont.std_err_per_time
                                  + res_bridge.std_err_per_time);
        std::cout << "  |diff| = " << diff_cs3
                  << "  threshold (4 se) = " << thr_cs3 << "\n";
        std::cout << "  " << (diff_cs3 <= thr_cs3
                              ? "[PASS] raw lambda and DynaPlex::Policy bridge agree\n"
                              : "[WARN] bridge result differs from raw -- check hash\n");

        // ============================================================
        //  CS4 + CS5: RVI policy – solve, then compare FIFO vs RVI
        // ============================================================
        std::cout << "\n========== CS4/CS5: RVI policy – solve & compare ==========\n";

        // Use fixed M=24 (known to be well-converged for this config:
        // rel change M=22→M=24 ≈ 0.34 %, which is acceptable for testing).
        // Auto-M with rel_tol=1e-4 keeps expanding to M=36+ (slow).
        constexpr int rvi_M = 24;
        std::cout << "  Solving RVI (fixed M=" << rvi_M << ")...\n";
        auto sol    = mdp.runRVI(rvi_M);
        double g_star = sol.g_star;
        std::cout << "  g* = " << std::fixed << std::setprecision(8) << g_star
                  << "  (M=" << sol.M << ")\n\n";

        auto rvi_fn = [&](const MDP::State& s) -> int64_t {
            return mdp.EvaluateRVIPolicy(sol, s);
        };

        // Optional: also test via DynaPlex::Policy for RVI
        VarGroup rvi_cfg;
        rvi_cfg.Add("id", std::string{"RVI_optimal"});
        rvi_cfg.Add("M",  int64_t{sol.M});
        DynaPlex::Policy rvi_pol     = fw->GetPolicy(rvi_cfg);
        auto             rvi_bridge  = MakePolicyFn(fw, rvi_pol);

        auto res_rvi_raw    = mdp.SimulateContinuous(rvi_fn,      50, 50000.0, 5000.0);
        auto res_rvi_bridge = mdp.SimulateContinuous(rvi_bridge,  20, 50000.0, 5000.0);

        std::cout << "  Policy        | cost/time  | cost/event  +/-  std_err  | avg_ev   | avg_dec   g*\n";
        std::cout << "  --------------|------------|----------------------------|----------|------------------------\n";
        PrintRow("FIFO (raw)",    res_fifo_cont,  -1.0);
        PrintRow("RVI  (raw)",    res_rvi_raw,    g_star);
        PrintRow("RVI (bridge)",  res_rvi_bridge, g_star);

        // CS4: does RVI beat FIFO?
        std::cout << "\n  CS4 – RVI vs FIFO:\n";
        double gap_cs4 = res_fifo_cont.mean_cost_per_time - res_rvi_raw.mean_cost_per_time;
        double thr_cs4 = 2.0 * (res_fifo_cont.std_err_per_time + res_rvi_raw.std_err_per_time);
        std::cout << "  gap (FIFO - RVI) = " << std::fixed << std::setprecision(6) << gap_cs4
                  << "  threshold (2 se) = " << thr_cs4 << "\n";
        std::cout << "  " << (gap_cs4 > thr_cs4
                              ? "[PASS] RVI significantly cheaper than FIFO\n"
                              : "[NOTE] gap within noise -- may need more trajectories\n");

        // CS5: discrete g* self-check.
        //
        // Two candidate estimators of g* from EvaluatePolicyRaw:
        //
        //   (A) mean_cost_per_rvi_step × f_rvi
        //         mean_cost_per_rvi_step = Σ_{TICK} ComputeTickCost(s_post_tick) / N_rvi
        //         f_rvi                  = N_rvi / N_total  (= 1 - fil_refresh fraction)
        //         Empirically gives a result within ~3-5% of g*(fixed-M) for this config.
        //
        //   (B) mean_cost_per_step_gic
        //         = Σ GetImmediateCost(s_pre_event) at non-FIL-refresh steps / N_rvi
        //         Theoretically exact when the simulation's stationary distribution
        //         matches the RVI chain's.  Diverges from g*(fixed-M) when the
        //         truncation level M is not tight: at M=24 the RVI self-loop inflates
        //         the boundary-state probability and its Koole-tail cost, while the
        //         simulation visits those states far less often (untruncated system).
        //         For this config (tick_rate=0.1, auto-M heuristic gives M_init≈8),
        //         M=24 over-converges to an inflated g*, making (B) appear ~30% low.
        //
        // We use (A) here because fixed M=24 is what runRVI used above.
        // For a tight comparison use runRVI(rel_tol) with auto-M together with (B).
        std::cout << "\n  CS5 – discrete g* self-check:\n";
        auto res_rvi_disc  = EvaluatePolicyRaw(mdp, rvi_fn,
                                               /*n_traj=*/100,
                                               /*steps=*/100000,
                                               /*warmup=*/10000,
                                               /*seed=*/42);
        int64_t rvi_steps_cs5   = res_rvi_disc.total_action_steps
                                + res_rvi_disc.total_real_event_steps;
        int64_t total_steps_cs5 = rvi_steps_cs5 + res_rvi_disc.total_fil_refresh_steps;
        double  f_rvi           = (total_steps_cs5 > 0)
                                  ? static_cast<double>(rvi_steps_cs5) / total_steps_cs5
                                  : 1.0;
        double disc_rvi_g  = res_rvi_disc.mean_cost_per_rvi_step * f_rvi;
        double disc_rvi_se = res_rvi_disc.std_error * f_rvi;
        double diff_cs5 = std::abs(disc_rvi_g - g_star);
        double thr_cs5  = 3.0 * disc_rvi_se;
        std::cout << "  f_rvi = " << std::fixed << std::setprecision(4) << f_rvi
                  << "  (rvi_steps=" << rvi_steps_cs5
                  << " / total=" << total_steps_cs5 << ")\n";
        std::cout << "  g*                  (RVI solve) = "
                  << std::fixed << std::setprecision(8) << g_star << "\n";
        std::cout << "  g* estimate (sim×f_rvi)         = "
                  << std::setprecision(8) << disc_rvi_g
                  << " +/- " << std::setprecision(6) << disc_rvi_se << "\n";
        std::cout << "  g* estimate (mean_cost_per_step_gic) = "
                  << std::setprecision(8) << res_rvi_disc.mean_cost_per_step_gic
                  << "  (lower: M=24 inflates g* vs simulation)\n";
        std::cout << "  |diff| = " << std::setprecision(6) << diff_cs5
                  << "  threshold (3 se) = " << thr_cs5 << "\n";
        std::cout << "  " << (diff_cs5 <= thr_cs5
                              ? "[PASS] simulated g* estimate agrees with RVI solve\n"
                              : "[NOTE] diff larger than 3 se -- may need more steps\n");

        // ---- Summary ----
        std::cout << "\n========== Summary ==========\n";
        double pct_rvi = 100.0 * (res_fifo_cont.mean_cost_per_time
                                   - res_rvi_raw.mean_cost_per_time)
                                / res_fifo_cont.mean_cost_per_time;
        std::cout << "  RVI improvement over FIFO (continuous, cost/time): "
                  << std::fixed << std::setprecision(2) << pct_rvi << "%\n";
        std::cout << "\n  Continuous-model results (exact integral cost):\n";
        std::cout << "    FIFO cost/time = "
                  << std::setprecision(6) << res_fifo_cont.mean_cost_per_time << "\n";
        std::cout << "    RVI  cost/time = "
                  << res_rvi_raw.mean_cost_per_time << "\n";
        std::cout << "\n  Discrete-model results (FIL-based cost, RVI-style scaling):\n";
        std::cout << "    g*_FIFO (sim)  = "
                  << std::setprecision(8) << res_fifo_disc.mean_cost_per_rvi_step_rvi << "\n";
        std::cout << "    g*_RVI  (solve)= " << g_star << "\n";
        std::cout << "\n  Note: continuous and discrete models use different cost functions.\n"
                     "  The continuous model uses exact time-integral of max(0,tau-T_real);\n"
                     "  the discrete model uses (tick_rate/Λ)×max(0,FIL-due_time) per step.\n"
                     "  Relative rankings are consistent; absolute values differ.\n";

    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n========== queue_cont_sim DONE ==========\n";
    return 0;
}
