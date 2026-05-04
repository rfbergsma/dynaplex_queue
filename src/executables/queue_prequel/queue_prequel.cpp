#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include "dynaplex/dynaplexprovider.h"

using namespace DynaPlex;

// ---------------------------------------------------------------------------
// Build a 1-server, 2-job MDP config programmatically.
//
// depth (max_queue_depth):
//   1 = FIL only   (first-in-line job age per type)
//   2 = FIL + SIL  (also track second-in-line age)
//
// fd (feature_queue_depth):
//   Controls how many queue-depth slots the NN feature vector includes.
//   -1 = default (same as depth).
//   Setting fd=1 on a depth=2 MDP gives the NN FIL-only information even
//   though the simulator tracks SIL -- this is the control condition P2a.
// ---------------------------------------------------------------------------
static VarGroup make_1s2j(
    double  lam0, double lam1,
    double  mu,
    double  due0, double due1,
    double  cost0, double cost1,
    int64_t depth       = 1,
    int64_t fd          = -1,   // -1 = use depth
    int64_t reward_type = 1)    // 0 = binary, 1 = queue-lateness (QL)
{
    VarGroup srv;
    srv.Add("servers",      int64_t(1));
    srv.Add("can_serve",    VarGroup::Int64Vec{0, 1});
    srv.Add("service_rate", mu);

    VarGroup cfg;
    cfg.Add("id",              std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers",       int64_t(1));
    cfg.Add("n_jobs",          int64_t(2));
    cfg.Add("tick_rate",       1.0);
    cfg.Add("arrival_rates",   VarGroup::DoubleVec{lam0, lam1});
    cfg.Add("cost_rates",      VarGroup::DoubleVec{cost0, cost1});
    cfg.Add("due_times",       VarGroup::DoubleVec{due0, due1});
    cfg.Add("reward_type",     reward_type);
    cfg.Add("max_queue_depth", depth);
    if (fd > 0 && fd != depth)
        cfg.Add("feature_queue_depth", fd);
    cfg.Add("server_type_0",   srv);
    return cfg;
}

int main()
{
    auto& dp = DynaPlexProvider::Get();

    // -----------------------------------------------------------------------
    // System parameters
    //   1 server, 2 job types, asymmetric costs (type 1 costs 3x more)
    //   rho = (lam0 + lam1) / mu = 0.30 / 0.40 = 0.75
    //
    // Asymmetric costs make RVI non-trivially different from FIFO, so P1
    // is a genuine convergence test (not just "both always-assign").
    // Queue-lateness reward (type 1) rewards forward-looking behaviour
    // and is more sensitive to SIL information than binary reward.
    // -----------------------------------------------------------------------
    const double  lam0        = 0.15;
    const double  lam1        = 0.15;
    const double  mu          = 0.4;
    const double  due         = 3.0;
    const double  cost0       = 1.0;
    const double  cost1       = 3.0;   // type 1 costs 3x more
    const int64_t reward_type = 1;     // queue-lateness

    dp.System() << "\n========== queue_prequel: FIL/SIL representation study ==========\n";
    dp.System() << "System : 1 server, 2 job types\n";
    dp.System() << "  lambda = [" << lam0 << ", " << lam1 << "]"
                << "  mu = " << mu
                << "  due = [" << due << ", " << due << "]\n";
    dp.System() << "  cost_rates = [" << cost0 << ", " << cost1 << "]"
                << "  reward = queue-lateness"
                << "  rho = " << std::setprecision(4) << (lam0 + lam1) / mu << "\n\n";

    // -----------------------------------------------------------------------
    // Shared DCL hyperparameters
    //   N = 20000, H = 100, M = 400, num_gens = 3
    //   (from queue_analysis1 sweep: best combination for simple 2x2 configs)
    // -----------------------------------------------------------------------
    VarGroup nn_arch{
        {"type",         std::string("mlp")},
        {"hidden_layers",VarGroup::Int64Vec{64, 32}}
    };
    VarGroup nn_train{
        {"early_stopping_patience", int64_t(3)}
    };
    VarGroup dcl_cfg{
        {"N",               int64_t(20000)},
        {"num_gens",        int64_t(3)},
        {"M",               int64_t(400)},
        {"H",               int64_t(100)},
        {"nn_architecture", nn_arch},
        {"nn_training",     nn_train}
    };

    // Evaluation settings (200 trajectories x 100 000 steps each)
    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", int64_t(200));
    eval_cfg.Add("periods_per_trajectory", int64_t(100000));

    // RVI policy config -- always solved at depth=1 (feature_queue_depth=1
    // tells the RVI_optimal policy to build a FIL-only internal MDP for the
    // BFS/value-iteration even when the outer MDP tracks SIL).
    VarGroup rvi_cfg{
        {"id",                  std::string("RVI_optimal")},
        {"M",                   int64_t(60)},
        {"feature_queue_depth", int64_t(1)}
    };

    // Accumulators for summary table
    double fifo_P1 = 0, rvi_P1 = 0, nn_P1 = 0;
    double fifo_P2a = 0, rvi_P2a = 0, nn_P2a = 0;
    double fifo_P2b = 0, rvi_P2b = 0, nn_P2b = 0;
    bool pass_P1 = false, pass_P2a = false, pass_P2b = false;

    // =====================================================================
    // Section P1 : depth = 1  (FIL representation only)
    //
    // Both RVI and NN see only FIL.  RVI is optimal within this information
    // structure.  The NN should converge to the same cost as RVI.
    // FIFO is the baseline.
    //
    // Expected outcome:
    //   FIFO  > RVI  ≈ NN     [PASS if |NN - RVI| <= 3 * se_combined]
    // =====================================================================
    dp.System() << "========== P1: depth=1 (FIL representation only) ==========\n";

    {
        auto cfg  = make_1s2j(lam0, lam1, mu, due, due, cost0, cost1,
                              /*depth=*/1, /*fd=*/-1, reward_type);
        auto mdp  = dp.GetMDP(cfg);
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto rvi  = mdp->GetPolicy(rvi_cfg);

        // Train NN starting from FIFO seed policy
        auto dcl  = dp.GetDCL(mdp, fifo, dcl_cfg);
        dcl.TrainPolicy();
        auto nn   = dcl.GetPolicy();

        auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);
        auto res      = comparer.Compare({fifo, rvi, nn});

        double fifo_m, rvi_m, nn_m;
        double fifo_e, rvi_e, nn_e;
        res[0].Get("mean",  fifo_m); res[0].Get("error", fifo_e);
        res[1].Get("mean",  rvi_m);  res[1].Get("error", rvi_e);
        res[2].Get("mean",  nn_m);   res[2].Get("error", nn_e);

        dp.System() << std::fixed << std::setprecision(6);
        dp.System() << "  Policy                  | Mean cost      | Std error\n";
        dp.System() << "  FIFO  (depth=1)         | " << fifo_m << "  | " << fifo_e << "\n";
        dp.System() << "  RVI   (depth=1, FIL)    | " << rvi_m  << "  | " << rvi_e  << "\n";
        dp.System() << "  NN    (depth=1, FIL)    | " << nn_m   << "  | " << nn_e   << "\n";

        double rvi_vs_fifo_pct = 100.0 * (fifo_m - rvi_m) / fifo_m;
        double gap              = std::abs(nn_m - rvi_m);
        double thr              = 3.0 * (nn_e + rvi_e);
        bool   pass             = gap <= thr;

        dp.System() << "\n";
        dp.System() << "  RVI improvement over FIFO : "
                    << std::setprecision(2) << rvi_vs_fifo_pct << "%\n";
        dp.System() << "  |NN - RVI| = " << std::setprecision(6) << gap
                    << "  threshold (3*se) = " << thr << "\n";
        dp.System() << (pass
            ? "  [PASS] NN converges to RVI optimum within FIL representation\n"
            : "  [NOTE] NN has not fully converged to RVI -- consider more generations\n");
        dp.System() << "\n";

        fifo_P1 = fifo_m; rvi_P1 = rvi_m; nn_P1 = nn_m;
        pass_P1 = pass;
    }

    // =====================================================================
    // Section P2a : depth=2 sim, FIL-only NN  (control condition)
    //
    // The simulator now tracks SIL (second-in-line age) in the state, but
    // the NN feature extractor is restricted to feature_queue_depth=1 so
    // the NN receives only FIL information -- identical to P1.
    //
    // RVI is still solved at depth=1 (same policy as P1).
    //
    // Expected outcome:
    //   FIFO  > RVI  ≈ NN(fd=1)    [PASS if |NN - RVI| <= 3 * se_combined]
    //
    // This confirms the control: restricting features to FIL prevents the
    // NN from doing better than the FIL-only RVI benchmark.
    // =====================================================================
    dp.System() << "========== P2a: depth=2, FIL-only NN (control) ==========\n";

    {
        // max_queue_depth=2 (simulator tracks SIL), feature_queue_depth=1 (NN sees FIL only)
        auto cfg  = make_1s2j(lam0, lam1, mu, due, due, cost0, cost1,
                              /*depth=*/2, /*fd=*/1, reward_type);
        auto mdp  = dp.GetMDP(cfg);
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto rvi  = mdp->GetPolicy(rvi_cfg);  // depth=1 RVI registered on depth=2 MDP

        auto dcl  = dp.GetDCL(mdp, fifo, dcl_cfg);
        dcl.TrainPolicy();
        auto nn   = dcl.GetPolicy();

        auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);
        auto res      = comparer.Compare({fifo, rvi, nn});

        double fifo_m, rvi_m, nn_m;
        double fifo_e, rvi_e, nn_e;
        res[0].Get("mean",  fifo_m); res[0].Get("error", fifo_e);
        res[1].Get("mean",  rvi_m);  res[1].Get("error", rvi_e);
        res[2].Get("mean",  nn_m);   res[2].Get("error", nn_e);

        dp.System() << std::fixed << std::setprecision(6);
        dp.System() << "  Policy                          | Mean cost      | Std error\n";
        dp.System() << "  FIFO  (depth=2)                 | " << fifo_m << "  | " << fifo_e << "\n";
        dp.System() << "  RVI   (depth=1 solve, FIL only) | " << rvi_m  << "  | " << rvi_e  << "\n";
        dp.System() << "  NN    (depth=2, fd=1, FIL only) | " << nn_m   << "  | " << nn_e   << "\n";

        double gap  = std::abs(nn_m - rvi_m);
        double thr  = 3.0 * (nn_e + rvi_e);
        bool   pass = gap <= thr;

        dp.System() << "\n";
        dp.System() << "  [Control] |NN(fd=1) - RVI| = " << std::setprecision(6) << gap
                    << "  threshold (3*se) = " << thr << "\n";
        dp.System() << (pass
            ? "  [PASS] FIL-only NN matches RVI: SIL invisible, same benchmark\n"
            : "  [NOTE] FIL-only NN deviates from RVI (unexpected in control)\n");
        dp.System() << "\n";

        fifo_P2a = fifo_m; rvi_P2a = rvi_m; nn_P2a = nn_m;
        pass_P2a = pass;
    }

    // =====================================================================
    // Section P2b : depth=2 sim, FIL+SIL NN  (main result)
    //
    // Same depth=2 simulator as P2a, but now feature_queue_depth=2 so the
    // NN receives both FIL and SIL for each job type.  The NN can exploit
    // knowledge of the second waiting job to plan decisions more effectively.
    //
    // RVI is still the depth=1 FIL-only policy (the best policy available
    // without SIL information).
    //
    // Expected outcome:
    //   FIFO  > RVI  > NN(fd=2)    [PASS if RVI - NN > 2 * se_combined]
    //
    // This is the key finding: additional queue-depth information (SIL)
    // allows the NN to outperform the exact FIL-only RVI optimum.
    // =====================================================================
    dp.System() << "========== P2b: depth=2, FIL+SIL NN (main result) ==========\n";

    {
        // max_queue_depth=2, feature_queue_depth=2 (NN sees FIL and SIL)
        auto cfg  = make_1s2j(lam0, lam1, mu, due, due, cost0, cost1,
                              /*depth=*/2, /*fd=*/2, reward_type);
        auto mdp  = dp.GetMDP(cfg);
        auto fifo = mdp->GetPolicy("FIFO policy");
        auto rvi  = mdp->GetPolicy(rvi_cfg);  // depth=1 RVI registered on depth=2 MDP

        // Start DCL from the FIL-only RVI policy so gen-0 data is high quality;
        // subsequent generations can then explore SIL-aware improvements.
        auto dcl  = dp.GetDCL(mdp, rvi, dcl_cfg);
        dcl.TrainPolicy();
        auto nn   = dcl.GetPolicy();

        auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);
        auto res      = comparer.Compare({fifo, rvi, nn});

        double fifo_m, rvi_m, nn_m;
        double fifo_e, rvi_e, nn_e;
        res[0].Get("mean",  fifo_m); res[0].Get("error", fifo_e);
        res[1].Get("mean",  rvi_m);  res[1].Get("error", rvi_e);
        res[2].Get("mean",  nn_m);   res[2].Get("error", nn_e);

        dp.System() << std::fixed << std::setprecision(6);
        dp.System() << "  Policy                            | Mean cost      | Std error\n";
        dp.System() << "  FIFO  (depth=2)                   | " << fifo_m << "  | " << fifo_e << "\n";
        dp.System() << "  RVI   (depth=1 solve, FIL only)   | " << rvi_m  << "  | " << rvi_e  << "\n";
        dp.System() << "  NN    (depth=2, fd=2, FIL+SIL)    | " << nn_m   << "  | " << nn_e   << "\n";

        double rvi_vs_fifo_pct = 100.0 * (fifo_m - rvi_m) / fifo_m;
        double nn_vs_rvi_pct   = 100.0 * (rvi_m  - nn_m)  / rvi_m;
        double gap              = rvi_m - nn_m;
        double thr              = 2.0 * (rvi_e + nn_e);
        bool   pass             = gap > thr;

        dp.System() << "\n";
        dp.System() << "  RVI improvement over FIFO : "
                    << std::setprecision(2) << rvi_vs_fifo_pct << "%\n";
        dp.System() << "  NN  improvement over RVI  : "
                    << std::setprecision(2) << nn_vs_rvi_pct << "%\n";
        dp.System() << "  gap (RVI - NN) = " << std::setprecision(6) << gap
                    << "  threshold (2*se) = " << thr << "\n";
        dp.System() << (pass
            ? "  [PASS] NN (FIL+SIL) significantly outperforms RVI (FIL-only)\n"
            : "  [NOTE] NN does not significantly beat RVI -- SIL may not add value here\n");
        dp.System() << "\n";

        fifo_P2b = fifo_m; rvi_P2b = rvi_m; nn_P2b = nn_m;
        pass_P2b = pass;
    }

    // =====================================================================
    // Summary table
    // =====================================================================
    dp.System() << "========== Summary ==========\n\n";

    auto safe_pct = [](double num, double den) -> double {
        return (den > 0.0) ? 100.0 * num / den : 0.0;
    };

    dp.System() << std::left
                << std::setw(8)  << "Section"
                << std::setw(8)  << "Depth"
                << std::setw(6)  << "FD"
                << std::setw(14) << "FIFO"
                << std::setw(14) << "RVI(d=1)"
                << std::setw(14) << "NN"
                << std::setw(14) << "RVI/FIFO(%)"
                << std::setw(14) << "NN/RVI(%)"
                << "\n";
    dp.System() << std::string(92, '-') << "\n";

    auto print_row = [&](const std::string& sec, int depth, int fd,
                         double fifo, double rvi, double nn)
    {
        dp.System() << std::left << std::fixed << std::setprecision(5)
                    << std::setw(8)  << sec
                    << std::setw(8)  << depth
                    << std::setw(6)  << fd
                    << std::setw(14) << fifo
                    << std::setw(14) << rvi
                    << std::setw(14) << nn
                    << std::setw(14) << std::setprecision(2) << safe_pct(fifo - rvi, fifo)
                    << std::setw(14) << std::setprecision(2) << safe_pct(rvi  - nn,  rvi)
                    << "\n";
    };

    print_row("P1",  1, 1, fifo_P1,  rvi_P1,  nn_P1);
    print_row("P2a", 2, 1, fifo_P2a, rvi_P2a, nn_P2a);
    print_row("P2b", 2, 2, fifo_P2b, rvi_P2b, nn_P2b);

    dp.System() << "\n";
    dp.System() << "  P1  : NN ≈ RVI within FIL representation          "
                << (pass_P1  ? "[PASS]" : "[NOTE]") << "\n";
    dp.System() << "  P2a : NN(fd=1) ≈ RVI, blind to SIL (control)      "
                << (pass_P2a ? "[PASS]" : "[NOTE]") << "\n";
    dp.System() << "  P2b : NN(fd=2) > RVI using FIL+SIL features        "
                << (pass_P2b ? "[PASS]" : "[NOTE]") << "\n";
    dp.System() << "\n";
    dp.System() << "========== queue_prequel DONE ==========\n";

    return 0;
}
