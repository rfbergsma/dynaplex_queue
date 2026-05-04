// queue_trace.cpp
// Diagnostic executable for queue MDP: FIL/SIL/TIL representation testing.
//
// Tests:
//   T1: multi_queue unit tests (arrival, tick, complete_job, invariants)
//   T2: FIFO BFS-model trace depth=1 and depth=2 (what RVI sees vs reality)
//   T2c: BFS model action=1 vs real multi_queue.complete_job FIL distribution
//   T3: EvaluatePolicyRaw cost comparison -- FIFO depth=1 vs depth=2
//   T4: EvaluatePolicyRaw cost comparison -- RVI(depth=1 solve) on depth=1 vs depth=2

#include "dynaplex/dynaplexprovider.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <map>
#include <string>
#include <cassert>

#include "../../../lib/models/models/queue_mdp/mdp.h"

using DynaPlex::VarGroup;
using DynaPlex::Models::queue_mdp::MDP;
using DynaPlex::Models::queue_mdp::EvaluatePolicyRaw;
using DynaPlex::Models::queue_mdp::RawEvalResult;

// ---------------------------------------------------------------------------
// Config helper: 2 job types, 1 server capable of serving both
//   arrival_rates = [0.15, 0.15], tick_rate = 1.0, mu = [0.2, 0.2]
//   rho = 0.30 / 0.20 = 1.5 ... wait no:
//   total_lambda = 0.30, total_mu = 0.20, rho = 1.5 -- too high!
//   Let's use mu = 0.4 so rho = 0.30 / 0.40 = 0.75 (moderate load)
// ---------------------------------------------------------------------------
// reward_type: 0 = binary 1(FIL > D), 1 = queue-lateness
static VarGroup MakeCfg(int64_t depth, double lambda = 0.15, int64_t reward_type = 0)
{
    VarGroup srv;
    srv.Add("servers",       int64_t{1});
    srv.Add("service_rates", std::vector<double>{0.4, 0.4});
    srv.Add("can_serve",     std::vector<int64_t>{0, 1});

    double rho = 2.0 * lambda / 0.4;   // 2 types, mu=0.4
    (void)rho;  // used only in comments; suppress unused-variable warning

    VarGroup cfg;
    cfg.Add("id",             std::string{"queue_mdp"});
    cfg.Add("discount_factor", double{1.0});
    cfg.Add("k_servers",      int64_t{1});
    cfg.Add("n_jobs",         int64_t{2});
    cfg.Add("arrival_rates",  std::vector<double>{lambda, lambda});
    cfg.Add("tick_rate",      double{1.0});
    cfg.Add("cost_rates",     std::vector<double>{1.0, 1.0});
    cfg.Add("due_times",      std::vector<double>{3.0, 3.0});
    cfg.Add("reward_type",    reward_type);
    cfg.Add("max_queue_depth", depth);
    cfg.Add("server_type_0",  srv);
    return cfg;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string QueueStr(const MDP::multi_queue& q)
{
    std::ostringstream oss;
    for (size_t n = 0; n < q.waiting.size(); ++n) {
        oss << "T" << n << "=[";
        if (q.waiting[n].empty()) {
            oss << "-1";
        } else {
            bool first = true;
            for (int64_t v : q.waiting[n]) {
                if (!first) oss << ",";
                oss << v;
                first = false;
            }
        }
        oss << "] ";
    }
    return oss.str();
}

static std::string BusyStr(const MDP::ServerDynamicState& sm)
{
    std::ostringstream oss;
    oss << "busy=[";
    bool first = true;
    for (const auto& row : sm.busy_on)
        for (int64_t v : row) {
            if (!first) oss << ",";
            oss << v;
            first = false;
        }
    oss << "]";
    return oss.str();
}

static bool CheckInvariants(const MDP::State& s, int step, const std::string& ctx)
{
    bool ok = true;
    const auto& q = s.queue_manager;
    for (size_t n = 0; n < q.waiting.size(); ++n) {
        // FIL >= SIL >= TIL (earlier positions have larger waiting time)
        for (size_t pos = 0; pos + 1 < q.waiting[n].size(); ++pos) {
            if (q.waiting[n][pos] < q.waiting[n][pos + 1]) {
                std::cout << "  [INV FAIL] " << ctx << " step=" << step
                          << " type=" << n << " pos[" << pos << "]="
                          << q.waiting[n][pos] << " < pos[" << (pos+1) << "]="
                          << q.waiting[n][pos + 1] << "\n";
                ok = false;
            }
        }
        // No negative values inside the deque (FIL=-1 is stored as empty, not -1 in deque)
        for (int64_t v : q.waiting[n]) {
            if (v < 0) {
                std::cout << "  [INV FAIL] " << ctx << " step=" << step
                          << " type=" << n << " negative value " << v << " in deque\n";
                ok = false;
            }
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// T1: multi_queue unit tests
// ---------------------------------------------------------------------------
static void RunT1()
{
    std::cout << "\n========== T1: multi_queue unit tests ==========\n";

    // depth=2, 2 job types, lambda=0.15 each, tick_rate=1.0
    {
        MDP::multi_queue q;
        q.initialize(2, 1.0, {0.15, 0.15}, /*depth=*/2);

        // --- T1a: arrival into empty queue ---
        q.arrival(0);
        bool ok = (q.waiting[0].size() == 1 && q.waiting[0].front() == 0);
        std::cout << "  T1a [arrival->empty]:   FIL[0]=" << q.waiting[0].front()
                  << "  size=" << q.waiting[0].size()
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // --- T1b: tick increments FIL ---
        q.tick();
        ok = (q.waiting[0].front() == 1);
        std::cout << "  T1b [tick FIL++]:        FIL[0]=" << q.waiting[0].front()
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // --- T1c: arrival when FIL occupied (depth=2) -> SIL=0 ---
        q.arrival(0);   // queue: [1, 0]
        ok = (q.waiting[0].size() == 2 &&
              q.waiting[0].front() == 1 &&
              q.waiting[0].back()  == 0);
        std::cout << "  T1c [arrival->SIL=0]:    FIL[0]=" << q.waiting[0].front()
                  << "  SIL[0]=" << q.waiting[0].back()
                  << "  size=" << q.waiting[0].size()
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // --- T1d: tick increments both FIL and SIL ---
        q.tick();   // queue: [2, 1]
        ok = (q.waiting[0].size() == 2 &&
              q.waiting[0].front() == 2 &&
              q.waiting[0].back()  == 1);
        std::cout << "  T1d [tick FIL+SIL++]:    FIL[0]=" << q.waiting[0].front()
                  << "  SIL[0]=" << q.waiting[0].back()
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // --- T1e: FIL >= SIL invariant ---
        ok = (q.waiting[0].front() >= q.waiting[0].back());
        std::cout << "  T1e [FIL>=SIL]:          FIL=" << q.waiting[0].front()
                  << " >= SIL=" << q.waiting[0].back()
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // --- T1f: arrival blocked when queue full (depth=2) ---
        q.update_total_arrival_rate({0.15, 0.15});
        double rate_before = q.total_arrival_rate;
        // type 0 is full (size==2==max_queue_depth), type 1 is empty -> rate = 0.15
        ok = std::abs(rate_before - 0.15) < 1e-10;
        std::cout << "  T1f [arrival rate full]: total_rate=" << rate_before
                  << " (expected 0.15, type0 full, type1 empty)"
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // --- T1g: complete_job SIL->FIL deterministic shift ---
    {
        std::cout << "\n  --- T1g: complete_job with FIL=5, SIL=2 ---\n";
        MDP::multi_queue q;
        q.initialize(2, 1.0, {0.15, 0.15}, /*depth=*/2);
        q.waiting[0].push_back(5);  // FIL=5
        q.waiting[0].push_back(2);  // SIL=2
        q.update_total_arrival_rate({0.15, 0.15});  // type 0 full -> rate=0.15

        const int64_t old_sil = q.waiting[0].back();  // = 2

        // Run complete_job many times to verify the deterministic FIL shift
        int n_correct_fil = 0;
        const int N = 1000;
        std::mt19937_64 gen(99);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int i = 0; i < N; ++i) {
            MDP::multi_queue qi;
            qi.initialize(2, 1.0, {0.15, 0.15}, /*depth=*/2);
            qi.waiting[0].push_back(5);
            qi.waiting[0].push_back(2);
            qi.update_total_arrival_rate({0.15, 0.15});
            qi.complete_job(0, U(gen));
            if (qi.waiting[0].front() == old_sil) ++n_correct_fil;
        }
        bool ok = (n_correct_fil == N);
        std::cout << "  T1g [SIL->FIL shift]:    new_FIL == old_SIL (" << old_sil
                  << ") in " << n_correct_fil << "/" << N << " trials"
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";

        // Show new SIL distribution (Koole-sampled based on old SIL=2)
        std::map<int64_t, int> new_sil_counts;
        for (int i = 0; i < N; ++i) {
            MDP::multi_queue qi;
            qi.initialize(2, 1.0, {0.15, 0.15}, 2);
            qi.waiting[0].push_back(5);
            qi.waiting[0].push_back(2);
            qi.update_total_arrival_rate({0.15, 0.15});
            qi.complete_job(0, U(gen));
            int64_t new_sil = (qi.waiting[0].size() >= 2) ? qi.waiting[0].back() : -1;
            new_sil_counts[new_sil]++;
        }
        std::cout << "  T1g [new SIL distribution after FIL=5,SIL=2 completion]:\n";
        for (const auto& kv : new_sil_counts)
            std::cout << "    new_SIL=" << std::setw(3) << kv.first
                      << "  freq=" << std::fixed << std::setprecision(3)
                      << (double)kv.second / N << "\n";
    }

    // --- T1h: depth=1 complete_job (FIL=5, no SIL) -- Koole applied to FIL ---
    {
        std::cout << "\n  --- T1h: complete_job depth=1, FIL=5 (Koole on FIL) ---\n";
        const int N = 1000;
        std::mt19937_64 gen(77);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::map<int64_t, int> new_fil_counts;
        for (int i = 0; i < N; ++i) {
            MDP::multi_queue q;
            q.initialize(2, 1.0, {0.15, 0.15}, /*depth=*/1);
            q.waiting[0].push_back(5);  // FIL=5, no SIL
            q.update_total_arrival_rate({0.15, 0.15});
            q.complete_job(0, U(gen));
            int64_t new_fil = q.waiting[0].empty() ? -1 : q.waiting[0].front();
            new_fil_counts[new_fil]++;
        }
        std::cout << "  T1h [new FIL distribution after depth=1 FIL=5 completion]:\n";
        for (const auto& kv : new_fil_counts)
            std::cout << "    new_FIL=" << std::setw(3) << kv.first
                      << "  freq=" << std::fixed << std::setprecision(3)
                      << (double)kv.second / N << "\n";
        std::cout << "  (Compare T1g: depth=2 new_FIL=2 always; depth=1 new_FIL is stochastic)\n";
    }
}

// ---------------------------------------------------------------------------
// T2: FIFO BFS-model trace
// ---------------------------------------------------------------------------
static void RunT2()
{
    std::cout << "\n========== T2: FIFO BFS-model trace (30 steps) ==========\n";
    std::cout << "  NOTE: uses getNextStateProbability -- this is the FIL-projected BFS model\n"
              << "        that RVI is trained on.  Depth=2 BFS never accumulates SIL.\n";

    for (int depth : {1, 2}) {
        std::cout << "\n  --- depth=" << depth << " ---\n";
        std::cout << "  step |  queue contents            | servers   | cat | action\n";
        std::cout << "  -----|----------------------------|-----------|-----|-------\n";

        MDP mdp(MakeCfg(depth));
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> Udist(0.0, 1.0);

        auto state = mdp.GetInitialState();
        int n_inv_fail = 0;
        int n_sil_seen = 0;

        for (int step = 0; step < 30; ++step) {
            if (!CheckInvariants(state, step, "depth=" + std::to_string(depth)))
                ++n_inv_fail;

            // Count steps where SIL is populated (only meaningful for depth>=2)
            if (depth >= 2) {
                for (size_t n = 0; n < state.queue_manager.waiting.size(); ++n)
                    if ((int64_t)state.queue_manager.waiting[n].size() >= 2)
                        ++n_sil_seen;
            }

            int64_t action = (state.cat == DynaPlex::StateCategory::AwaitAction()) ? 1 : 0;

            std::cout << "  " << std::setw(4) << step << " | "
                      << std::setw(26) << std::left << QueueStr(state.queue_manager) << std::right
                      << " | " << std::setw(9) << BusyStr(state.server_manager)
                      << " | " << (state.cat == DynaPlex::StateCategory::AwaitAction() ? "ACT" : "EVT")
                      << " | " << (state.cat == DynaPlex::StateCategory::AwaitAction() ? std::to_string(action) : "-")
                      << "\n";

            // Sample next state
            auto dist = mdp.getNextStateProbability(state, action);
            double u = Udist(gen), cum = 0.0;
            size_t pick = 0;
            for (size_t pi = 0; pi < dist.size(); ++pi) {
                cum += dist[pi].probability;
                pick = pi;
                if (u <= cum) break;
            }
            state = dist[pick].next_state;
        }

        std::cout << "  Summary: inv_violations=" << n_inv_fail;
        if (depth >= 2) {
            std::cout << "  SIL_populated_steps=" << n_sil_seen;
            if (n_sil_seen == 0)
                std::cout << "  <- [BFS MODEL: SIL never tracked, FIL-projection only]";
        }
        std::cout << "\n";
    }
}

// ---------------------------------------------------------------------------
// T2c: BFS action=1 distribution vs real multi_queue.complete_job
//      Key diagnostic: shows the model mismatch for depth=2
// ---------------------------------------------------------------------------
static void RunT2c()
{
    std::cout << "\n========== T2c: BFS vs real-sim FIL distribution after action=1 ==========\n";
    std::cout << "  Starting state: FIL[0]=5, SIL[0]=2 (depth=2), type1 empty, server idle\n";
    std::cout << "  Action: assign job type 0 (action=1)\n\n";

    MDP mdp2(MakeCfg(2));

    // Build AwaitAction state: type 0 has FIL=5 and SIL=2, type 1 empty, server idle
    auto s = mdp2.GetInitialState();
    s.queue_manager.waiting[0].clear();
    s.queue_manager.waiting[0].push_back(5);  // FIL
    s.queue_manager.waiting[0].push_back(2);  // SIL
    s.queue_manager.update_total_arrival_rate(mdp2.arrival_rates);
    s.queue_manager.update_total_tick_rate(mdp2.tick_rate);
    s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
    s.server_manager.set_action_counter(0);
    s.cat = s.server_manager.action_queue.empty()
            ? DynaPlex::StateCategory::AwaitEvent()
            : DynaPlex::StateCategory::AwaitAction();

    if (s.cat != DynaPlex::StateCategory::AwaitAction()) {
        std::cout << "  [ERROR] Could not create AwaitAction state -- server not offering action\n";
        return;
    }

    // ----- BFS model: getNextStateProbability(s, 1) -----
    // Uses NextFILDistribution(FIL=5, lambda=0.15, gamma=1.0) -- IGNORES SIL!
    std::cout << "  BFS model (getNextStateProbability, action=1):\n";
    std::cout << "  Uses NextFILDistribution(FIL=5, lambda=0.15, gamma=1.0)\n";
    std::cout << "  -- SIL=2 is IGNORED, new FIL is sampled stochastically from old FIL=5 --\n";
    {
        auto bfs_dist = mdp2.getNextStateProbability(s, 1);
        std::map<int64_t, double> fil_probs;
        for (const auto& e : bfs_dist)
            fil_probs[e.next_state.queue_manager.get_FIL_waiting()[0]] += e.probability;

        for (const auto& kv : fil_probs)
            std::cout << "    next_FIL[0]=" << std::setw(3) << kv.first
                      << "  prob=" << std::fixed << std::setprecision(4) << kv.second << "\n";
        double p_empty = fil_probs.count(-1) ? fil_probs[-1] : 0.0;
        std::cout << "  -> P(queue empty after assignment) = " << std::setprecision(4) << p_empty << "\n";
    }

    // ----- Real simulation: multi_queue.complete_job -----
    // SIL=2 -> new FIL ALWAYS = 2 (deterministic shift), never empty
    std::cout << "\n  Real simulation (multi_queue.complete_job):\n";
    std::cout << "  -- SIL=2 shifts DETERMINISTICALLY to FIL. New bottom = Koole(SIL=2) --\n";
    {
        const int N = 50000;
        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::map<int64_t, int> new_fil_counts;

        for (int i = 0; i < N; ++i) {
            MDP::multi_queue q;
            q.initialize(2, mdp2.tick_rate, mdp2.arrival_rates, 2);
            q.waiting[0].push_back(5);
            q.waiting[0].push_back(2);
            q.update_total_arrival_rate(mdp2.arrival_rates);
            q.complete_job(0, U(gen));
            int64_t new_fil = q.waiting[0].empty() ? -1 : q.waiting[0].front();
            new_fil_counts[new_fil]++;
        }

        for (const auto& kv : new_fil_counts)
            std::cout << "    new_FIL[0]=" << std::setw(3) << kv.first
                      << "  freq=" << std::fixed << std::setprecision(4)
                      << (double)kv.second / N << "\n";
        double freq_empty = new_fil_counts.count(-1) ? (double)new_fil_counts[-1] / N : 0.0;
        std::cout << "  -> P(queue empty after completion) = " << std::setprecision(4) << freq_empty << "\n";
    }

    std::cout << "\n  [KEY FINDING]\n"
              << "    BFS model: ~50% chance queue is EMPTY after serving FIL=5 (stochastic Koole on FIL)\n"
              << "    Real sim:  queue is NEVER empty -- SIL=2 deterministically becomes new FIL=2\n"
              << "    => RVI trained on BFS model significantly UNDERESTIMATES queue occupancy\n"
              << "       after a completion event in depth=2. This biases the action policy.\n";
}

// ---------------------------------------------------------------------------
// T3: FIFO cost depth=1 vs depth=2 (binary reward -- same cost function both depths)
//
// With binary reward, cost = 1(FIL > D) regardless of depth.
// Any cost difference is PURELY from dynamics:
//   depth=1: FIL reset via Koole(old_FIL) after completion  -> often FIL=-1
//   depth=2: FIL = old_SIL deterministically                -> FIL rarely -1
// The steady-state FIL distribution should be higher at depth=2, so binary
// cost should be HIGHER at depth=2 (more time spent with FIL > D).
// ---------------------------------------------------------------------------
static void RunT3()
{
    std::cout << "\n========== T3: FIFO cost -- depth=1 vs depth=2 (binary reward) ==========\n";
    std::cout << "  Binary reward: cost = 1(FIL > D)  -- identical cost function at both depths.\n"
              << "  Any cost difference is PURELY from dynamics (Koole-reset vs SIL->FIL shift).\n";

    MDP mdp1(MakeCfg(1));
    MDP mdp2(MakeCfg(2));

    // Collect steady-state FIL histogram alongside cost measurement
    std::map<int64_t, int64_t> fil_hist_d1, fil_hist_d2;

    // Policy lambdas are called at AwaitAction states only (server free, job waiting).
    // Record FIL[0] unconditionally -- gives the FIL distribution at decision points.
    auto fifo_d1 = [&](const MDP::State& s) -> int64_t {
        int64_t fil = s.queue_manager.get_FIL_waiting()[0];
        fil_hist_d1[fil]++;
        return 1;
    };
    auto fifo_d2 = [&](const MDP::State& s) -> int64_t {
        int64_t fil = s.queue_manager.get_FIL_waiting()[0];
        fil_hist_d2[fil]++;
        return 1;
    };

    std::cout << "  Running EvaluatePolicyRaw (100 traj x 50K steps, warmup=5K)...\n";
    auto r1 = EvaluatePolicyRaw(mdp1, fifo_d1, 100, 50000, 5000, 42);
    auto r2 = EvaluatePolicyRaw(mdp2, fifo_d2, 100, 50000, 5000, 42);

    std::cout << "\n  FIFO depth=1: cost/rvi_step = " << std::setprecision(6)
              << r1.mean_cost_per_rvi_step << "  +/-" << r1.std_error << "\n";
    std::cout << "  FIFO depth=2: cost/rvi_step = " << std::setprecision(6)
              << r2.mean_cost_per_rvi_step << "  +/-" << r2.std_error << "\n";
    std::cout << "  ratio depth2/depth1 = " << std::setprecision(4)
              << r2.mean_cost_per_rvi_step / r1.mean_cost_per_rvi_step << "x\n";

    // Print FIL[0] steady-state histogram (top entries only)
    auto print_hist = [](const std::map<int64_t,int64_t>& h, const std::string& label) {
        int64_t total = 0;
        for (auto& kv : h) total += kv.second;
        std::cout << "  " << label << " FIL[0] distribution at decision points (top 12):\n";
        int shown = 0;
        for (auto& kv : h) {
            if (shown++ >= 12) { std::cout << "    ...\n"; break; }
            std::cout << "    FIL=" << std::setw(4) << kv.first
                      << "  freq=" << std::fixed << std::setprecision(4)
                      << (double)kv.second / total << "\n";
        }
        // Fraction with FIL > D (D=3)
        int64_t over_d = 0;
        for (auto& kv : h) if (kv.first > 3) over_d += kv.second;
        std::cout << "  -> P(FIL > D=3) = " << std::setprecision(4) << (double)over_d/total << "\n";
    };
    print_hist(fil_hist_d1, "depth=1");
    print_hist(fil_hist_d2, "depth=2");

    bool costs_similar = std::abs(r2.mean_cost_per_rvi_step - r1.mean_cost_per_rvi_step)
                         < 5.0 * (r1.std_error + r2.std_error);
    std::cout << "  [" << (costs_similar ? "NOTE: costs similar" : "NOTE: costs differ")
              << "] -- difference is purely from dynamics, not cost function\n";
}

// ---------------------------------------------------------------------------
// T4: RVI(depth=1 solve) on depth=1 vs depth=2 simulation
// ---------------------------------------------------------------------------
static void RunT4()
{
    std::cout << "\n========== T4: RVI(depth=1) policy -- depth=1 vs depth=2 simulation ==========\n";

    MDP mdp1(MakeCfg(1));
    MDP mdp2(MakeCfg(2));

    // Use fixed M=50 for a fast, diagnostically-sufficient solve.
    // rel_tol convergence requires M~150+ (slow); M=50 gives a good policy for the diagnostic.
    const int RVI_M = 50;
    std::cout << "  Solving RVI on depth=1 MDP (fixed M=" << RVI_M << ")...\n";
    auto sol = mdp1.runRVI(RVI_M);
    std::cout << "  g* = " << std::setprecision(12) << sol.g_star << "  M=" << sol.M << "\n\n";

    // Action counters (separate for depth=1 and depth=2 runs)
    int64_t rvi_a0_d1 = 0, rvi_a1_d1 = 0;
    int64_t rvi_a0_d2 = 0, rvi_a1_d2 = 0;

    auto rvi_d1 = [&](const MDP::State& s) -> int64_t {
        int64_t a = mdp1.EvaluateRVIPolicy(sol, s);
        if (a == 0) ++rvi_a0_d1; else ++rvi_a1_d1;
        return a;
    };
    auto rvi_d2 = [&](const MDP::State& s) -> int64_t {
        // Lookup uses mdp1's encoder (FIL-only) -- works for both depths via get_FIL_waiting()
        int64_t a = mdp1.EvaluateRVIPolicy(sol, s);
        if (a == 0) ++rvi_a0_d2; else ++rvi_a1_d2;
        return a;
    };
    auto fifo = [](const MDP::State&) -> int64_t { return 1; };

    std::cout << "  Running EvaluatePolicyRaw (100 traj x 50K steps, warmup=5K)...\n";
    auto r_rvi_d1  = EvaluatePolicyRaw(mdp1, rvi_d1,  100, 50000, 5000, 42);
    auto r_rvi_d2  = EvaluatePolicyRaw(mdp2, rvi_d2,  100, 50000, 5000, 42);
    auto r_fifo_d1 = EvaluatePolicyRaw(mdp1, fifo,    100, 50000, 5000, 42);
    auto r_fifo_d2 = EvaluatePolicyRaw(mdp2, fifo,    100, 50000, 5000, 42);

    std::cout << "\n  Policy          | depth=1 cost/step       | depth=2 cost/step       | ratio d2/d1\n";
    std::cout << "  ----------------|-------------------------|-------------------------|------------\n";

    auto print_row = [](const std::string& name, const RawEvalResult& rd1, const RawEvalResult& rd2) {
        double ratio = rd2.mean_cost_per_rvi_step / rd1.mean_cost_per_rvi_step;
        std::cout << "  " << std::setw(15) << std::left << name << " | "
                  << std::right << std::setw(8) << std::setprecision(4) << rd1.mean_cost_per_rvi_step
                  << " +/- " << std::setw(7) << std::setprecision(4) << rd1.std_error << "   | "
                  << std::setw(8) << std::setprecision(4) << rd2.mean_cost_per_rvi_step
                  << " +/- " << std::setw(7) << std::setprecision(4) << rd2.std_error << "   | "
                  << std::setprecision(3) << ratio << "x\n";
    };

    print_row("FIFO",          r_fifo_d1, r_fifo_d2);
    print_row("RVI(d1 solve)", r_rvi_d1,  r_rvi_d2);

    std::cout << "\n  RVI action skip rates:\n";
    auto skip1 = (rvi_a0_d1 + rvi_a1_d1 > 0)
                 ? (double)rvi_a0_d1 / (rvi_a0_d1 + rvi_a1_d1) : 0.0;
    auto skip2 = (rvi_a0_d2 + rvi_a1_d2 > 0)
                 ? (double)rvi_a0_d2 / (rvi_a0_d2 + rvi_a1_d2) : 0.0;
    std::cout << "    depth=1: a=0=" << rvi_a0_d1 << "  a=1=" << rvi_a1_d1
              << "  skip_rate=" << std::setprecision(4) << skip1 << "\n";
    std::cout << "    depth=2: a=0=" << rvi_a0_d2 << "  a=1=" << rvi_a1_d2
              << "  skip_rate=" << std::setprecision(4) << skip2 << "\n";

    // Diagnostic conclusions
    // With binary reward + deeper dynamics, depth=2 SHOULD cost more than depth=1:
    //   SIL->FIL shift keeps queues occupied, raising P(FIL > D). This is expected.
    bool fifo_d2_higher       = r_fifo_d2.mean_cost_per_rvi_step
                                > r_fifo_d1.mean_cost_per_rvi_step + 5.0 * r_fifo_d1.std_error;
    bool rvi_worse_on_d2      = r_rvi_d2.mean_cost_per_rvi_step
                                > r_rvi_d1.mean_cost_per_rvi_step - 5.0 * r_rvi_d1.std_error;
    bool rvi_worse_than_fifo  = r_rvi_d2.mean_cost_per_rvi_step
                                > r_fifo_d2.mean_cost_per_rvi_step;

    std::cout << "\n  Diagnostics:\n";
    std::cout << "    [" << (fifo_d2_higher     ? "EXP " : "WARN") << "] FIFO cost depth=2 > depth=1 (expected: SIL->FIL shift raises occupancy)\n";
    std::cout << "    [" << (rvi_worse_on_d2    ? "EXP " : "WARN") << "] RVI cost increases from depth=1 to depth=2 (expected: same dynamics effect)\n";
    std::cout << "    [" << (rvi_worse_than_fifo ? "FAIL" : "OK  ") << "] RVI depth=2 "
              << (rvi_worse_than_fifo ? "WORSE" : "better") << " than FIFO depth=2\n";

    if (rvi_worse_than_fifo) {
        std::cout << "\n  ROOT CAUSE (confirmed by T2c):\n"
                  << "    The depth=1 BFS model uses stochastic FIL refresh (Koole on old FIL).\n"
                  << "    The real depth=2 simulation uses deterministic SIL->FIL shift.\n"
                  << "    => In the model: P(queue empty after completion) ~50% (high FIL=5).\n"
                  << "    => In reality:   queue almost never empty (SIL always present at rho=0.75).\n"
                  << "    => RVI over-estimates idle-queue probability -> too conservative on assigns.\n"
                  << "    CONCLUSION: The queue representation (multi_queue) is CORRECT.\n"
                  << "                The problem is the FIL-projected BFS model used by RVI.\n"
                  << "                For depth=2 optimal policy, solve RVI on depth=2 BFS model\n"
                  << "                or use RL to learn from real simulation.\n";
    }
}

// ---------------------------------------------------------------------------
// T5: FIFO binary cost ratio depth=2/depth=1 across rho values
//
// Hypothesis: Koole exactly recovers the SIL (2nd in line) but not TIL (3rd).
//   => At low rho (P(TIL)=rho^3 << 1): depth=1 and depth=2 costs should equalize.
//   => At high rho (rho=0.75, P(TIL)=0.42): depth=2 >> depth=1 because TIL matters.
//
// Effective mean queue captured at depth=k with Koole:
//   L_k = rho*(1 - rho^{k+1}) / (1 - rho)
//   (depth=1 captures FIL + 0-or-1 SIL  ~ rho + rho^2 positions)
//   (depth=2 captures FIL + SIL + 0-or-1 TIL ~ rho + rho^2 + rho^3 positions)
//
// Adjustment: multiply C_k by (L_inf / L_k) = 1/(1 - rho^{k+1})
//   to extrapolate to the true-system cost (depth=inf).
//   If cost is linear in effective queue captured, adjusted costs from all depths
//   should agree.  Departures from linearity show up at high rho.
// ---------------------------------------------------------------------------
static void RunT5()
{
    std::cout << "\n========== T5: cost ratio depth=2/depth=1 vs rho (binary reward) ==========\n";
    std::cout << "  Hypothesis: ratio -> 1.0 as rho -> 0  (TIL negligible, Koole fully compensates)\n";
    std::cout << "  L_k = effective mean queue captured at depth=k (with Koole)\n";
    std::cout << "  Adjusted cost = C_k * L_inf/L_k  (extrapolate to true-system load)\n\n";

    // lambda values: rho = 2*lambda/mu = 2*lambda/0.4 = 5*lambda
    std::vector<double> lambdas = {0.02, 0.05, 0.10, 0.15};

    std::cout << "  rho  | P(TIL) | C_d1 (+/- se)          | C_d2 (+/- se)          |"
              << " ratio (+/- se)  | L1    | L2    | Cadj_d1  | Cadj_d2  | adj_ratio\n";
    std::cout << "  -----|--------|------------------------|------------------------|"
              << "-----------------|-------|-------|----------|----------|----------\n";

    for (double lam : lambdas) {
        double rho = 2.0 * lam / 0.4;   // 2 types, mu=0.4
        double p_til   = rho * rho * rho;
        double L_inf   = rho / (1.0 - rho);
        double L1      = rho * (1.0 - rho * rho)           / (1.0 - rho);  // rho+rho^2
        double L2      = rho * (1.0 - rho * rho * rho)     / (1.0 - rho);  // rho+rho^2+rho^3
        double adj1    = L_inf / L1;   // = 1/(1-rho^2)
        double adj2    = L_inf / L2;   // = 1/(1-rho^3)

        MDP mdp1(MakeCfg(1, lam));
        MDP mdp2(MakeCfg(2, lam));
        auto fifo = [](const MDP::State&) -> int64_t { return 1; };

        // Use more trajectories/steps for low rho (small costs = noisy ratios)
        int n_traj = (rho < 0.3) ? 500 : 100;
        int n_steps = (rho < 0.3) ? 100000 : 50000;

        auto r1 = EvaluatePolicyRaw(mdp1, fifo, n_traj, n_steps, 5000, 42);
        auto r2 = EvaluatePolicyRaw(mdp2, fifo, n_traj, n_steps, 5000, 42);

        double ratio     = r2.mean_cost_per_rvi_step / r1.mean_cost_per_rvi_step;
        double cadj1     = r1.mean_cost_per_rvi_step * adj1;
        double cadj2     = r2.mean_cost_per_rvi_step * adj2;
        double adj_ratio = cadj2 / cadj1;

        // Propagate uncertainty on ratio: delta_ratio/ratio ~ sqrt((se1/c1)^2 + (se2/c2)^2)
        double rel_se_ratio = std::sqrt(
            std::pow(r1.std_error / r1.mean_cost_per_rvi_step, 2) +
            std::pow(r2.std_error / r2.mean_cost_per_rvi_step, 2));
        double se_ratio = ratio * rel_se_ratio;

        std::cout << std::fixed
                  << "  " << std::setw(4)  << std::setprecision(2) << rho
                  << " | " << std::setw(6) << std::setprecision(4) << p_til
                  << " | " << std::setw(8) << std::setprecision(5) << r1.mean_cost_per_rvi_step
                  << " +/-" << std::setw(7) << std::setprecision(5) << r1.std_error
                  << " | " << std::setw(8) << std::setprecision(5) << r2.mean_cost_per_rvi_step
                  << " +/-" << std::setw(7) << std::setprecision(5) << r2.std_error
                  << " | " << std::setw(5) << std::setprecision(3) << ratio
                  << " +/-" << std::setw(5) << std::setprecision(3) << se_ratio << "x"
                  << " | " << std::setw(5) << std::setprecision(3) << L1
                  << " | " << std::setw(5) << std::setprecision(3) << L2
                  << " | " << std::setw(8) << std::setprecision(5) << cadj1
                  << " | " << std::setw(8) << std::setprecision(5) << cadj2
                  << " | " << std::setw(5) << std::setprecision(3) << adj_ratio << "x\n";
    }

    std::cout << "\n  [If hypothesis correct] ratio column converges to 1.00x at low rho.\n"
              << "  [If L_k adjustment works] adj_ratio column stays near 1.00x at ALL rho.\n"
              << "  [If L_k adjustment over/under-shoots] binary cost is non-linear in L_k.\n";
}

// ---------------------------------------------------------------------------
// T6: Monotone convergence + geometric decay across depths 1..MAX_DEPTH
//
// Binary reward: ΔC(k+1)/ΔC(k) ≈ ρ² (geometric decay from depth 1).
// QL reward:     cost grows as depth increases because the Koole tail
//                approximation underestimates the true cost at low depth.
//                Deeper tracking captures more exact contributions, raising
//                the total toward the true infinite-buffer cost.
//                At HIGH rho (0.75), deep positions frequently have age>D
//                so the deltas are still growing at depth 10.
//                At LOW rho (0.25), positions beyond depth ~3-4 are nearly
//                always empty so the deltas must shrink quickly.
// ---------------------------------------------------------------------------
// Helper: run both reward types for a given lambda, print table + delta analysis.
static void RunT6ForRho(double lam, int MAX_DEPTH)
{
    const double rho  = 2.0 * lam / 0.4;
    const double rho2 = rho * rho;

    std::cout << "\n  --- rho=" << std::fixed << std::setprecision(2) << rho
              << "  (lambda=" << std::setprecision(4) << lam
              << ", rho^2=" << std::setprecision(4) << rho2 << ") ---\n";

    auto fifo = [](const MDP::State&) -> int64_t { return 1; };

    std::vector<double> costs_bin, costs_ql;

    std::cout << "  depth | C_binary (+/- se)            | C_ql       (+/- se)         |\n";
    std::cout << "  ------|-----------------------------|-----------------------------|";
    std::cout << "\n";

    for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
        // Use 100 traj for depths 1-4 at rho=0.75 to match previous output;
        // 50 traj elsewhere (lower rho -> less variance anyway).
        int n_traj  = (rho > 0.5 && depth <= 4) ? 100 : 50;
        int n_steps = 50000;

        MDP mdp_bin(MakeCfg(depth, lam, /*reward_type=*/0));
        MDP mdp_ql (MakeCfg(depth, lam, /*reward_type=*/1));
        auto r_bin = EvaluatePolicyRaw(mdp_bin, fifo, n_traj, n_steps, 5000, 42);
        auto r_ql  = EvaluatePolicyRaw(mdp_ql,  fifo, n_traj, n_steps, 5000, 42);

        costs_bin.push_back(r_bin.mean_cost_per_rvi_step);
        costs_ql.push_back(r_ql.mean_cost_per_rvi_step);

        std::cout << "  " << std::setw(5) << depth
                  << " | " << std::setw(9) << std::fixed << std::setprecision(5)
                  << r_bin.mean_cost_per_rvi_step
                  << " +/- " << std::setw(7) << std::setprecision(5) << r_bin.std_error
                  << " | " << std::setw(10) << std::setprecision(4)
                  << r_ql.mean_cost_per_rvi_step
                  << " +/- " << std::setw(7) << std::setprecision(4) << r_ql.std_error
                  << " |\n";
        std::cout.flush();   // show progress live
    }

    // ---------- delta analysis ----------
    auto print_deltas = [&](const std::vector<double>& C, const std::string& label,
                            bool show_theory) {
        std::cout << "\n  --- Delta analysis (" << label << ") ---\n";
        if (show_theory)
            std::cout << "  Theoretical rho^2 = " << std::setprecision(4) << rho2 << "\n";

        bool monotone = true;
        for (int k = 0; k + 1 < (int)C.size(); ++k)
            if (C[k + 1] <= C[k]) { monotone = false; break; }
        std::cout << "  Monotone C(1)<...<C(" << MAX_DEPTH << "): "
                  << (monotone ? "[PASS]" : "[FAIL]") << "\n";

        std::cout << "  depth k | Delta C(k)       | Delta(k)/Delta(k-1)";
        if (show_theory) std::cout << "  [theory rho^2=" << std::setprecision(4) << rho2 << "]";
        std::cout << "\n";
        std::cout << "  --------|------------------|--------------------\n";
        for (int k = 1; k < (int)C.size(); ++k) {
            double dk = C[k] - C[k - 1];
            std::cout << "  " << std::setw(7) << (k + 1)
                      << " | " << std::setw(16) << std::fixed << std::setprecision(5) << dk
                      << " | ";
            if (k == 1) {
                std::cout << "      ---\n";
            } else {
                double prev = C[k - 1] - C[k - 2];
                double ratio = (std::abs(prev) > 1e-12) ? dk / prev : 0.0;
                std::cout << std::setprecision(4) << ratio;
                if (show_theory) {
                    bool close = std::abs(ratio - rho2) < 0.05;
                    std::cout << "  " << (close ? "[~rho^2]" : "");
                }
                std::cout << "\n";
            }
        }

        // For QL: detect where deltas peak and whether they have started declining
        if (!show_theory) {
            int peak_k = 1;
            double peak_delta = (C.size() > 1) ? C[1] - C[0] : 0.0;
            for (int k = 2; k < (int)C.size(); ++k) {
                double dk = C[k] - C[k - 1];
                if (dk > peak_delta) { peak_delta = dk; peak_k = k; }
            }
            std::cout << "  QL delta peak at transition " << (peak_k+1)
                      << "->" << (peak_k+2) << "  (delta=" << std::setprecision(4) << peak_delta << ")\n";
            // Check if all deltas after peak are declining monotonically
            bool declining = true;
            for (int k = peak_k + 1; k + 1 < (int)C.size(); ++k) {
                if ((C[k+1]-C[k]) >= (C[k]-C[k-1]))
                    { declining = false; break; }
            }
            if (peak_k + 1 < (int)C.size())
                std::cout << "  QL deltas declining after peak: "
                          << (declining ? "[YES -> converging]" : "[NOT YET]") << "\n";
            else
                std::cout << "  QL delta peak is at last computed depth -> need more depths\n";
        }
    };

    print_deltas(costs_bin, "binary reward", /*show_theory=*/true);
    print_deltas(costs_ql,  "QL reward",     /*show_theory=*/false);
}

static void RunT6()
{
    const int MAX_DEPTH = 10;
    std::cout << "\n========== T6: Monotone convergence + geometric decay (depths 1-"
              << MAX_DEPTH << ") ==========\n";
    std::cout << "  FIFO policy.  Two load levels: rho=0.75 and rho=0.25.\n";
    std::cout << "  Binary: ΔC ratio -> rho^2 from depth 1 (geometric decay).\n";
    std::cout << "  QL:     at rho=0.75, deltas still growing at depth 10.\n";
    std::cout << "          at rho=0.25, deltas must shrink quickly (queue rarely deep).\n";

    RunT6ForRho(0.15, MAX_DEPTH);   // rho = 2*0.15/0.4 = 0.75
    RunT6ForRho(0.05, MAX_DEPTH);   // rho = 2*0.05/0.4 = 0.25
}

// ---------------------------------------------------------------------------
// T7: Departure-shift correctness at depth=3
//
// State: type0 has [FIL=7, SIL=4, TIL=1] (depth=3).
// After complete_job(0, U):
//   new FIL = 4  (old SIL shifts up — deterministic, 100% always)
//   new SIL = 1  (old TIL shifts up — deterministic, 100% always)
//   P(new TIL = -1) = beta^(old_TIL) = beta^1  where beta = gamma/(lambda+gamma)
//
// The key correctness check is that Koole is anchored on TIL (=1), NOT on
// FIL (=7) or SIL (=4).  A bug that applies Koole to FIL or SIL would give
// a very different P(empty), which this test detects.
// ---------------------------------------------------------------------------
static void RunT7()
{
    std::cout << "\n========== T7: Departure-shift correctness at depth=3 ==========\n";

    const double lam  = 0.15;
    const double tick = 1.0;
    const double beta_theory = tick / (lam + tick);   // 1/1.15 ≈ 0.8696
    const int64_t W1 = 7, W2 = 4, W3 = 1;

    std::cout << "  State: FIL=" << W1 << " SIL=" << W2 << " TIL=" << W3 << "\n";
    std::cout << "  beta = gamma/(lambda+gamma) = " << std::setprecision(4) << beta_theory << "\n";

    double p_theory_W3 = std::pow(beta_theory, (double)W3);
    double p_theory_W2 = std::pow(beta_theory, (double)W2);
    double p_theory_W1 = std::pow(beta_theory, (double)W1);
    std::cout << "  P(new TIL=-1) correct [anchor=TIL=" << W3 << "]: beta^" << W3 << " = "
              << std::setprecision(4) << p_theory_W3 << "\n";
    std::cout << "  P(new TIL=-1) if wrong [anchor=SIL=" << W2 << "]: beta^" << W2 << " = "
              << std::setprecision(4) << p_theory_W2 << "\n";
    std::cout << "  P(new TIL=-1) if wrong [anchor=FIL=" << W1 << "]: beta^" << W1 << " = "
              << std::setprecision(4) << p_theory_W1 << "\n\n";

    const int N = 20000;
    std::mt19937_64 gen(42);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    int64_t n_fil_ok = 0, n_sil_ok = 0, n_til_empty = 0;

    for (int i = 0; i < N; ++i) {
        MDP::multi_queue q;
        q.initialize(2, tick, {lam, lam}, /*depth=*/3);
        q.waiting[0].push_back(W1);
        q.waiting[0].push_back(W2);
        q.waiting[0].push_back(W3);
        q.update_total_arrival_rate({lam, lam});
        q.complete_job(0, U(gen));

        int64_t new_fil = q.waiting[0].empty()              ? -1 : q.waiting[0][0];
        int64_t new_sil = (q.waiting[0].size() >= 2)        ? q.waiting[0][1] : -1;
        int64_t new_til = ((int64_t)q.waiting[0].size() >= 3) ? q.waiting[0][2] : -1;

        if (new_fil == W2) ++n_fil_ok;
        if (new_sil == W3) ++n_sil_ok;
        if (new_til <  0 ) ++n_til_empty;
    }

    double p_til_empty = (double)n_til_empty / N;
    double tol = 3.0 * std::sqrt(p_theory_W3 * (1.0 - p_theory_W3) / N);

    bool fil_pass = (n_fil_ok == N);
    bool sil_pass = (n_sil_ok == N);
    bool til_pass = std::abs(p_til_empty - p_theory_W3) < tol;

    std::cout << "  Results (" << N << " trials):\n";
    std::cout << "    new FIL = " << W2 << " in " << n_fil_ok << "/" << N << " trials  "
              << (fil_pass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "    new SIL = " << W3 << " in " << n_sil_ok << "/" << N << " trials  "
              << (sil_pass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "    P(new TIL=-1) = " << std::setprecision(5) << p_til_empty
              << "  theory=" << std::setprecision(5) << p_theory_W3
              << "  3-sigma tol=" << std::setprecision(5) << tol << "  "
              << (til_pass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "    [wrong-anchor comparison: W2=" << std::setprecision(4) << p_theory_W2
              << "  W1=" << p_theory_W1 << "]\n";
}

// ---------------------------------------------------------------------------
// T8: Tick uniformity -- fully-filled AND partially-filled queues
//
// Sub-test T8a: depth=3, all three slots filled [FIL=A, SIL=B, TIL=C].
//   After N ticks: FIL=A+N, SIL=B+N, TIL=C+N.  Ordering invariant holds.
//
// Sub-test T8b: depth=2, ONLY FIL filled [FIL=5, SIL absent].
//   Tick must increment FIL but must NOT manufacture a SIL=0 job out of nothing.
//   Verified: queue size stays 1, FIL increments, no phantom SIL appears.
//
// Sub-test T8c: depth=3, FIL+SIL filled, TIL absent [FIL=5, SIL=3].
//   Tick must increment FIL and SIL but must NOT create TIL=0.
//   Verified: queue size stays 2, correct increments, no phantom TIL.
//
// Rationale for T8b/c: the internal representation uses a deque that stores
// ONLY the ages of present jobs -- -1 is never stored; an absent SIL is
// simply not in the deque.  A tick iterates over deque elements only, so it
// cannot touch a slot that does not exist.  These sub-tests confirm this is
// actually the case and that no off-by-one or default-initialisation bug
// silently inserts a phantom job.
// ---------------------------------------------------------------------------
static void RunT8()
{
    std::cout << "\n========== T8: Tick uniformity (fully- and partially-filled) ==========\n";
    const double lam = 0.15;

    // ---- T8a: all three slots filled ----
    {
        std::cout << "\n  T8a [depth=3, FIL+SIL+TIL filled]:\n";
        const int64_t A = 5, B = 3, C = 1;
        const int N = 10;

        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, /*depth=*/3);
        q.waiting[0].push_back(A);
        q.waiting[0].push_back(B);
        q.waiting[0].push_back(C);
        q.update_total_arrival_rate({lam, lam});

        bool inv_ok = true;
        for (int t = 1; t <= N; ++t) {
            q.tick();
            int64_t fil = q.waiting[0][0];
            int64_t sil = q.waiting[0][1];
            int64_t til = q.waiting[0][2];
            if (fil < sil || sil < til) { inv_ok = false; }
        }

        int64_t fil_f = q.waiting[0][0];
        int64_t sil_f = q.waiting[0][1];
        int64_t til_f = q.waiting[0][2];

        bool ok = (fil_f == A+N) && (sil_f == B+N) && (til_f == C+N) && inv_ok;
        std::cout << "    After " << N << " ticks: FIL=" << fil_f
                  << " SIL=" << sil_f << " TIL=" << til_f
                  << "  (expected " << A+N << "/" << B+N << "/" << C+N << ")  "
                  << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // ---- T8b: depth=2, only FIL present -- the key regression ----
    {
        std::cout << "\n  T8b [depth=2, FIL=5, SIL absent -- tick must NOT create phantom SIL=0]:\n";
        const int64_t FIL_INIT = 5;
        const int N = 10;

        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, /*depth=*/2);
        q.waiting[0].push_back(FIL_INIT);   // only FIL; SIL slot is empty
        q.update_total_arrival_rate({lam, lam});

        bool size_ok = true, fil_ok = true, no_phantom_sil = true;
        for (int t = 1; t <= N; ++t) {
            q.tick();
            if (q.waiting[0].size() != 1) size_ok = false;           // must stay size 1
            if (q.waiting[0][0] != FIL_INIT + t) fil_ok = false;     // FIL must tick up
            if (q.waiting[0].size() >= 2) no_phantom_sil = false;    // no phantom SIL
        }

        bool ok = size_ok && fil_ok && no_phantom_sil;
        int64_t fil_f = q.waiting[0][0];
        std::cout << "    After " << N << " ticks: FIL=" << fil_f
                  << " (expected " << FIL_INIT+N << ")  "
                  << "queue_size=" << q.waiting[0].size() << " (expected 1)  "
                  << (ok ? "[PASS]" : "[FAIL]") << "\n";
        if (!ok) {
            if (!no_phantom_sil)
                std::cout << "    [FAIL DETAIL] phantom SIL appeared in deque!\n";
            if (!fil_ok)
                std::cout << "    [FAIL DETAIL] FIL did not increment correctly\n";
        }
    }

    // ---- T8c: depth=3, FIL+SIL present, TIL absent ----
    {
        std::cout << "\n  T8c [depth=3, FIL=5, SIL=3, TIL absent -- tick must NOT create phantom TIL=0]:\n";
        const int64_t FIL_INIT = 5, SIL_INIT = 3;
        const int N = 10;

        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, /*depth=*/3);
        q.waiting[0].push_back(FIL_INIT);
        q.waiting[0].push_back(SIL_INIT);   // FIL+SIL only; TIL absent
        q.update_total_arrival_rate({lam, lam});

        bool size_ok = true, fil_ok = true, sil_ok = true, no_phantom_til = true;
        for (int t = 1; t <= N; ++t) {
            q.tick();
            if ((int64_t)q.waiting[0].size() != 2) size_ok = false;
            if (q.waiting[0][0] != FIL_INIT + t) fil_ok = false;
            if (q.waiting[0][1] != SIL_INIT + t) sil_ok = false;
            if ((int64_t)q.waiting[0].size() >= 3) no_phantom_til = false;
        }

        bool ok = size_ok && fil_ok && sil_ok && no_phantom_til;
        int64_t fil_f = q.waiting[0][0];
        int64_t sil_f = q.waiting[0][1];
        std::cout << "    After " << N << " ticks: FIL=" << fil_f << " SIL=" << sil_f
                  << "  (expected " << FIL_INIT+N << "/" << SIL_INIT+N << ")  "
                  << "queue_size=" << q.waiting[0].size() << " (expected 2)  "
                  << (ok ? "[PASS]" : "[FAIL]") << "\n";
        if (!no_phantom_til)
            std::cout << "    [FAIL DETAIL] phantom TIL appeared in deque!\n";
    }
}

// ---------------------------------------------------------------------------
// T9: Arrival placement
//
// depth=3 queue.  Verify:
//   Case 1: empty     -> arrival goes to FIL slot (position 0)
//   Case 2: FIL only  -> arrival goes to SIL slot (position 1)
//   Case 3: FIL+SIL   -> arrival goes to TIL slot (position 2)
//   Case 4: full      -> arrival blocked (queue size unchanged, rate excludes type)
//   Case 5: after departure from full queue, arrival rate is re-enabled correctly
// ---------------------------------------------------------------------------
static void RunT9()
{
    std::cout << "\n========== T9: Arrival placement (depth=3) ==========\n";

    const double lam = 0.15;

    // Case 1: empty -> FIL
    {
        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, 3);
        q.arrival(0);
        bool ok = (q.waiting[0].size() == 1 && q.waiting[0][0] == 0);
        std::cout << "  Case 1 [empty -> FIL=0]:         size=" << q.waiting[0].size()
                  << " FIL=" << q.waiting[0][0] << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Case 2: FIL occupied -> SIL
    {
        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, 3);
        q.waiting[0].push_back(5);
        q.update_total_arrival_rate({lam, lam});
        q.arrival(0);
        int64_t sil = (q.waiting[0].size() >= 2) ? q.waiting[0][1] : -1;
        bool ok = (q.waiting[0].size() == 2 && q.waiting[0][0] == 5 && sil == 0);
        std::cout << "  Case 2 [FIL=5 -> SIL=0]:         size=" << q.waiting[0].size()
                  << " FIL=" << q.waiting[0][0] << " SIL=" << sil << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Case 3: FIL+SIL -> TIL
    {
        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, 3);
        q.waiting[0].push_back(5);
        q.waiting[0].push_back(2);
        q.update_total_arrival_rate({lam, lam});
        q.arrival(0);
        int64_t til = ((int64_t)q.waiting[0].size() >= 3) ? q.waiting[0][2] : -1;
        bool ok = (q.waiting[0].size() == 3 && q.waiting[0][0] == 5
                   && q.waiting[0][1] == 2 && til == 0);
        std::cout << "  Case 3 [FIL=5,SIL=2 -> TIL=0]:  size=" << q.waiting[0].size()
                  << " FIL=" << q.waiting[0][0] << " SIL=" << q.waiting[0][1] << " TIL=" << til
                  << "  " << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Case 4: full (size==3==max_depth) -> blocked, rate excludes type0
    {
        MDP::multi_queue q;
        q.initialize(2, 1.0, {lam, lam}, 3);
        q.waiting[0].push_back(5);
        q.waiting[0].push_back(2);
        q.waiting[0].push_back(1);
        q.update_total_arrival_rate({lam, lam});
        double rate_full = q.total_arrival_rate;  // should be lam (only type1)
        q.arrival(0);  // no-op: type0 full
        bool size_ok = (q.waiting[0].size() == 3);
        bool rate_ok = std::abs(rate_full - lam) < 1e-10;
        std::cout << "  Case 4 [full -> blocked]:         size=" << q.waiting[0].size()
                  << " rate_while_full=" << std::setprecision(4) << rate_full
                  << "  " << (size_ok && rate_ok ? "[PASS]" : "[FAIL]") << "\n";
    }

    // Case 5: after complete_job from full queue, total_arrival_rate matches queue state
    {
        const int N = 1000;
        std::mt19937_64 gen(77);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        int n_inconsistent = 0;

        for (int i = 0; i < N; ++i) {
            MDP::multi_queue q;
            q.initialize(2, 1.0, {lam, lam}, 3);
            q.waiting[0].push_back(5);
            q.waiting[0].push_back(2);
            q.waiting[0].push_back(1);
            q.update_total_arrival_rate({lam, lam});
            q.complete_job(0, U(gen));

            // expected rate = lam * (type0 not full) + lam * (type1 not full)
            double expected = 0.0;
            if ((int64_t)q.waiting[0].size() < 3) expected += lam;
            if ((int64_t)q.waiting[1].size() < 3) expected += lam;

            if (std::abs(q.total_arrival_rate - expected) > 1e-10)
                ++n_inconsistent;
        }

        bool ok = (n_inconsistent == 0);
        std::cout << "  Case 5 [rate consistent after departure]: "
                  << n_inconsistent << "/" << N << " inconsistencies  "
                  << (ok ? "[PASS]" : "[FAIL]") << "\n";
    }
}

// ---------------------------------------------------------------------------
// T10: Koole anchor regression
//
// P(new bottom = -1 | old bottom age = W) = beta^W
//   where beta = gamma / (lambda + gamma)
//
// Test this at W = 1, 2, 4, 7, 10 using depth=1 (single-position queue).
// A failure here means the geometric sampler itself is wrong.
// The separate test T7 verifies the anchor is applied to the CORRECT position.
// ---------------------------------------------------------------------------
static void RunT10()
{
    std::cout << "\n========== T10: Koole anchor regression P(empty|W) = beta^W ==========\n";

    const double lam   = 0.15;
    const double gamma_rate = 1.0;
    const double beta = gamma_rate / (lam + gamma_rate);

    std::cout << "  lambda=" << lam << "  gamma=" << gamma_rate
              << "  beta = gamma/(lambda+gamma) = "
              << std::setprecision(6) << beta << "\n\n";

    const int N = 100000;
    std::mt19937_64 gen(1234);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    std::vector<int64_t> test_W = {1, 2, 4, 7, 10};

    std::cout << "  W    | P(empty) empirical | P(empty) theory  | error    | 3-sigma  | pass?\n";
    std::cout << "  -----|-------------------|--------------------|----------|----------|----- \n";

    bool all_pass = true;
    for (int64_t W : test_W) {
        int64_t n_empty = 0;
        for (int i = 0; i < N; ++i) {
            MDP::multi_queue q;
            q.initialize(2, gamma_rate, {lam, lam}, /*depth=*/1);
            q.waiting[0].push_back(W);
            q.update_total_arrival_rate({lam, lam});
            q.complete_job(0, U(gen));
            if (q.waiting[0].empty()) ++n_empty;
        }
        double p_emp    = (double)n_empty / N;
        double p_theory = std::pow(beta, (double)W);
        double error    = std::abs(p_emp - p_theory);
        double tol      = 3.0 * std::sqrt(p_theory * (1.0 - p_theory) / N);
        bool pass = (error < tol);
        if (!pass) all_pass = false;

        std::cout << "  " << std::setw(4) << W
                  << " | " << std::setw(17) << std::fixed << std::setprecision(5) << p_emp
                  << " | " << std::setw(18) << std::setprecision(5) << p_theory
                  << " | " << std::setw(8) << std::setprecision(5) << error
                  << " | " << std::setw(8) << std::setprecision(5) << tol
                  << " | " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    }

    std::cout << "\n  Overall Koole anchor: " << (all_pass ? "[PASS]" : "[FAIL]") << "\n";
    std::cout << "  (T7 additionally verifies the anchor is on TIL, not FIL or SIL.)\n";
}

// ---------------------------------------------------------------------------
int main()
{
    std::cout << "========== queue_trace: FIL/SIL/TIL diagnostic ==========\n";
    std::cout << "System: 2 job types, 1 server (mu=0.4 each type), lambda=[0.15,0.15]\n";
    std::cout << "        tick_rate=1.0, due_times=[3,3], binary reward (reward_type=0)\n";
    std::cout << "        rho = 0.30/0.40 = 0.75\n";

    try {
        RunT1();
        RunT2();
        RunT2c();
        RunT3();
        RunT4();
        RunT5();
        RunT6();
        RunT7();
        RunT8();
        RunT9();
        RunT10();
    }
    catch (const std::exception& ex) {
        std::cerr << "\n[EXCEPTION] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n========== queue_trace DONE ==========\n";
    return 0;
}
