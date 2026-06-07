// action_label_test.cpp
//
// Verifies the is_fifo_winner / is_cmu_winner / is_rfq_winner labels produced by
// LabelActionQueue, plus the policy decisions of FIFOPolicy, CmuPolicy, and
// ReverseFIFOPolicy.
//
// Scenarios
//   S1  1 pool C=1, FIL=[8,4]           basic below-diagonal
//   S2  1 pool C=1, FIL=[4,8]           above-diagonal (type-1 older)
//   S3  1 pool C=1, only type-0 waiting  single candidate
//   S4  1 pool C=2, FIL=[8,4]           HARD: multi-server pool, both servers idle
//   S5  1 pool C=2, 1 busy + FIL=[*,4]  HARD: C=2 pool with 1 server already busy
//   S6  2 pools C=1 each, FIL=[8,4]     HARD: two independent pools
//   S7  1 pool C=2, 3 job types          HARD: C=2 with 3 candidates
//   S8  FIL tie FIL=[6,6] C=1           edge case: strict-inequality tie-breaking
//
// Exit code: 0 = all assertions pass, 1 = at least one FAIL.

#include <iostream>
#include <iomanip>
#include <deque>
#include <vector>
#include <string>
#include <cassert>

#include "../../../lib/models/models/queue_mdp/mdp.h"
#include "../../../lib/models/models/queue_mdp/policies.h"

using DynaPlex::VarGroup;
using DynaPlex::Models::queue_mdp::MDP;

// =========================================================================
// Helpers
// =========================================================================

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const std::string& msg)
{
    if (cond) {
        std::cout << "    [PASS] " << msg << "\n";
        ++g_pass;
    } else {
        std::cout << "    [FAIL] " << msg << "\n";
        ++g_fail;
    }
}

static void print_queue(const std::deque<MDP::Action>& q,
                        const std::vector<int64_t>&   FIL,
                        const std::vector<double>&     cost_rates,
                        const std::vector<MDP::ServerStaticInfo>& si)
{
    for (size_t i = 0; i < q.size(); ++i) {
        const auto& a = q[i];
        const int64_t fil = (a.job_type >= 0 && (size_t)a.job_type < FIL.size())
                          ? FIL[(size_t)a.job_type] : -1;
        double cmu = 0.0;
        {
            int idx = MDP::ServerDynamicState::canServeIndex(si, a.server_index, a.job_type);
            if (idx >= 0)
                cmu = si[(size_t)a.server_index].mu_kj[(size_t)idx]
                    * ((a.job_type >= 0 && (size_t)a.job_type < cost_rates.size())
                       ? cost_rates[(size_t)a.job_type] : 0.0);
        }
        std::cout << "    alpha=" << i
                  << "  pool=" << a.server_index << " job=" << a.job_type
                  << "  FIL=" << fil
                  << "  c*mu=" << std::fixed << std::setprecision(1) << cmu
                  << "  | fifo=" << a.is_fifo_winner
                  <<   " cmu="  << a.is_cmu_winner
                  <<   " rfq="  << a.is_rfq_winner
                  << "\n";
    }
}

// Build a state with:
//   - server pool k busy according to busy_counts[k] (counts for each can_serve slot)
//   - FIL values set via fil_values (index = job type, -1 = not waiting)
static MDP::State make_state(const MDP& mdp,
                             const std::vector<std::vector<int64_t>>& busy_counts,
                             const std::vector<int64_t>& fil_values)
{
    MDP::State s;
    s.queue_manager.initialize(
        mdp.n_jobs, mdp.tick_rate, mdp.arrival_rates, /*depth=*/1);
    for (int64_t n = 0; n < (int64_t)fil_values.size(); ++n)
        if (fil_values[(size_t)n] >= 0)
            s.queue_manager.set_fil(n, fil_values[(size_t)n]);

    s.server_manager.initialize(&mdp.server_static_info, mdp.n_jobs);
    for (size_t k = 0; k < busy_counts.size() && k < s.server_manager.busy_on.size(); ++k)
        for (size_t j = 0; j < busy_counts[k].size() && j < s.server_manager.busy_on[k].size(); ++j)
            s.server_manager.busy_on[k][j] = busy_counts[k][j];

    s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting(), mdp.cost_rates);
    s.server_manager.set_action_counter(0);
    s.cat = s.server_manager.action_queue.empty()
          ? DynaPlex::StateCategory::AwaitEvent()
          : DynaPlex::StateCategory::AwaitAction();
    return s;
}

// Walk the policy through all alpha steps and return the actions taken.
static std::vector<int64_t> trace_policy(
    const MDP& mdp, MDP::State s,
    const std::function<int64_t(const MDP::State&)>& get_action)
{
    std::vector<int64_t> actions;
    while (mdp.GetStateCategory(s).IsAwaitAction()) {
        int64_t a = get_action(s);
        actions.push_back(a);
        mdp.ModifyStateWithAction(s, a);
    }
    return actions;
}

// =========================================================================
// MDP config builders
// =========================================================================

static MDP make_1pool_1server(int64_t n_jobs,
                              const std::vector<double>& cost_rates,
                              double mu = 0.35)
{
    std::vector<double> mu_vec(n_jobs, mu);
    std::vector<int64_t> can_serve(n_jobs);
    for (int64_t j = 0; j < n_jobs; ++j) can_serve[(size_t)j] = j;

    VarGroup srv;
    srv.Add("servers",       int64_t{1});
    srv.Add("service_rates", mu_vec);
    srv.Add("can_serve",     can_serve);

    std::vector<double> arrivals(n_jobs, 0.25);
    std::vector<double> dues(n_jobs, 3.0);

    VarGroup cfg;
    cfg.Add("id",             std::string{"queue_mdp"});
    cfg.Add("discount_factor", 1.0);
    cfg.Add("reward_type",    int64_t{0});
    cfg.Add("k_servers",      int64_t{1});
    cfg.Add("n_jobs",         n_jobs);
    cfg.Add("tick_rate",      1.0);
    cfg.Add("arrival_rates",  arrivals);
    cfg.Add("cost_rates",     cost_rates);
    cfg.Add("due_times",      dues);
    cfg.Add("server_type_0",  srv);
    return MDP{cfg};
}

static MDP make_1pool_2server(const std::vector<double>& cost_rates)
{
    const int64_t n_jobs = (int64_t)cost_rates.size();
    std::vector<double>  mu_vec(n_jobs, 0.35);
    std::vector<int64_t> can_serve(n_jobs);
    for (int64_t j = 0; j < n_jobs; ++j) can_serve[(size_t)j] = j;

    VarGroup srv;
    srv.Add("servers",       int64_t{2});   // <-- 2-server pool
    srv.Add("service_rates", mu_vec);
    srv.Add("can_serve",     can_serve);

    std::vector<double> arrivals(n_jobs, 0.25);
    std::vector<double> dues(n_jobs, 3.0);

    VarGroup cfg;
    cfg.Add("id",             std::string{"queue_mdp"});
    cfg.Add("discount_factor", 1.0);
    cfg.Add("reward_type",    int64_t{0});
    cfg.Add("k_servers",      int64_t{1});
    cfg.Add("n_jobs",         n_jobs);
    cfg.Add("tick_rate",      1.0);
    cfg.Add("arrival_rates",  arrivals);
    cfg.Add("cost_rates",     cost_rates);
    cfg.Add("due_times",      dues);
    cfg.Add("server_type_0",  srv);
    return MDP{cfg};
}

static MDP make_2pool_1server_each(const std::vector<double>& cost_rates)
{
    const int64_t n_jobs = (int64_t)cost_rates.size();
    std::vector<double>  mu_vec(n_jobs, 0.35);
    std::vector<int64_t> can_serve(n_jobs);
    for (int64_t j = 0; j < n_jobs; ++j) can_serve[(size_t)j] = j;

    VarGroup srv;
    srv.Add("servers",       int64_t{1});
    srv.Add("service_rates", mu_vec);
    srv.Add("can_serve",     can_serve);

    std::vector<double> arrivals(n_jobs, 0.25);
    std::vector<double> dues(n_jobs, 3.0);

    VarGroup cfg;
    cfg.Add("id",             std::string{"queue_mdp"});
    cfg.Add("discount_factor", 1.0);
    cfg.Add("reward_type",    int64_t{0});
    cfg.Add("k_servers",      int64_t{2});   // <-- 2 pools
    cfg.Add("n_jobs",         n_jobs);
    cfg.Add("tick_rate",      1.0);
    cfg.Add("arrival_rates",  arrivals);
    cfg.Add("cost_rates",     cost_rates);
    cfg.Add("due_times",      dues);
    cfg.Add("server_type_0",  srv);
    cfg.Add("server_type_1",  srv);
    return MDP{cfg};
}

// =========================================================================
// Individual scenarios
// =========================================================================

static void run_S1()
{
    std::cout << "\n--- S1: 1 pool C=1, FIL=[8,4]  (below diagonal) ---\n";
    MDP mdp = make_1pool_1server(2, {100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0}}, {8, 4});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // alpha=0: (pool0, j0, FIL=8)  cmu=35  fifo=1  cmu=0  rfq=0
    CHECK(q.size() == 2,                        "queue size = 2");
    CHECK(q[0].job_type        == 0,            "alpha=0 candidate is j0");
    CHECK(q[0].is_fifo_winner  == 1,            "alpha=0 is_fifo_winner=1");
    CHECK(q[0].is_cmu_winner   == 0,            "alpha=0 is_cmu_winner=0  (j1 has higher c*mu)");
    CHECK(q[0].is_rfq_winner   == 0,            "alpha=0 is_rfq_winner=0  (j1 is newer)");
    // alpha=1: (pool0, j1, FIL=4)  cmu=105  fifo=0  cmu=1  rfq=1
    CHECK(q[1].job_type        == 1,            "alpha=1 candidate is j1");
    CHECK(q[1].is_fifo_winner  == 0,            "alpha=1 is_fifo_winner=0");
    CHECK(q[1].is_cmu_winner   == 1,            "alpha=1 is_cmu_winner=1");
    CHECK(q[1].is_rfq_winner   == 1,            "alpha=1 is_rfq_winner=1");

    // Policy trace
    auto fifo_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::FIFOPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(!fifo_acts.empty() && fifo_acts[0] == 1, "FIFOPolicy assigns at alpha=0");

    auto cmu_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::CmuPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(cmu_acts.size() >= 2 && cmu_acts[0] == 0 && cmu_acts[1] == 1,
          "CmuPolicy skips alpha=0, assigns alpha=1");

    auto rfq_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::ReverseFIFOPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(rfq_acts.size() >= 2 && rfq_acts[0] == 0 && rfq_acts[1] == 1,
          "ReverseFIFOPolicy skips alpha=0, assigns alpha=1");
}

static void run_S2()
{
    std::cout << "\n--- S2: 1 pool C=1, FIL=[4,8]  (above diagonal, j1 older) ---\n";
    MDP mdp = make_1pool_1server(2, {100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0}}, {4, 8});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // alpha=0: j1 FIL=8   fifo=1 cmu=1 rfq=0
    CHECK(q[0].job_type       == 1, "alpha=0 candidate is j1 (older)");
    CHECK(q[0].is_fifo_winner == 1, "alpha=0 is_fifo_winner=1");
    CHECK(q[0].is_cmu_winner  == 1, "alpha=0 is_cmu_winner=1  (j1 is also highest c*mu)");
    CHECK(q[0].is_rfq_winner  == 0, "alpha=0 is_rfq_winner=0  (j0 is newer)");
    // alpha=1: j0 FIL=4   fifo=0 cmu=0 rfq=1
    CHECK(q[1].job_type       == 0, "alpha=1 candidate is j0 (newer)");
    CHECK(q[1].is_fifo_winner == 0, "alpha=1 is_fifo_winner=0");
    CHECK(q[1].is_cmu_winner  == 0, "alpha=1 is_cmu_winner=0");
    CHECK(q[1].is_rfq_winner  == 1, "alpha=1 is_rfq_winner=1");
}

static void run_S3()
{
    std::cout << "\n--- S3: 1 pool C=1, only j0 waiting  (single candidate) ---\n";
    MDP mdp = make_1pool_1server(2, {100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0}}, {8, -1});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    CHECK(q.size() == 1,           "queue size = 1");
    CHECK(q[0].is_fifo_winner == 1, "single candidate: fifo=1");
    CHECK(q[0].is_cmu_winner  == 1, "single candidate: cmu=1");
    CHECK(q[0].is_rfq_winner  == 1, "single candidate: rfq=1");
}

static void run_S4()
{
    std::cout << "\n--- S4 HARD: 1 pool C=2, FIL=[8,4], both servers idle ---\n";
    MDP mdp = make_1pool_2server({100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0}}, {8, 4});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // Both entries should be winners on all three criteria (capacity=2 >= 2 entries)
    CHECK(q.size() == 2,           "queue size = 2");
    CHECK(q[0].is_fifo_winner == 1, "alpha=0 fifo=1  (cap=2: both are FIFO winners)");
    CHECK(q[0].is_cmu_winner  == 1, "alpha=0 cmu=1   (cap=2: both are cmu winners)");
    CHECK(q[0].is_rfq_winner  == 1, "alpha=0 rfq=1   (cap=2: both are rfq winners)");
    CHECK(q[1].is_fifo_winner == 1, "alpha=1 fifo=1");
    CHECK(q[1].is_cmu_winner  == 1, "alpha=1 cmu=1");
    CHECK(q[1].is_rfq_winner  == 1, "alpha=1 rfq=1");

    // ModifyStateWithAction always exits to AwaitEvent after action=1.
    // So each policy takes exactly one action=1 in this phase; the second
    // job is assigned in the next phase (after the FIL-refresh event).
    // Here we verify the FIRST-PHASE decision only.
    auto fifo_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::FIFOPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(fifo_acts.size() == 1 && fifo_acts[0] == 1,
          "FIFOPolicy assigns at alpha=0 (both fifo=1; first-phase decision)");

    auto cmu_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::CmuPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(cmu_acts.size() == 1 && cmu_acts[0] == 1,
          "CmuPolicy assigns at alpha=0 (both cmu=1; first-phase decision)");

    auto rfq_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::ReverseFIFOPolicy p(nullptr, VarGroup{});
        return p.GetAction(st); });
    CHECK(rfq_acts.size() == 1 && rfq_acts[0] == 1,
          "ReverseFIFOPolicy assigns at alpha=0 (both rfq=1; first-phase decision)");
}

static void run_S5()
{
    std::cout << "\n--- S5 HARD: 1 pool C=2, 1 server already busy, only j1 waiting ---\n";
    // busy_on[pool0][j0_idx=0] = 1  → effective C = 2-1 = 1
    MDP mdp = make_1pool_2server({100.0, 300.0});
    MDP::State s = make_state(mdp, {{1,0}}, {-1, 4});  // j0 busy, j1 waiting
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    CHECK(q.size() == 1,           "queue size = 1  (only j1 waiting)");
    CHECK(q[0].job_type       == 1, "candidate is j1");
    CHECK(q[0].is_fifo_winner == 1, "fifo=1  (single candidate, effective C=1)");
    CHECK(q[0].is_cmu_winner  == 1, "cmu=1");
    CHECK(q[0].is_rfq_winner  == 1, "rfq=1");
}

static void run_S6()
{
    std::cout << "\n--- S6 HARD: 2 pools C=1 each, both idle, FIL=[8,4] ---\n";
    MDP mdp = make_2pool_1server_each({100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0},{0,0}}, {8, 4});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // Queue (FIFO sorted): [(p0,j0,8),(p1,j0,8),(p0,j1,4),(p1,j1,4)]
    // Each pool C=1:
    //   pool0: fifo winner=j0, cmu winner=j1, rfq winner=j1
    //   pool1: same
    CHECK(q.size() == 4, "queue size = 4");
    // Pool 0 entries
    CHECK(q[0].server_index == 0 && q[0].job_type == 0, "q[0] = (pool0, j0)");
    CHECK(q[0].is_fifo_winner == 1, "q[0] fifo=1");
    CHECK(q[0].is_cmu_winner  == 0, "q[0] cmu=0  (j1 has higher c*mu for pool0)");
    CHECK(q[0].is_rfq_winner  == 0, "q[0] rfq=0  (j1 is newer for pool0)");
    CHECK(q[2].server_index == 0 && q[2].job_type == 1, "q[2] = (pool0, j1)");
    CHECK(q[2].is_fifo_winner == 0, "q[2] fifo=0");
    CHECK(q[2].is_cmu_winner  == 1, "q[2] cmu=1");
    CHECK(q[2].is_rfq_winner  == 1, "q[2] rfq=1");
    // Pool 1 entries (same per-pool labels)
    CHECK(q[1].server_index == 1 && q[1].job_type == 0, "q[1] = (pool1, j0)");
    CHECK(q[1].is_fifo_winner == 1, "q[1] fifo=1");
    CHECK(q[1].is_cmu_winner  == 0, "q[1] cmu=0");
    CHECK(q[1].is_rfq_winner  == 0, "q[1] rfq=0");
    CHECK(q[3].server_index == 1 && q[3].job_type == 1, "q[3] = (pool1, j1)");
    CHECK(q[3].is_fifo_winner == 0, "q[3] fifo=0");
    CHECK(q[3].is_cmu_winner  == 1, "q[3] cmu=1");
    CHECK(q[3].is_rfq_winner  == 1, "q[3] rfq=1");
}

static void run_S7()
{
    std::cout << "\n--- S7 HARD: 1 pool C=2, 3 job types, FIL=[8,6,4], c=[100,200,300] ---\n";
    MDP mdp = make_1pool_2server({100.0, 200.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0,0}}, {8, 6, 4});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // C=2; c*mu: j0=35, j1=70, j2=105
    // FIFO top-2: j0(FIL=8), j1(FIL=6)  → fifo=[1,1,0]
    // Cmu  top-2: j2(105),   j1(70)      → cmu =[0,1,1]
    // RFQ  top-2: j2(FIL=4), j1(FIL=6)  → rfq =[0,1,1]
    CHECK(q.size() == 3, "queue size = 3");
    CHECK(q[0].job_type == 0, "alpha=0 is j0 (oldest)");
    CHECK(q[1].job_type == 1, "alpha=1 is j1");
    CHECK(q[2].job_type == 2, "alpha=2 is j2 (newest)");

    CHECK(q[0].is_fifo_winner == 1, "j0 fifo=1  (0 older entries < C=2)");
    CHECK(q[1].is_fifo_winner == 1, "j1 fifo=1  (1 older entry  < C=2)");
    CHECK(q[2].is_fifo_winner == 0, "j2 fifo=0  (2 older entries = C=2)");

    CHECK(q[0].is_cmu_winner  == 0, "j0 cmu=0   (2 higher-cmu entries = C=2)");
    CHECK(q[1].is_cmu_winner  == 1, "j1 cmu=1   (1 higher-cmu entry  < C=2)");
    CHECK(q[2].is_cmu_winner  == 1, "j2 cmu=1   (0 higher-cmu entries < C=2)");

    CHECK(q[0].is_rfq_winner  == 0, "j0 rfq=0   (2 newer entries = C=2)");
    CHECK(q[1].is_rfq_winner  == 1, "j1 rfq=1   (1 newer entry  < C=2)");
    CHECK(q[2].is_rfq_winner  == 1, "j2 rfq=1   (0 newer entries < C=2)");

    // Each phase ends after one action=1 (ModifyStateWithAction → AwaitEvent).
    // FIFO: assigns j0 at alpha=0 (fifo=1) → phase trace {1}
    // Cmu:  skips j0 (cmu=0), assigns j1 (cmu=1) → phase trace {0,1}
    // RFQ:  skips j0 (rfq=0), assigns j1 (rfq=1) → phase trace {0,1}
    auto fifo_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::FIFOPolicy p(nullptr, VarGroup{}); return p.GetAction(st); });
    CHECK(fifo_acts.size() == 1 && fifo_acts[0] == 1,
          "FIFOPolicy phase trace: {1}  (j0 is fifo_winner)");

    auto cmu_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::CmuPolicy p(nullptr, VarGroup{}); return p.GetAction(st); });
    CHECK(cmu_acts.size() == 2 && cmu_acts[0] == 0 && cmu_acts[1] == 1,
          "CmuPolicy phase trace: {0,1}  (skip j0 cmu=0, assign j1 cmu=1)");

    auto rfq_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::ReverseFIFOPolicy p(nullptr, VarGroup{}); return p.GetAction(st); });
    CHECK(rfq_acts.size() == 2 && rfq_acts[0] == 0 && rfq_acts[1] == 1,
          "ReverseFIFOPolicy phase trace: {0,1}  (skip j0 rfq=0, assign j1 rfq=1)");
}

static void run_S8()
{
    std::cout << "\n--- S8: FIL tie FIL=[6,6] C=1  (strict-inequality edge case) ---\n";
    MDP mdp = make_1pool_1server(2, {100.0, 300.0});
    MDP::State s = make_state(mdp, {{0,0}}, {6, 6});
    const auto& q = s.server_manager.action_queue;
    print_queue(q, s.queue_manager.get_FIL_waiting(), mdp.cost_rates, mdp.server_static_info);

    // Strict inequality: tied FIL does NOT count as "better"
    // → count_fifo_better = 0 for BOTH → both fifo=1
    // → count_rfq_better  = 0 for BOTH → both rfq=1
    // → count_cmu_better: j0(35)<j1(105) so j0 has 1 better → cmu=0; j1 has 0 → cmu=1
    CHECK(q[0].is_fifo_winner == 1, "j0 fifo=1  (tied FIL: strict < gives 0 better)");
    CHECK(q[1].is_fifo_winner == 1, "j1 fifo=1  (tied FIL: strict < gives 0 better)");
    CHECK(q[0].is_rfq_winner  == 1, "j0 rfq=1   (tied FIL: strict > gives 0 newer)");
    CHECK(q[1].is_rfq_winner  == 1, "j1 rfq=1   (tied FIL: strict > gives 0 newer)");
    CHECK(q[0].is_cmu_winner  == 0, "j0 cmu=0   (j1 has strictly higher c*mu)");
    CHECK(q[1].is_cmu_winner  == 1, "j1 cmu=1");

    // FIFO assigns whichever comes first in FIFO sort (j0 by tie-breaking on job index)
    auto fifo_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::FIFOPolicy p(nullptr, VarGroup{}); return p.GetAction(st); });
    CHECK(!fifo_acts.empty() && fifo_acts[0] == 1, "FIFOPolicy assigns at alpha=0");

    // ReverseFIFO: both rfq=1 → also assigns at alpha=0 (same as FIFO on tie)
    auto rfq_acts = trace_policy(mdp, s, [](const MDP::State& st){
        DynaPlex::Models::queue_mdp::ReverseFIFOPolicy p(nullptr, VarGroup{}); return p.GetAction(st); });
    CHECK(!rfq_acts.empty() && rfq_acts[0] == 1, "ReverseFIFOPolicy also assigns at alpha=0 (tie)");
}

// =========================================================================
// main
// =========================================================================

int main()
{
    std::cout << "========================================\n";
    std::cout << "  action_label_test\n";
    std::cout << "========================================\n";

    run_S1();
    run_S2();
    run_S3();
    run_S4();
    run_S5();
    run_S6();
    run_S7();
    run_S8();

    std::cout << "\n========================================\n";
    std::cout << "  Results: " << g_pass << " passed, " << g_fail << " failed\n";
    std::cout << "========================================\n";
    return g_fail > 0 ? 1 : 0;
}
