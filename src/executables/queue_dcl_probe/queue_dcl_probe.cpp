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
//   mode(candidate_queue; use pe for per-event) hold(0)
//   N(20000) M(400) tick(3.0) base_h(100) eval_traj(100) eval_periods(50000)
//   eval_warmup(128) eval_seed(13021984) readout_battery(0)
//   rvi_diag(0) rvi_post_tick_cost(0) rvi_policy_stable(0)
//   visit_steps(0) visit_warmup(20000)
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
#include <tuple>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "dynaplex/retrievestate.h"
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

// Policy wrapper around a directly computed RVI solution.  The normal policy
// factory intentionally hides the solution object; diagnostics need access to
// g*, the stopping criterion, and the exact action map without solving RVI a
// second time.
class RVISolutionPolicy : public DynaPlex::PolicyInterface {
    const qm::MDP* mdp;
    std::shared_ptr<const qm::MDP::RVISolution> solution;
    DynaPlex::VarGroup config;
public:
    RVISolutionPolicy(const qm::MDP* mdp,
                      std::shared_ptr<const qm::MDP::RVISolution> solution)
        : mdp(mdp), solution(std::move(solution))
    {
        config.Add("id", std::string("RVI_solution_diagnostic"));
        config.Add("M", (int64_t)this->solution->M);
    }
    std::string TypeIdentifier() const override { return "RVI_solution_diagnostic"; }
    const DynaPlex::VarGroup& GetConfig() const override { return config; }
    void SetAction(std::span<DynaPlex::Trajectory> trajectories) const override {
        for (auto& t : trajectories) {
            const auto& s = DynaPlex::RetrieveState<qm::MDP::State>(t.GetState());
            t.NextAction = mdp->EvaluateRVIPolicy(*solution, s);
        }
    }
};

static std::string DecisionContext(const qm::MDP::State& s)
{
    const int64_t counter = s.server_manager.get_action_counter();
    int64_t pool = -1;
    if (counter >= 0 && counter < (int64_t)s.server_manager.action_queue.size())
        pool = s.server_manager.action_queue[(size_t)counter].server_index;

    std::ostringstream out;
    out << "pool=" << pool << ",counter=" << counter << ",busy=";
    for (size_t k = 0; k < s.server_manager.busy_on.size(); ++k) {
        if (k) out << '/';
        out << '[';
        for (size_t j = 0; j < s.server_manager.busy_on[k].size(); ++j) {
            if (j) out << ',';
            out << s.server_manager.busy_on[k][j];
        }
        out << ']';
    }
    const auto fil = s.queue_manager.get_FIL_waiting();
    out << ",FIL=(";
    for (size_t n = 0; n < fil.size(); ++n) {
        if (n) out << ',';
        out << fil[n];
    }
    out << ')';
    return out.str();
}

// Compare PPO and RVI on the actual state distribution, rather than on one
// hand-constructed canonical slice.  We run once under each driver policy;
// every AwaitAction state is queried by both policies before the driver's
// action is applied.  The result is therefore visitation weighted and includes
// every server occupancy, empty-queue pattern, pool, and action_counter reached.
static void PrintVisitedPolicyAgreement(
    const qm::MDP& raw_mdp,
    const DynaPlex::MDP& fw_mdp,
    const DynaPlex::Policy& ppo,
    const DynaPlex::Policy& rvi,
    bool drive_with_ppo,
    int64_t warmup_steps,
    int64_t sample_steps,
    int64_t rng_seed)
{
    DynaPlex::Trajectory traj;
    traj.RNGProvider.SeedEventStreams(true, rng_seed, 0, 0);
    auto one = std::span<DynaPlex::Trajectory>(&traj, 1);
    fw_mdp->InitiateState(one);

    const auto& driver = drive_with_ppo ? ppo : rvi;
    for (int64_t step = 0; step < warmup_steps; ++step) {
        if (traj.Category.IsAwaitAction())
            fw_mdp->IncorporateAction(one, driver);
        else
            fw_mdp->IncorporateEvent(one);
    }

    const int A = (int)raw_mdp.n_jobs + 1;
    std::vector<std::vector<int64_t>> pairs(
        (size_t)A, std::vector<int64_t>((size_t)A, 0));
    std::map<std::string, std::pair<int64_t,int64_t>> context_counts;
    std::map<std::string, int64_t> mismatch_states;
    int64_t decisions = 0, disagreements = 0, invalid = 0;

    for (int64_t step = 0; step < sample_steps; ++step) {
        if (!traj.Category.IsAwaitAction()) {
            fw_mdp->IncorporateEvent(one);
            continue;
        }

        const auto& state = DynaPlex::RetrieveState<qm::MDP::State>(traj.GetState());
        const std::string full_context = DecisionContext(state);
        const auto comma = full_context.find(",FIL=");
        const std::string structural_context = full_context.substr(0, comma);

        ppo->SetAction(one);
        const int64_t a_ppo = traj.NextAction;
        rvi->SetAction(one);
        const int64_t a_rvi = traj.NextAction;

        ++decisions;
        auto& ctx = context_counts[structural_context];
        ++ctx.first;
        if (a_ppo < 0 || a_ppo >= A || a_rvi < 0 || a_rvi >= A) {
            ++invalid;
        } else {
            ++pairs[(size_t)a_rvi][(size_t)a_ppo];
            if (a_ppo != a_rvi) {
                ++disagreements;
                ++ctx.second;
                ++mismatch_states[full_context + ":" +
                    std::to_string(a_rvi) + "->" + std::to_string(a_ppo)];
            }
        }

        traj.NextAction = drive_with_ppo ? a_ppo : a_rvi;
        fw_mdp->IncorporateAction(one);
    }

    const double agreement = decisions > 0
        ? 100.0 * (double)(decisions - disagreements - invalid) / (double)decisions
        : 0.0;
    std::cout << "\n  [visited-policy-compare] driver="
              << (drive_with_ppo ? "PPO" : "RVI")
              << " decisions=" << decisions
              << " agreement=" << std::fixed << std::setprecision(3) << agreement << "%"
              << " disagreements=" << disagreements
              << " invalid=" << invalid << "\n";
    std::cout << "  rows=RVI action, columns=PPO action\n      ";
    for (int a = 0; a < A; ++a) std::cout << std::setw(10) << ("PPO" + std::to_string(a));
    std::cout << "\n";
    for (int r = 0; r < A; ++r) {
        std::cout << "  RVI" << r << ' ';
        for (int a = 0; a < A; ++a) std::cout << std::setw(10) << pairs[(size_t)r][(size_t)a];
        std::cout << "\n";
    }

    std::vector<std::pair<std::string,std::pair<int64_t,int64_t>>> contexts(
        context_counts.begin(), context_counts.end());
    std::sort(contexts.begin(), contexts.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second.second > rhs.second.second;
    });
    std::cout << "  structural contexts (visits, disagreements):\n";
    for (const auto& [name, counts] : contexts) {
        if (counts.second == 0) continue;
        std::cout << "    " << name << "  visits=" << counts.first
                  << "  disagree=" << counts.second
                  << " (" << std::setprecision(3)
                  << 100.0 * (double)counts.second / (double)counts.first << "%)\n";
    }

    std::vector<std::pair<std::string,int64_t>> top(mismatch_states.begin(), mismatch_states.end());
    std::sort(top.begin(), top.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second > rhs.second;
    });
    std::cout << "  most visited disagreement states:\n";
    for (size_t i = 0; i < std::min<size_t>(20, top.size()); ++i)
        std::cout << "    " << top[i].second << "x  " << top[i].first << "\n";
}

// Exhaustive policy comparison on the canonical two-job state slice used by
// the queue diagnostics: pool 0 has one type-0 job in service, pool 1 is idle,
// both queues are nonempty, and the next per-event decision belongs to pool 1.
// Unlike the older heatmap printer, this understands per-event actions
// directly (0=idle, 1=type 0, 2=type 1).
static void PrintCanonicalPolicyAgreement(
    const qm::MDP& raw_mdp,
    const DynaPlex::MDP& fw_mdp,
    const DynaPlex::Policy& ppo,
    const DynaPlex::Policy& rvi,
    int max_fil)
{
    if (!raw_mdp.per_event_mode || raw_mdp.n_jobs != 2 || raw_mdp.k_servers < 2) {
        std::cout << "  [policy-compare] skipped: canonical grid requires "
                     "per_event, two jobs, and at least two pools\n";
        return;
    }

    DynaPlex::Trajectory traj;
    traj.RNGProvider.SeedEventStreams(true, 42, 0, 0);
    auto one = std::span<DynaPlex::Trajectory>(&traj, 1);
    fw_mdp->InitiateState(one);

    const int A = (int)raw_mdp.n_jobs + 1;
    std::vector<std::vector<int64_t>> pair_counts(
        (size_t)A, std::vector<int64_t>((size_t)A, 0));
    int64_t total = 0, agree = 0;
    std::vector<std::tuple<int,int,int64_t,int64_t>> examples;

    for (int f0 = 0; f0 <= max_fil; ++f0) {
        for (int f1 = 0; f1 <= max_fil; ++f1) {
            auto& s = DynaPlex::RetrieveState<qm::MDP::State>(traj.GetState());
            s = qm::MDP::State{};
            s.queue_manager.initialize(raw_mdp.n_jobs, raw_mdp.tick_rate,
                                       raw_mdp.arrival_rates, raw_mdp.max_queue_depth);
            s.queue_manager.set_fil(0, (int64_t)f0);
            s.queue_manager.set_fil(1, (int64_t)f1);
            s.server_manager.initialize(&raw_mdp.server_static_info, raw_mdp.n_jobs);
            s.server_manager.busy_on[0][0] = 1;
            s.server_manager.generate_actions_per_event(s.queue_manager.get_FIL_waiting());
            s.server_manager.set_action_counter(0);
            s.server_manager.update_total_service_rate();
            s.cat = DynaPlex::StateCategory::AwaitAction();
            traj.Category = s.cat;

            ppo->SetAction(one);
            const int64_t a_ppo = traj.NextAction;
            rvi->SetAction(one);
            const int64_t a_rvi = traj.NextAction;

            if (a_ppo < 0 || a_ppo >= A || a_rvi < 0 || a_rvi >= A) {
                std::cout << "  [policy-compare] invalid action at FIL=("
                          << f0 << "," << f1 << "): PPO=" << a_ppo
                          << " RVI=" << a_rvi << "\n";
                continue;
            }
            ++total;
            ++pair_counts[(size_t)a_rvi][(size_t)a_ppo];
            if (a_ppo == a_rvi) {
                ++agree;
            } else if (examples.size() < 30) {
                examples.push_back({f0, f1, a_rvi, a_ppo});
            }
        }
    }

    const double pct = total > 0 ? 100.0 * (double)agree / (double)total : 0.0;
    std::cout << "\n  [policy-compare] canonical FIL grid 0.." << max_fil
              << ": agreement=" << agree << "/" << total
              << " (" << std::fixed << std::setprecision(2) << pct << "%)\n";
    std::cout << "  rows=RVI action, columns=PPO action (0=idle,1=type0,2=type1)\n      ";
    for (int a = 0; a < A; ++a) std::cout << std::setw(10) << ("PPO" + std::to_string(a));
    std::cout << "\n";
    for (int r = 0; r < A; ++r) {
        std::cout << "  RVI" << r << " ";
        for (int a = 0; a < A; ++a)
            std::cout << std::setw(10) << pair_counts[(size_t)r][(size_t)a];
        std::cout << "\n";
    }
    if (!examples.empty()) {
        std::cout << "  first disagreements (FIL0,FIL1:RVI->PPO):";
        for (const auto& [f0, f1, ar, ap] : examples)
            std::cout << " (" << f0 << "," << f1 << ":" << ar << "->" << ap << ")";
        std::cout << "\n";
    }
}

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
    const int64_t EVAL_WARMUP  = I("eval_warmup", 128);
    const int64_t EVAL_SEED    = I("eval_seed", 13021984);
    // Diagnostic-only.  The locked headline evaluation below always compares
    // exactly the predeclared PPO argmax policy with RVI.  Keeping alternative
    // readouts off by default prevents post-evaluation policy selection.
    const bool    READOUT_BATTERY = I("readout_battery", 0) != 0;
    const bool    RVI_DIAG      = I("rvi_diag", 0) != 0;
    const int64_t RVI_DIAG_TRAJ = I("rvi_diag_traj", 32);
    const int64_t RVI_DIAG_STEPS= I("rvi_diag_steps", 200000);
    const int64_t RVI_DIAG_WARM = I("rvi_diag_warmup", 20000);
    const int64_t RVI_DIAG_THREADS = I("rvi_diag_threads", 8);
    const int64_t VISIT_STEPS   = I("visit_steps", 0);
    const int64_t VISIT_WARMUP  = I("visit_warmup", 20000);
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
    // hold=1: an idle capacity-unit decision remains in force until the next
    // arrival or completion instead of being reconsidered at every tick.
    if (I("hold", 0) != 0)        cfg.Set("hold_actions_until_real_event", true);
    // force_late=1: SLA escalation — late FILs are served with forced priority
    // (per_event only); the learned policy decides pre-deadline only.
    if (I("force_late", 0) != 0) cfg.Set("force_late_service", true);

    const int64_t H = int64_t((double)BASE_H * TICK_RATE);

    // Lambda from the raw MDP (for physical cost rate = mean * Lambda).
    qm::MDP raw_mdp(cfg);
    const double Lambda = raw_mdp.uniformization_rate;

    auto mdp = dp.GetMDP(cfg);

    VarGroup eval_cfg;
    eval_cfg.Add("number_of_trajectories", EVAL_TRAJ);
    eval_cfg.Add("periods_per_trajectory",  EVAL_PERIODS);
    eval_cfg.Add("warmup_periods",          EVAL_WARMUP);
    eval_cfg.Add("rng_seed",               EVAL_SEED);
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
    DynaPlex::Policy rvi_policy;
    std::shared_ptr<qm::MDP::RVISolution> rvi_solution;
    if (BENCH_MODE == 1) {
        auto fifo = mdp->GetPolicy("FIFO policy");
        // RVI truncation control.  rvi_m>0 pins the FIL clamp to a FIXED M
        // (bypasses the auto-select loop) so g*(M) can be swept directly — the
        // clean way to map convergence for tail-heavy rewards (queue-lateness,
        // tardiness flux) where the auto-loop's between-M criterion stops early.
        // Otherwise rvi_tol drives the auto-select loop (default 0.01; too loose
        // for tail-heavy rewards, whose g* keeps drifting with FIL depth).
        if (RVI_DIAG) {
            qm::MDP::RVISolution sol;
            if (I("rvi_m", 0) > 0) {
                sol = raw_mdp.runRVI(
                    (int)I("rvi_m", 0),
                    (int)I("rvi_max_iter", 10000),
                    false,
                    (int)I("rvi_policy_stable", 0),
                    I("rvi_post_tick_cost", 0) != 0);
            } else {
                sol = raw_mdp.runRVI(D("rvi_tol", 0.01), false);
            }
            rvi_solution = std::make_shared<qm::MDP::RVISolution>(std::move(sol));
            rvi_policy = std::make_shared<RVISolutionPolicy>(&raw_mdp, rvi_solution);
        } else {
            VarGroup rvi_cfg{{"id", std::string("RVI_optimal")}, {"silent", int64_t(1)}};
            if (I("rvi_m", 0) > 0) rvi_cfg.Add("M", I("rvi_m", 0));
            else                   rvi_cfg.Add("rel_tol", D("rvi_tol", 0.01));
            rvi_policy = mdp->GetPolicy(rvi_cfg);
        }

        auto bench = comparer.Compare({fifo, rvi_policy});
        bench[0].Get("mean", fifo_mean);
        bench[1].Get("mean", rvi_mean);

        if (RVI_DIAG && rvi_solution) {
            const auto& sol = *rvi_solution;
            const char* stop = sol.stopped_by_span ? "span"
                : sol.stopped_by_g_stable ? "g_stable" : "max_iter";
            std::cout << "\n  [rvi-diag] M=" << sol.M
                      << " states=" << sol.state_count
                      << " transitions=" << sol.transition_count
                      << " iterations=" << sol.iterations
                      << " stop=" << stop
                      << " post_tick_cost=" << (sol.post_tick_cost ? 1 : 0)
                      << " g*=" << std::setprecision(12) << sol.g_star
                      << " final_span=" << sol.final_span
                      << " final_policy_changes=" << sol.final_policy_changes
                      << " policy_stable=" << sol.policy_stable_count << "\n"
                      << "  [rvi-diag] evaluating extracted policy on raw simulator...\n"
                      << std::flush;
            auto raw = qm::EvaluatePolicyRawParallel(
                raw_mdp, rvi_policy, RVI_DIAG_TRAJ, RVI_DIAG_STEPS,
                RVI_DIAG_WARM, 4242, RVI_DIAG_THREADS);
            std::cout << "  [rvi-diag] raw policy evaluation:"
                      << " event_cost/rvi_step=" << raw.mean_cost_per_rvi_step
                      << " explicit_rvi_cost/rvi_step=" << raw.mean_cost_per_rvi_step_rvi
                      << " immediate_cost/rvi_step=" << raw.mean_cost_per_step_gic
                      << " gic_minus_g*=" << (raw.mean_cost_per_step_gic - sol.g_star)
                      << " std_error=" << raw.std_error
                      << " action_steps=" << raw.total_action_steps
                      << " real_events=" << raw.total_real_event_steps
                      << " fil_refresh=" << raw.total_fil_refresh_steps << "\n"
                      << std::flush;
        }
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
              << "  mode=" << S("mode", "candidate_queue")
              << "  hold=" << (I("hold", 0) != 0 ? 1 : 0)
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

            double nn_mean = 0.0;
            try {
                if (rvi_policy) {
                    // Locked audit comparison: both already-frozen DynaPlex policy
                    // objects enter ONE comparer call.  PolicyComparer resets every
                    // policy to the same evaluation RNG streams and trajectory IDs,
                    // so `error` for PPO is the paired SE of PPO-RVI.
                    std::cout << "  [audit-eval] locked policies in one paired Compare() call...\n"
                              << "      RVI type=" << rvi_policy->TypeIdentifier() << "\n"
                              << "      PPO type=" << nn->TypeIdentifier() << " (fixed argmax)\n"
                              << "      trajectories=" << EVAL_TRAJ
                              << " periods=" << EVAL_PERIODS
                              << " warmup=" << EVAL_WARMUP
                              << " eval_seed=" << EVAL_SEED << "\n" << std::flush;

                    auto audit = comparer.Compare({rvi_policy, nn}, 0);
                    double rvi_audit_mean = 0.0, rvi_audit_error = 0.0;
                    double nn_error = 0.0, paired_delta = 0.0, paired_error = 0.0;
                    audit[0].Get("absolute_mean",  rvi_audit_mean);
                    audit[0].Get("absolute_error", rvi_audit_error);
                    audit[1].Get("absolute_mean",  nn_mean);
                    audit[1].Get("absolute_error", nn_error);
                    audit[1].Get("mean",            paired_delta);
                    audit[1].Get("error",           paired_error);

                    std::cout << std::fixed << std::setprecision(6)
                              << "      RVI*Lambda=" << rvi_audit_mean * Lambda
                              << "  SE=" << rvi_audit_error * Lambda << "\n"
                              << "      PPO*Lambda=" << nn_mean * Lambda
                              << "  SE=" << nn_error * Lambda << "\n"
                              << "      paired(PPO-RVI)*Lambda=" << paired_delta * Lambda
                              << "  paired_SE=" << paired_error * Lambda
                              << "  CI95=[" << (paired_delta - 1.96 * paired_error) * Lambda
                              << ',' << (paired_delta + 1.96 * paired_error) * Lambda << "]\n"
                              << std::flush;
                } else {
                    // No RVI available (e.g. large instances): still evaluate the
                    // one predeclared PPO policy through PolicyComparer.
                    auto only_ppo = comparer.Compare({nn});
                    only_ppo[0].Get("absolute_mean", nn_mean);
                }

                if (READOUT_BATTERY) {
                    // Explicitly diagnostic and never used for the headline score.
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
                    std::cout << "  [diagnostic-only] readout battery ("
                              << rpols.size() << " policies; headline remains locked argmax)...\n"
                              << std::flush;
                    auto res = comparer.Compare(rpols);
                    for (size_t r = 0; r < rpols.size(); ++r) {
                        double m = 0.0; res[r].Get("absolute_mean", m);
                        std::cout << "      [" << std::left << std::setw(12) << rnames[r] << "] "
                                  << std::fixed << std::setprecision(4)
                                  << "NN*Lambda=" << std::setw(10) << m * Lambda
                                  << "  NN/RVI=" << m / norm << "\n" << std::flush;
                    }
                }
            } catch (const std::exception& e) {
                std::cout << "  [eval EXCEPTION] " << e.what() << "\n" << std::flush;
                continue;
            }
            const double nn_ratio = nn_mean / norm;
            ratios.push_back(nn_ratio);

            const int compare_grid = (int)I("compare_grid", 0);
            if (compare_grid > 0) {
                if (!rvi_policy) {
                    std::cout << "  [policy-compare] skipped: use bench=1 to construct RVI\n";
                } else {
                    PrintCanonicalPolicyAgreement(raw_mdp, mdp, nn, rvi_policy, compare_grid);
                }
            }

            if (VISIT_STEPS > 0 && rvi_policy) {
                PrintVisitedPolicyAgreement(raw_mdp, mdp, nn, rvi_policy,
                                            true, VISIT_WARMUP, VISIT_STEPS, seed + 7000);
                PrintVisitedPolicyAgreement(raw_mdp, mdp, nn, rvi_policy,
                                            false, VISIT_WARMUP, VISIT_STEPS, seed + 8000);
            }

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
