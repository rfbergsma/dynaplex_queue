// queue_matrix.cpp
//
// Factorial experiment driver for the queue routing problem. Sweeps:
//     cell    in {Exp2 (fully flexible), Exp3 (specialist+generalist)}
//     method  in {DCL (gen-1 from FIFO), PPO}
//     reward  in {0 (binary FIL>D), 2 (binary + potential-based shaping)}
//     (reward 1 = unbounded queue-lateness was dropped from the sweep: NaN-prone)
//     budget  in {1x, 10x}   (DCL: N=20k/200k ; PPO: 300/3000 updates)
//     seed    in {1..NSEEDS}
// Default NSEEDS=8 -> 2*2*2*2*8 = 128 runs.
//
// For every run we train one policy and score it (by simulation) against that cell's
// RVI optimum under the SAME reward, reporting:
//     NN/RVI        - cost ratio to optimum (1.0 = optimal)
//     gap_closed%   - 100*(FIFO-NN)/(FIFO-RVI): 100 = optimal, 0 = FIFO, <0 = worse than FIFO
// The seed *distribution* of these is the deliverable (exposes bimodal collapse that a
// single seed hides). Output: one CSV row per run.
//
// Base policy is regular FIFO (descending sort); labels="all". These are held fixed.
//
// ---- Parallelising on Snellius ----
// A single process runs the selected runs SEQUENTIALLY. PPO training is single-threaded,
// so the PPO-10x corner (~80 min/run) makes a full sequential run many hours. To use the
// 192 cores, slice the run list across a SLURM job array:
//     queue_matrix <start> <count>
// runs runs[start, start+count) and writes queue_matrix_part<start>.csv. Launch e.g. 128
// array tasks (start=task_id, count=1); each does one run; concatenate the part files after.
// With no arguments it runs the entire matrix and writes queue_matrix.csv.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/policy.h"
#include "dynaplex/policycomparer.h"
#include "../../../lib/models/models/queue_mdp/mdp.h"

using namespace DynaPlex;
namespace qm = DynaPlex::Models::queue_mdp;

// ----------------------- fixed settings -----------------------
constexpr int64_t NSEEDS    = 8;
constexpr double  TICK_RATE = 3.0;
constexpr int64_t BASE_H    = 100;                // H = BASE_H*TICK_RATE = 300
static const std::string LABELS      = "all";
static const std::string ACTION_SORT = "fifo";

// DCL
constexpr int64_t DCL_N_1X  = 20000;
constexpr int64_t DCL_N_10X = 200000;
constexpr int64_t DCL_M     = 400;
// PPO
constexpr int64_t PPO_UPD_1X  = 300;
constexpr int64_t PPO_UPD_10X = 3000;

// eval
constexpr int64_t EVAL_TRAJ    = 100;
constexpr int64_t EVAL_PERIODS = 500000;

// ----------------------- cell configs -----------------------
// Mirrors mm1_baseline / queue_dcl_probe exactly.
static VarGroup exp3_config() {
    VarGroup srv0; srv0.Add("servers", int64_t(1)); srv0.Add("can_serve", VarGroup::Int64Vec{0});
    srv0.Add("service_rates", VarGroup::DoubleVec{1.0});
    VarGroup srv1; srv1.Add("servers", int64_t(1)); srv1.Add("can_serve", VarGroup::Int64Vec{0,1});
    srv1.Add("service_rates", VarGroup::DoubleVec{1.0,1.0});
    VarGroup cfg;
    cfg.Add("id", std::string("queue_mdp"));
    cfg.Add("discount_factor", 1.0);
    cfg.Add("k_servers", int64_t(2));
    cfg.Add("n_jobs",    int64_t(2));
    cfg.Add("tick_rate", 1.0);
    cfg.Add("reward_type", int64_t(0));
    cfg.Add("arrival_rates", VarGroup::DoubleVec{0.8,0.2});
    cfg.Add("cost_rates",    VarGroup::DoubleVec{100.0,300.0});
    cfg.Add("due_times",     VarGroup::DoubleVec{1.0,1.0});
    cfg.Add("server_type_0", srv0);
    cfg.Add("server_type_1", srv1);
    return cfg;
}

static VarGroup cell_config(DynaPlexProvider& dp, int cell, int64_t reward_type) {
    VarGroup cfg;
    if (cell == 2) {
        auto path = dp.FilePath({"mdp_config_examples","queue_mdp"}, "mdp_config_asym_cost_2s.json");
        cfg = VarGroup::LoadFromFile(path);
        cfg.Set("due_times", VarGroup::DoubleVec{3.0,3.0});
    } else {
        cfg = exp3_config();
    }
    cfg.Set("tick_rate",     TICK_RATE);
    cfg.Set("action_sort",   ACTION_SORT);
    cfg.Set("action_labels", LABELS);
    cfg.Set("reward_type",   reward_type);
    return cfg;
}

// Per-(cell,reward) shared benchmarks (RVI is expensive; compute once and cache).
struct CellBench {
    DynaPlex::MDP mdp;
    DynaPlex::Utilities::PolicyComparer comparer;
    double fifo_mean, rvi_mean, Lambda;
};

int main(int argc, char** argv) {
    auto& dp = DynaPlexProvider::Get();

    // --- build the flat run list: cell, method, reward, budget(1 or 10), seed ---
    struct Run { int cell; std::string method; int64_t reward; int budget; int64_t seed; };
    std::vector<Run> runs;
    for (int cell : {2,3})
        for (std::string method : {std::string("dcl"), std::string("ppo")})
            for (int64_t reward : {int64_t(0), int64_t(2)})
                for (int budget : {1,10})
                    for (int64_t seed = 1; seed <= NSEEDS; ++seed)
                        runs.push_back({cell, method, reward, budget, seed});

    // --- arg parsing: [start count] selects a slice for job-array parallelism ---
    int64_t start = 0, count = (int64_t)runs.size();
    std::string out_name = "queue_matrix.csv";
    if (argc >= 3) {
        start = std::atoll(argv[1]);
        count = std::atoll(argv[2]);
        out_name = "queue_matrix_part" + std::to_string(start) + ".csv";
    }
    const int64_t end = std::min<int64_t>(start + count, (int64_t)runs.size());

    const std::string csv_path = dp.FilePath({"csv_results"}, out_name);
    std::ofstream csv(csv_path);
    const std::string header = "cell,method,reward,budget,seed,FIFO_L,RVI_L,NN_L,NN_over_RVI,gap_closed_pct";
    csv << header << "\n" << std::flush;
    dp.System() << "[queue_matrix] writing " << csv_path << "\n";
    dp.System() << header << "\n";

    const int64_t H = int64_t(BASE_H * TICK_RATE);

    // cache of per-(cell,reward) benchmarks within this process
    std::map<std::pair<int,int64_t>, CellBench> cache;

    auto get_bench = [&](int cell, int64_t reward) -> CellBench& {
        auto key = std::make_pair(cell, reward);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        VarGroup cfg = cell_config(dp, cell, reward);
        qm::MDP raw(cfg);
        const double Lambda = raw.uniformization_rate;
        auto mdp = dp.GetMDP(cfg);
        VarGroup eval_cfg;
        eval_cfg.Add("number_of_trajectories", EVAL_TRAJ);
        eval_cfg.Add("periods_per_trajectory",  EVAL_PERIODS);
        auto comparer = dp.GetPolicyComparer(mdp, eval_cfg);
        auto fifo = mdp->GetPolicy("FIFO policy");
        VarGroup rvi_cfg{{"id", std::string("RVI_optimal")},{"rel_tol",0.01},{"silent",int64_t(1)}};
        auto rvi = mdp->GetPolicy(rvi_cfg);
        auto b = comparer.Compare({fifo, rvi});
        double fm=0, rm=0; b[0].Get("mean",fm); b[1].Get("mean",rm);
        cache.emplace(key, CellBench{ mdp, comparer, fm, rm, Lambda });
        return cache.at(key);
    };

    for (int64_t i = start; i < end; ++i) {
        const Run& r = runs[(size_t)i];
        CellBench& cb = get_bench(r.cell, r.reward);

        double nn_mean = 0.0;
        try {
            if (r.method == "dcl") {
                VarGroup dcl;
                dcl.Add("N", r.budget == 10 ? DCL_N_10X : DCL_N_1X);
                dcl.Add("M", DCL_M);
                dcl.Add("H", H);
                dcl.Add("num_gens", int64_t(1));
                dcl.Add("silent", true);
                dcl.Add("rng_seed", r.seed);
                VarGroup arch; arch.Add("type", std::string("mlp"));
                arch.Add("hidden_layers", VarGroup::Int64Vec{64,32,2});
                dcl.Add("nn_architecture", arch);
                VarGroup nt; nt.Add("early_stopping_patience", int64_t(3));
                dcl.Add("nn_training", nt);
                auto fifo = cb.mdp->GetPolicy("FIFO policy");
                auto d = dp.GetDCL(cb.mdp, fifo, dcl);
                d.TrainPolicy();
                auto nn = d.GetPolicies()[(size_t)1];
                cb.comparer.Compare({nn})[0].Get("mean", nn_mean);
            } else { // ppo
                VarGroup p;
                p.Add("num_envs", int64_t(16));
                p.Add("rollout_steps", int64_t(256));
                p.Add("num_updates", r.budget == 10 ? PPO_UPD_10X : PPO_UPD_1X);
                p.Add("epochs_per_update", int64_t(4));
                p.Add("mini_batch_size", int64_t(256));
                p.Add("learning_rate", 3e-4);
                p.Add("gae_gamma", 0.99);
                p.Add("entropy_coef", 0.01);
                p.Add("rng_seed", r.seed);
                p.Add("silent", true);
                VarGroup arch; arch.Add("hidden_layers", VarGroup::Int64Vec{64,32});
                p.Add("nn_architecture", arch);
                auto ppo = dp.GetPPO(cb.mdp, nullptr, p);
                ppo.TrainPolicy();
                auto nn = ppo.GetPolicy();
                cb.comparer.Compare({nn})[0].Get("mean", nn_mean);
            }
        } catch (const std::exception& e) {
            dp.System() << "[run " << i << " EXCEPTION] " << e.what() << "\n";
            continue;
        }

        const double nn_rvi   = (cb.rvi_mean > 1e-12) ? nn_mean / cb.rvi_mean : 0.0;
        const double denom    = (cb.fifo_mean - cb.rvi_mean);
        const double gap_closed = (std::abs(denom) > 1e-12)
                                ? 100.0 * (cb.fifo_mean - nn_mean) / denom : 0.0;

        std::ostringstream row;
        row << std::fixed << std::setprecision(4)
            << "Exp" << r.cell << "," << r.method << "," << r.reward << "," << r.budget << "x," << r.seed << ","
            << cb.fifo_mean * cb.Lambda << "," << cb.rvi_mean * cb.Lambda << "," << nn_mean * cb.Lambda << ","
            << nn_rvi << "," << std::setprecision(1) << gap_closed;
        csv << row.str() << "\n" << std::flush;
        dp.System() << "[" << (i+1) << "/" << end << "] " << row.str() << "\n";
    }

    dp.System() << "[queue_matrix] done. CSV: " << csv_path << "\n";
    return 0;
}
