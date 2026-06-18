// queue_dsweep.cpp
//
// Deadline sweep: tests the hypothesis that the hardness of the queue cells is driven by
// the DEADLINE (sparse, deadline-gated binary reward). At D=0 the binary cost fires whenever
// any job waits (FIL>0) -> a dense, immediate signal like M/M/1 (which both DCL and PPO solve).
// As D grows, the reward goes silent in the pre-deadline region. If success degrades with D,
// the hardness is the reward sparsity -> reward shaping is the fix.
//
// Sweep (reward_type=0 throughout):
//     cell    in {Exp2, Exp3}
//     method  in {DCL (gen-1 from FIFO), PPO}
//     D_ticks in {0, 3, 6, 9}     (deadline in ticks; due_times = D_ticks/tick_rate)
//     seed    in {1..NSEEDS}
// Default NSEEDS=3 -> 2*2*4*3 = 48 runs.
//
// Output: CSV row per run: cell,method,D_ticks,seed,FIFO_L,RVI_L,NN_L,NN_over_RVI,gap_closed_pct.
//
// Parallelise on Snellius via slicing:  queue_dsweep <start> <count>  -> queue_dsweep_part<start>.csv
// (PPO here uses the 1x budget = 300 updates, single-threaded.)  No args = full sweep sequential.

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

constexpr int64_t NSEEDS    = 3;
constexpr double  TICK_RATE = 3.0;
constexpr int64_t BASE_H    = 100;
static const std::string LABELS = "all", ACTION_SORT = "fifo";
constexpr int64_t DCL_N = 20000, DCL_M = 400;
constexpr int64_t PPO_UPDATES = 300;
constexpr int64_t EVAL_TRAJ = 100, EVAL_PERIODS = 500000;

static VarGroup exp3_config() {
    VarGroup s0; s0.Add("servers",int64_t(1)); s0.Add("can_serve",VarGroup::Int64Vec{0}); s0.Add("service_rates",VarGroup::DoubleVec{1.0});
    VarGroup s1; s1.Add("servers",int64_t(1)); s1.Add("can_serve",VarGroup::Int64Vec{0,1}); s1.Add("service_rates",VarGroup::DoubleVec{1.0,1.0});
    VarGroup cfg;
    cfg.Add("id",std::string("queue_mdp")); cfg.Add("discount_factor",1.0);
    cfg.Add("k_servers",int64_t(2)); cfg.Add("n_jobs",int64_t(2)); cfg.Add("tick_rate",1.0);
    cfg.Add("reward_type",int64_t(0));
    cfg.Add("arrival_rates",VarGroup::DoubleVec{0.8,0.2}); cfg.Add("cost_rates",VarGroup::DoubleVec{100.0,300.0});
    cfg.Add("due_times",VarGroup::DoubleVec{1.0,1.0});
    cfg.Add("server_type_0",s0); cfg.Add("server_type_1",s1);
    return cfg;
}

// build cell config with the deadline (ticks) overridden for both job types
static VarGroup cell_config(DynaPlexProvider& dp, int cell, int64_t D_ticks) {
    VarGroup cfg;
    if (cell == 2) {
        auto path = dp.FilePath({"mdp_config_examples","queue_mdp"}, "mdp_config_asym_cost_2s.json");
        cfg = VarGroup::LoadFromFile(path);
    } else cfg = exp3_config();
    const double D_phys = (double)D_ticks / TICK_RATE;     // physical units (MDP multiplies by tick_rate)
    cfg.Set("due_times", VarGroup::DoubleVec{D_phys, D_phys});
    cfg.Set("tick_rate", TICK_RATE);
    cfg.Set("action_sort", ACTION_SORT);
    cfg.Set("action_labels", LABELS);
    cfg.Set("reward_type", int64_t(0));
    return cfg;
}

struct Bench { DynaPlex::MDP mdp; DynaPlex::Utilities::PolicyComparer comparer; double fifo, rvi, Lambda; };

int main(int argc, char** argv) {
    auto& dp = DynaPlexProvider::Get();

    struct Run { int cell; std::string method; int64_t D; int64_t seed; };
    std::vector<Run> runs;
    for (int cell : {2,3})
        for (std::string method : {std::string("dcl"), std::string("ppo")})
            for (int64_t D : {int64_t(0), int64_t(3), int64_t(6), int64_t(9)})
                for (int64_t seed = 1; seed <= NSEEDS; ++seed)
                    runs.push_back({cell, method, D, seed});

    int64_t start = 0, count = (int64_t)runs.size();
    std::string out_name = "queue_dsweep.csv";
    if (argc >= 3) { start = std::atoll(argv[1]); count = std::atoll(argv[2]);
                     out_name = "queue_dsweep_part" + std::to_string(start) + ".csv"; }
    const int64_t end = std::min<int64_t>(start + count, (int64_t)runs.size());

    const std::string csv_path = dp.FilePath({"csv_results"}, out_name);
    std::ofstream csv(csv_path);
    const std::string header = "cell,method,D_ticks,seed,FIFO_L,RVI_L,NN_L,NN_over_RVI,gap_closed_pct";
    csv << header << "\n" << std::flush;
    dp.System() << "[queue_dsweep] writing " << csv_path << "\n" << header << "\n";

    const int64_t H = int64_t(BASE_H * TICK_RATE);
    std::map<std::pair<int,int64_t>, Bench> cache;

    auto get_bench = [&](int cell, int64_t D) -> Bench& {
        auto key = std::make_pair(cell, D);
        auto it = cache.find(key); if (it != cache.end()) return it->second;
        VarGroup cfg = cell_config(dp, cell, D);
        qm::MDP raw(cfg); double Lambda = raw.uniformization_rate;
        auto mdp = dp.GetMDP(cfg);
        VarGroup ec; ec.Add("number_of_trajectories",EVAL_TRAJ); ec.Add("periods_per_trajectory",EVAL_PERIODS);
        auto comparer = dp.GetPolicyComparer(mdp, ec);
        auto fifo = mdp->GetPolicy("FIFO policy");
        VarGroup rc{{"id",std::string("RVI_optimal")},{"rel_tol",0.01},{"silent",int64_t(1)}};
        auto rvi = mdp->GetPolicy(rc);
        auto b = comparer.Compare({fifo,rvi}); double fm=0,rm=0; b[0].Get("mean",fm); b[1].Get("mean",rm);
        cache.emplace(key, Bench{mdp, comparer, fm, rm, Lambda});
        return cache.at(key);
    };

    for (int64_t i = start; i < end; ++i) {
        const Run& r = runs[(size_t)i];
        Bench& cb = get_bench(r.cell, r.D);
        double nn_mean = 0.0;
        try {
            if (r.method == "dcl") {
                VarGroup d; d.Add("N",DCL_N); d.Add("M",DCL_M); d.Add("H",H);
                d.Add("num_gens",int64_t(1)); d.Add("silent",true); d.Add("rng_seed",r.seed);
                VarGroup a; a.Add("type",std::string("mlp")); a.Add("hidden_layers",VarGroup::Int64Vec{64,32,2}); d.Add("nn_architecture",a);
                VarGroup nt; nt.Add("early_stopping_patience",int64_t(3)); d.Add("nn_training",nt);
                auto fifo = cb.mdp->GetPolicy("FIFO policy");
                auto dcl = dp.GetDCL(cb.mdp, fifo, d); dcl.TrainPolicy();
                cb.comparer.Compare({dcl.GetPolicies()[(size_t)1]})[0].Get("mean", nn_mean);
            } else {
                VarGroup p; p.Add("num_envs",int64_t(16)); p.Add("rollout_steps",int64_t(256));
                p.Add("num_updates",PPO_UPDATES); p.Add("epochs_per_update",int64_t(4)); p.Add("mini_batch_size",int64_t(256));
                p.Add("learning_rate",3e-4); p.Add("gae_gamma",0.99); p.Add("entropy_coef",0.01);
                p.Add("rng_seed",r.seed); p.Add("silent",true);
                VarGroup a; a.Add("hidden_layers",VarGroup::Int64Vec{64,32}); p.Add("nn_architecture",a);
                auto ppo = dp.GetPPO(cb.mdp, nullptr, p); ppo.TrainPolicy();
                cb.comparer.Compare({ppo.GetPolicy()})[0].Get("mean", nn_mean);
            }
        } catch (const std::exception& e) { dp.System() << "[run " << i << " EXCEPTION] " << e.what() << "\n"; continue; }

        const double nn_rvi = (cb.rvi>1e-12)? nn_mean/cb.rvi : 0.0;
        const double denom = cb.fifo - cb.rvi;
        const double gap = (std::abs(denom)>1e-12)? 100.0*(cb.fifo-nn_mean)/denom : 0.0;
        std::ostringstream row;
        row << std::fixed << std::setprecision(4)
            << "Exp" << r.cell << "," << r.method << "," << r.D << "," << r.seed << ","
            << cb.fifo*cb.Lambda << "," << cb.rvi*cb.Lambda << "," << nn_mean*cb.Lambda << ","
            << nn_rvi << "," << std::setprecision(1) << gap;
        csv << row.str() << "\n" << std::flush;
        dp.System() << "[" << (i+1) << "/" << end << "] " << row.str() << "\n";
    }
    dp.System() << "[queue_dsweep] done. CSV: " << csv_path << "\n";
    return 0;
}
