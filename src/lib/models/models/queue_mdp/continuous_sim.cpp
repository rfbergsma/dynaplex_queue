// continuous_sim.cpp
// Continuous-time event-driven simulator for queue_mdp::MDP.
//
// Design summary
// --------------
// The discrete MDP uses a uniformised chain: every "step" is one of
//   arrival | tick | service-completion | nothing | action-decision.
// The continuous simulator replaces this with exact exponential clocks:
//   inter-arrival times ~ Exp(lambda_n), service times ~ Exp(mu).
//
// Decision occasions in the discrete MDP
// ---------------------------------------
// Decisions (AwaitAction) are triggered at:
//   1. Arrivals that find an idle server  -- generate_actions() called fresh
//   2. Service completions               -- generate_actions() called after FIL refresh
//   3. Ticks (and "nothing" events)      -- ONLY when the previous action sequence
//                                           ended with all options skipped (action=0
//                                           for every candidate).  In that case the
//                                           action queue is non-empty but the counter
//                                           was reset to 0; the tick advances FIL by 1
//                                           and re-enters AwaitAction so the policy
//                                           can reconsider with the updated sojourn times.
//
// How this simulator handles case 3
// -----------------------------------
// Tick events are retained in the continuous simulator, fired at regular
// intervals of 1/tick_rate physical time units.  After each tick event
// the sojourn times are advanced by exactly 1/tick_rate and the policy
// is re-queried if unprocessed actions remain -- faithfully mirroring
// the discrete behaviour.
//
// Cost is integrated exactly between consecutive events (piecewise
// linear sojourn time inside each inter-event interval).

#include "mdp.h"
#include <random>
#include <limits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace DynaPlex::Models::queue_mdp {

// ============================================================
//  Internal helpers  (anonymous namespace)
// ============================================================
namespace {

// Exact integral of  c * max(0, tau(t) - T_real)  over [0, dt]
// where tau(t) = tau0 + t  (sojourn time grows linearly between events).
double IntegrateQL(double tau0, double T_real, double c, double dt)
{
    if (dt <= 0.0) return 0.0;
    if (tau0 >= T_real) {
        // Already overdue at interval start: linear + triangular area
        return c * ((tau0 - T_real) * dt + 0.5 * dt * dt);
    }
    // Time until the job first becomes overdue
    double t_cross = T_real - tau0;
    if (t_cross >= dt) return 0.0;       // stays under deadline throughout
    double remaining = dt - t_cross;     // time spent overdue
    return c * 0.5 * remaining * remaining;
}

// Exact integral of  c * 1{tau(t) > T_real}  over [0, dt].
double IntegrateBinary(double tau0, double T_real, double c, double dt)
{
    if (dt <= 0.0) return 0.0;
    if (tau0 >= T_real) return c * dt;
    double t_cross = T_real - tau0;
    if (t_cross >= dt) return 0.0;
    return c * (dt - t_cross);
}

// Build an AwaitAction MDP::State from the continuous sim variables.
// Called at every scheduling decision point so the policy can be queried.
//
// FIL values are obtained by discretising exact sojourn times:
//   f_n = floor(tau_n * tick_rate),  or -1 if queue[n] is empty.
//
// The busy_on[][] configuration mirrors the current server assignments.
MDP::State BuildQueryState(
    const MDP&                              mdp,
    double                                  t,
    const std::vector<double>&              server_finish,
    const std::vector<int64_t>&             server_type,
    const std::vector<std::deque<double>>&  queues)
{
    MDP::State s;

    // --- Queue manager: set discretised FIL values ---
    s.queue_manager.initialize(
        mdp.n_jobs, mdp.tick_rate, mdp.arrival_rates, mdp.max_queue_depth);

    for (int64_t n = 0; n < mdp.n_jobs; ++n) {
        if (queues[(size_t)n].empty()) {
            s.queue_manager.set_fil(n, -1LL);
        } else {
            double tau = t - queues[(size_t)n].front();  // exact sojourn time
            int64_t f  = static_cast<int64_t>(tau * mdp.tick_rate);  // floor
            s.queue_manager.set_fil(n, f);
        }
    }

    // --- Server manager: mirror busy_on[][] from continuous state ---
    s.server_manager.initialize(&mdp.server_static_info, mdp.n_jobs);
    for (size_t k = 0; k < server_finish.size(); ++k) {
        if (server_type[k] >= 0) {
            int64_t job = server_type[k];
            int idx = MDP::ServerDynamicState::canServeIndex(
                mdp.server_static_info, (int64_t)k, job);
            if (idx >= 0)
                s.server_manager.busy_on[k][(size_t)idx] = 1;
        }
    }

    // --- Generate the FIFO-ordered action queue and set action counter ---
    s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting(), mdp.cost_rates);
    s.server_manager.set_action_counter(0);
    s.server_manager.update_total_service_rate();
    s.next_fil_job_type = -1;
    s.cat = DynaPlex::StateCategory::AwaitAction();
    return s;
}

} // anonymous namespace


// ============================================================
//  MDP::SimulateContinuous
// ============================================================
MDP::ContinuousSimResult MDP::SimulateContinuous(
    std::function<int64_t(const State&)> policy_fn,
    int    n_traj,
    double t_max,
    double t_warmup) const
{
    constexpr double INF = std::numeric_limits<double>::infinity();
    const int64_t K = (int64_t)server_static_info.size();
    const int64_t N = n_jobs;

    // Physical deadline for each type  (due_times[n] is in ticks)
    std::vector<double> T_real(N, INF);
    for (int64_t n = 0; n < N; ++n)
        if (tick_rate > 0.0 && n < (int64_t)due_times.size())
            T_real[(size_t)n] = due_times[(size_t)n] / tick_rate;

    std::vector<double> per_time_results;
    std::vector<double> per_event_results;
    std::vector<long long> events_per_traj;
    std::vector<long long> decisions_per_traj;
    per_time_results.reserve(n_traj);
    per_event_results.reserve(n_traj);

    for (int traj = 0; traj < n_traj; ++traj) {

        std::mt19937_64 rng(42ULL + (uint64_t)traj);

        // Sample from Exp(rate); returns INF if rate == 0
        auto exp_sample = [&](double rate) -> double {
            if (rate <= 0.0) return INF;
            std::exponential_distribution<double> dist(rate);
            return dist(rng);
        };

        // ---- Continuous state ----
        std::vector<double>             server_finish(K, INF);   // completion time; INF = idle
        std::vector<int64_t>            server_type(K, -1);      // type being served; -1 = idle
        std::vector<std::deque<double>> queues(N);                // waiting jobs: arrival times, oldest first
        std::vector<double>             next_arrival(N);          // time of next arrival per type

        for (int64_t n = 0; n < N; ++n)
            next_arrival[(size_t)n] = exp_sample(arrival_rates[(size_t)n]);

        // Tick events fire at regular intervals of 1/tick_rate physical time.
        // They mirror the discrete system's FIL-advance events: after a tick
        // we re-query the policy if there are idle servers and waiting jobs,
        // exactly as the discrete handler does when action_queue is non-empty.
        const double tick_interval = (tick_rate > 0.0) ? 1.0 / tick_rate : INF;
        double next_tick = tick_interval;   // time of the next tick event

        // ---- Statistics (accumulated after warmup) ----
        double    cost_accum  = 0.0;
        double    time_accum  = 0.0;
        long long n_events    = 0;
        long long n_decisions = 0;
        bool      recording   = false;   // true once t >= t_warmup
        double    t           = 0.0;

        // ---- Cost integrator: accumulate exact cost over [t, t+dt] ----
        auto add_cost = [&](double dt) {
            for (int64_t n = 0; n < N; ++n) {
                if (queues[(size_t)n].empty()) continue;
                double tau0 = t - queues[(size_t)n].front();
                double c    = cost_rates[(size_t)n];
                if (reward_type == 1)
                    cost_accum += IntegrateQL    (tau0, T_real[(size_t)n], c, dt);
                else
                    cost_accum += IntegrateBinary(tau0, T_real[(size_t)n], c, dt);
            }
            time_accum += dt;
        };

        // ---- Assignment helper ----
        // Processes all assignment decisions at the current event time.
        //
        // Design: mirrors the discrete MDP exactly.  In the discrete chain,
        // after each action=1 the system enters an AwaitEvent FIL-refresh step
        // before the next AwaitAction.  That refresh re-queries generate_actions
        // with the updated queue state (the assigned job is now absent).
        // Here we replicate that by rebuilding the query state from the updated
        // continuous variables after every assignment (outer for-loop).
        // The inner while-loop handles skip decisions (action=0) until the first
        // assignment or until all candidates are exhausted.
        auto make_assignments = [&]() {
            for (;;) {
                // Rebuild from current continuous state (post last assignment)
                State s = BuildQueryState(*this, t, server_finish, server_type, queues);
                if (s.server_manager.action_queue.empty()) break;

                bool assigned = false;
                while (s.server_manager.get_action_counter()
                       < (int64_t)s.server_manager.action_queue.size())
                {
                    int64_t action = policy_fn(s);
                    if (recording) ++n_decisions;

                    if (action == 1) {
                        int64_t ac  = s.server_manager.get_action_counter();
                        int64_t k   = s.server_manager.action_queue[(size_t)ac].server_index;
                        int64_t job = s.server_manager.action_queue[(size_t)ac].job_type;

                        // Mirror to continuous state: dequeue job, start service
                        if (!queues[(size_t)job].empty())
                            queues[(size_t)job].pop_front();
                        server_type[(size_t)k] = job;

                        int idx = ServerDynamicState::canServeIndex(
                            server_static_info, k, job);
                        double mu = (idx >= 0)
                                    ? server_static_info[(size_t)k].mu_kj[(size_t)idx]
                                    : 0.4;
                        server_finish[(size_t)k] = t + exp_sample(mu);

                        assigned = true;
                        break;  // rebuild after this assignment (FIL-refresh step)
                    }

                    s.server_manager.take_action(0);   // skip: advance counter
                }

                if (!assigned) break;  // all candidates skipped → done
            }
        };

        // ---- Main event loop ----
        const double t_end = t_warmup + t_max;

        while (t < t_end) {

            // Find the next event: earliest among server completions,
            // arrivals, and the next periodic tick.
            // win_kind: 0 = arrival, 1 = completion, 2 = tick
            double  next_t   = INF;
            int     win_kind = -1;
            int64_t win_idx  = -1;

            for (int64_t k = 0; k < K; ++k) {
                if (server_finish[(size_t)k] < next_t) {
                    next_t   = server_finish[(size_t)k];
                    win_kind = 1;
                    win_idx  = k;
                }
            }
            for (int64_t n = 0; n < N; ++n) {
                if (next_arrival[(size_t)n] < next_t) {
                    next_t   = next_arrival[(size_t)n];
                    win_kind = 0;
                    win_idx  = n;
                }
            }
            if (next_tick < next_t) {
                next_t   = next_tick;
                win_kind = 2;
                win_idx  = -1;
            }

            // Clamp to simulation end so we don't overshoot t_end
            bool sentinel = (next_t >= t_end);
            if (sentinel) next_t = t_end;

            double dt = next_t - t;

            // Integrate cost over [t, next_t].
            // On the first interval that straddles the warmup boundary,
            // start recording from t_warmup only.
            if (!recording && next_t >= t_warmup) {
                recording = true;
                double t_save = t;
                t = t_warmup;
                add_cost(next_t - t_warmup);
                t = t_save;
            } else if (recording) {
                add_cost(dt);
            }

            t = next_t;

            if (sentinel) break;

            // Process the winning event
            if (win_kind == 1) {
                // Service completion at pool win_idx
                server_finish[(size_t)win_idx] = INF;
                server_type[(size_t)win_idx]   = -1;
                // Real event: count and make fresh assignment decisions
                if (recording) ++n_events;
                make_assignments();
            } else if (win_kind == 0) {
                // Arrival of type win_idx: job joins back of waiting queue
                queues[(size_t)win_idx].push_back(t);
                next_arrival[(size_t)win_idx] = t + exp_sample(arrival_rates[(size_t)win_idx]);
                // Real event: count and make fresh assignment decisions
                if (recording) ++n_events;
                make_assignments();
            } else {
                // Tick event: sojourn times advance by tick_interval.
                // No queue changes — cost was already integrated continuously.
                // Re-query the policy so the discrete system's "reconsider
                // after skip" behaviour is faithfully reproduced.
                next_tick += tick_interval;
                make_assignments();   // no-op if all servers busy or queues empty
            }
        }

        // Store trajectory results
        if (time_accum > 0.0) {
            per_time_results.push_back(cost_accum / time_accum);
            per_event_results.push_back(
                n_events > 0 ? cost_accum / (double)n_events : 0.0);
            events_per_traj.push_back(n_events);
            decisions_per_traj.push_back(n_decisions);
        }
    }

    // ---- Aggregate across trajectories ----
    const int M = (int)per_time_results.size();
    if (M == 0) {
        std::cerr << "[SimulateContinuous] WARNING: no valid trajectories\n";
        return {0.0, 0.0, 0.0, 0, 0};
    }

    double mean_t = 0.0, mean_e = 0.0;
    for (double v : per_time_results)  mean_t += v;
    for (double v : per_event_results) mean_e += v;
    mean_t /= M;
    mean_e /= M;

    double var = 0.0;
    for (double v : per_time_results) {
        double d = v - mean_t;
        var += d * d;
    }
    double std_err = (M > 1) ? std::sqrt(var / ((double)M * (M - 1))) : 0.0;

    long long avg_ev = 0, avg_dec = 0;
    for (long long v : events_per_traj)   avg_ev  += v;
    for (long long v : decisions_per_traj) avg_dec += v;
    avg_ev  /= M;
    avg_dec /= M;

    ContinuousSimResult res;
    res.mean_cost_per_time  = mean_t;
    res.mean_cost_per_event = mean_e;
    res.std_err_per_time    = std_err;
    res.avg_events          = avg_ev;
    res.avg_decisions       = avg_dec;
    return res;
}

// ============================================================
//  MDP::TraceContinuous
// ============================================================
void MDP::TraceContinuous(
    std::function<int64_t(const State&)> policy_fn,
    double  t_trace,
    int64_t rng_seed) const
{
    constexpr double INF = std::numeric_limits<double>::infinity();
    const int64_t K = (int64_t)server_static_info.size();
    const int64_t N = n_jobs;

    std::mt19937_64 rng((uint64_t)rng_seed);
    auto exp_sample = [&](double rate) -> double {
        if (rate <= 0.0) return INF;
        std::exponential_distribution<double> dist(rate);
        return dist(rng);
    };

    // ---- Continuous state ----
    std::vector<double>             server_finish(K, INF);
    std::vector<int64_t>            server_type(K, -1);
    std::vector<std::deque<double>> queues(N);
    std::vector<double>             next_arrival(N);

    for (int64_t n = 0; n < N; ++n)
        next_arrival[(size_t)n] = exp_sample(arrival_rates[(size_t)n]);

    const double tick_interval = (tick_rate > 0.0) ? 1.0 / tick_rate : INF;
    double next_tick = tick_interval;
    double t = 0.0;

    // ---- Print helpers ----
    // Prints current queue sojourn times and server states compactly.
    auto print_state_str = [&]() -> std::string {
        std::ostringstream oss;
        oss << "Q[";
        for (int64_t n = 0; n < N; ++n) {
            if (n > 0) oss << " ";
            oss << "T" << n << ":";
            if (queues[(size_t)n].empty()) {
                oss << "empty";
            } else {
                bool first = true;
                for (double at : queues[(size_t)n]) {
                    oss << (first ? "" : ",");
                    oss << std::fixed << std::setprecision(2) << (t - at);
                    first = false;
                }
            }
        }
        oss << "]  S[";
        for (int64_t k = 0; k < K; ++k) {
            if (k > 0) oss << " ";
            oss << "S" << k << ":";
            if (server_type[(size_t)k] < 0)
                oss << "idle";
            else
                oss << "T" << server_type[(size_t)k];
        }
        oss << "]";
        return oss.str();
    };

    // Runs the policy loop, prints each decision, and mirrors assignments
    // into the continuous state.  Returns the number of decisions examined.
    // After each action=1 the query state is rebuilt from the updated
    // continuous variables (mirrors the discrete FIL-refresh step), so
    // subsequent iterations see the correct residual queue.
    auto trace_and_assign = [&]() -> int {
        int n_examined = 0;

        for (;;) {
            State s = BuildQueryState(*this, t, server_finish, server_type, queues);
            if (s.server_manager.action_queue.empty()) break;

            bool assigned = false;
            while (s.server_manager.get_action_counter()
                   < (int64_t)s.server_manager.action_queue.size())
            {
                int64_t ac  = s.server_manager.get_action_counter();
                int64_t k   = s.server_manager.action_queue[(size_t)ac].server_index;
                int64_t job = s.server_manager.action_queue[(size_t)ac].job_type;

                double tau_job = queues[(size_t)job].empty()
                                 ? -1.0
                                 : t - queues[(size_t)job].front();
                int64_t fil = (tau_job >= 0.0)
                              ? static_cast<int64_t>(tau_job * tick_rate)
                              : -1LL;

                int64_t action = policy_fn(s);
                ++n_examined;

                if (action == 1) {
                    std::cout << "      assign S" << k << " <- T" << job
                              << " (tau=" << std::fixed << std::setprecision(2) << tau_job
                              << " FIL=" << fil << ")";
                    if (!queues[(size_t)job].empty())
                        queues[(size_t)job].pop_front();
                    server_type[(size_t)k] = job;
                    int idx = ServerDynamicState::canServeIndex(
                        server_static_info, k, job);
                    double mu = (idx >= 0)
                                ? server_static_info[(size_t)k].mu_kj[(size_t)idx]
                                : 0.4;
                    double finish_t = t + exp_sample(mu);
                    server_finish[(size_t)k] = finish_t;
                    std::cout << " (completes ~t=" << std::setprecision(3) << finish_t << ")\n";

                    assigned = true;
                    break;  // rebuild after assignment
                } else {
                    std::cout << "      skip  T" << job
                              << " (tau=" << std::fixed << std::setprecision(2) << tau_job
                              << " FIL=" << fil << ")\n";
                    s.server_manager.take_action(0);
                }
            }

            if (!assigned) break;
        }
        return n_examined;
    };

    // ---- Header ----
    std::cout << "\n=== TraceContinuous  seed=" << rng_seed
              << "  t_trace=" << std::fixed << std::setprecision(1) << t_trace
              << "  tick_rate=" << tick_rate << " ===\n";
    std::cout << "  Format: [t]  EVENT  | state | -> decisions\n";
    std::cout << "  " << std::string(70, '-') << "\n";

    int n_events = 0;

    while (t < t_trace) {

        // Find next event
        double  next_t   = INF;
        int     win_kind = -1;
        int64_t win_idx  = -1;

        for (int64_t k = 0; k < K; ++k)
            if (server_finish[(size_t)k] < next_t) {
                next_t = server_finish[(size_t)k]; win_kind = 1; win_idx = k;
            }
        for (int64_t n = 0; n < N; ++n)
            if (next_arrival[(size_t)n] < next_t) {
                next_t = next_arrival[(size_t)n]; win_kind = 0; win_idx = n;
            }
        if (next_tick < next_t) {
            next_t = next_tick; win_kind = 2; win_idx = -1;
        }

        if (next_t >= t_trace) break;
        t = next_t;
        ++n_events;

        if (win_kind == 1) {
            // Service completion
            int64_t type_done = server_type[(size_t)win_idx];
            server_finish[(size_t)win_idx] = INF;
            server_type[(size_t)win_idx]   = -1;

            std::cout << std::fixed << std::setprecision(3)
                      << "t=" << std::setw(7) << t
                      << "  COMPLETION S" << win_idx
                      << " (T" << type_done << ") | "
                      << print_state_str() << "\n";
            int nd = trace_and_assign();
            if (nd == 0) std::cout << "      (no candidates)\n";

        } else if (win_kind == 0) {
            // Arrival
            queues[(size_t)win_idx].push_back(t);
            next_arrival[(size_t)win_idx] =
                t + exp_sample(arrival_rates[(size_t)win_idx]);

            std::cout << std::fixed << std::setprecision(3)
                      << "t=" << std::setw(7) << t
                      << "  ARRIVAL   T" << win_idx
                      << "           | "
                      << print_state_str() << "\n";
            int nd = trace_and_assign();
            if (nd == 0) std::cout << "      (no candidates)\n";

        } else {
            // Tick
            next_tick += tick_interval;

            std::cout << std::fixed << std::setprecision(3)
                      << "t=" << std::setw(7) << t
                      << "  TICK                  | "
                      << print_state_str() << "\n";
            int nd = trace_and_assign();
            if (nd == 0) std::cout << "      (no deferred actions)\n";
        }
    }

    std::cout << "  " << std::string(70, '-') << "\n";
    std::cout << "=== Trace complete: " << n_events
              << " events in t=[0," << std::fixed << std::setprecision(1)
              << t_trace << "] ===\n";
}

} // namespace DynaPlex::Models::queue_mdp
