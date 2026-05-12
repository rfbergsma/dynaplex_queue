#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include "dynaplex/mdp.h"       // DynaPlex::MDP, DynaPlex::Trajectory
#include "dynaplex/policy.h"    // DynaPlex::Policy
#include <deque>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <functional>
#include <unordered_map>
#include <cmath>
#include <cstdlib>

namespace DynaPlex::Models {
	namespace queue_mdp /*must be consistent everywhere for complete mdp definition and associated policies and states (if not defined inline).*/
	{		
		class MDP
		{			
		public:	
			double discount_factor;
			
			struct ServerStaticInfo {
				int64_t servers = 0;
				std::vector<double> mu_kj;       // mu_kj[j] = service rate for can_serve[j]
				std::vector<int64_t> can_serve;  // types of jobs it can serve
			};

			struct Action {
				int64_t server_index;
				int64_t job_type;

				// --- Default constructor (required by VarGroup and STL containers) ---
				Action() : server_index(0), job_type(0) {}

				// --- Full constructor ---
				Action(int64_t s, int64_t j) : server_index(s), job_type(j) {}

				// --- Serialization: convert Action  VarGroup ---
				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vg;
					vg.Add("server_index", server_index);
					vg.Add("job_type", job_type);
					return vg;
				}

				// --- Deserialization: convert VarGroup Action ---
				explicit Action(const DynaPlex::VarGroup& vg) {
					vg.Get("server_index", server_index);
					vg.Get("job_type", job_type);
				}
			};


			struct ServerDynamicState{
				// Reference to static info (not owned)
				const std::vector<ServerStaticInfo>* static_info = nullptr;
				std::vector<std::vector<int64_t>> busy_on;
				
				std::deque<Action> action_queue;
				int64_t action_counter;

				double total_service_rate;

				ServerDynamicState() = default;
				

				void SortActionsFIFO(std::deque<Action>& actions,
					const std::vector<int64_t>& FIL_waiting)
				{
					// Stable sort so that ties keep previous relative order (useful for reproducibility)
					std::stable_sort(actions.begin(), actions.end(),
						[&](const Action& a, const Action& b) {
							const int64_t wa = (a.job_type >= 0 && (size_t)a.job_type < FIL_waiting.size())
								? FIL_waiting[(size_t)a.job_type]
								: INT64_MIN;
							const int64_t wb = (b.job_type >= 0 && (size_t)b.job_type < FIL_waiting.size())
								? FIL_waiting[(size_t)b.job_type]
								: INT64_MIN;

							// Primary: "oldest first" (larger FIL first)
							if (wa != wb) return wa > wb;

							// Secondary: smaller server pool index first
							if (a.server_index != b.server_index) return a.server_index < b.server_index;

							// Tertiary: smaller job index first
							return a.job_type < b.job_type;
						});
				}

				void initialize(const std::vector<ServerStaticInfo>* static_info_ptr, int64_t n_jobs) {
					static_info = static_info_ptr;
					busy_on.clear();
					if (!static_info) return;
					busy_on.resize(static_info->size());
					for (size_t k = 0; k < static_info->size(); ++k) {
						busy_on[k].resize((*static_info)[k].can_serve.size(), 0);
					}

					total_service_rate = 0.0;
					//initialize empty queue
					action_queue.clear();
					action_counter = 0;

					
				}
				
				void generate_actions(std::vector<int64_t> FIL_waiting) {
					#if QUEUE_MDP_DEBUG
					std::cout << "\n[QMDP] generate_actions called\n";
					std::cout << "[QMDP]   old queue size = " << action_queue.size()
						<< ", old action_counter = " << action_counter << "\n";
					#endif
							
					// generate possible actions based on current busy_on and FIL_waiting
					// if FIL_waiting[n] >=0 and there is idle server for job n, add action
					action_queue.clear();
					
					for (size_t n = 0; n < FIL_waiting.size(); ++n) {
						if (FIL_waiting[n] >= 0) { // job n is waiting
							for (size_t k = 0; k < busy_on.size(); ++k) {
								int idx = canServeIndex(*static_info, static_cast<int64_t>(k), static_cast<int64_t>(n));
								int64_t busy_servers = n_servers_busy_server_k(k);
								if (idx >= 0) { // server k can serve job n
									if (busy_servers < (*static_info)[k].servers) {
										// there is idle server for job n
										// add option as many times as there is capacity
										action_queue.push_back(Action{ static_cast<int64_t>(k), static_cast<int64_t>(n) });
										
									}
								}

							}
						}
					}

					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   new queue size = " << action_queue.size()
						<< ", action_counter (unchanged) = " << action_counter << "\n";
					#endif

					SortActionsFIFO(action_queue, FIL_waiting);

				}
	
				int64_t n_servers_busy_server_k(int64_t k) const {
					int64_t total_busy = 0;

					for (size_t j = 0; j < busy_on[k].size(); ++j) {
						total_busy += busy_on[k][j];
					}
					return total_busy;
				}

				// can_assign job if action can be taken based on current busy_on
				bool can_assign_job(int64_t k, int64_t job) const {
					int idx = canServeIndex(*static_info, k, job);
					if (idx < 0) return false;

					// Check if the number of busy servers for this server type is less than the total servers
					if (n_servers_busy_server_k(k) >= (*static_info)[(size_t)k].servers)
						return false;

					// Check if this specific job slot is available for this server type
					return true;
				}



				
				// function takes action and modifies the server dynamic state accordingly
				// increases busy_on for the corresponding server and job type
				// removes actions from the queue that can no longer be taken
				
				void take_action(const int64_t& action) {
					// --- Skip action: just move to next action in the queue ---
					if (action == 0) {
						if (action_counter < (int64_t)action_queue.size()) {
							++action_counter;
						}
						return;
					}

					// --- Execute current action at action_counter ---
					if (action_counter < 0 || action_counter >= (int64_t)action_queue.size()) {
						throw std::runtime_error("action_counter out of bounds in take_action");
					}

					// Snapshot old state
					std::deque<Action> old_queue = action_queue;
					int64_t old_counter = action_counter;

					Action taken_action = action_queue[(size_t)action_counter];

					// --- Perform the action (busy_on, etc.) ---
					assign_job(taken_action.server_index, taken_action.job_type);

					// --- Update action_queue: remove ONLY impossible actions ---
					
					action_queue.erase(
						std::remove_if(
							action_queue.begin(),
							action_queue.end(),
							[this](const Action& a) {
								return !can_assign_job(a.server_index, a.job_type);
							}
						),
						action_queue.end()
					);

					// --- Recompute action_counter ---
					//
					// processed = old_queue[0 .. old_counter]  (INCLUSIVE!)
					// new_counter = number of actions at the front of action_queue
					//               that belong to processed.

					int64_t new_counter = 0;

					for (const auto& new_act : action_queue) {
						bool was_processed = false;

						// Note the <= : we include the executed action as "processed"
						for (int64_t i = 0; i <= old_counter; ++i) {
							const auto& old_act = old_queue[(size_t)i];
							if (old_act.server_index == new_act.server_index &&
								old_act.job_type == new_act.job_type) {
								was_processed = true;
								break;
							}
						}

						if (was_processed) {
							++new_counter;
						}
						else {
							// First unprocessed action in the new queue  stop
							break;
						}
					}

					action_counter = new_counter;
				}


				void set_action_counter(int64_t count) {
					action_counter = count;
				}

				int64_t get_action_counter() const {
					return static_cast<int64_t>(action_counter);
				}

				void assign_job(int64_t k, int64_t job) {
					int idx = canServeIndex(*static_info, k, job);
					if (idx < 0) return;
					if (busy_on[(size_t)k][(size_t)idx] >= (*static_info)[(size_t)k].servers) return;
					busy_on[(size_t)k][(size_t)idx] += 1;

				}

				/*
				//complete job n at server k remove job from busy_on
				void complete_job(int64_t k, int64_t job) {
					int idx = canServeIndex(*static_info, k, job);
					if (idx < 0) return;
					if (busy_on[(size_t)k][(size_t)idx] <= 0) return;
					busy_on[(size_t)k][(size_t)idx] -= 1;
				}
				
				*/
				void complete_job(int64_t k, int64_t job) {
					int idx = canServeIndex(*static_info, k, job);

					#if QUEUE_MDP_DEBUG
						std::cout << "[QMDP]   complete_job called with server=" << k
							<< ", job=" << job << "\n";
						if (idx < 0) {
							std::cout << "[QMDP]   complete_job: canServeIndex < 0 for (k=" << k
								<< ", job=" << job << "), no-op\n";
						}
						else {
							std::cout << "[QMDP]   complete_job: idx=" << idx
								<< ", busy_on[k][idx]=" << busy_on[(size_t)k][(size_t)idx]
								<< "\n";
						}
					#endif

						if (idx < 0) {
							std::cout << "Cannot complete job in server manager" << std::endl;
						}
							
						
						if (busy_on[(size_t)k][(size_t)idx] <= 0) {
					#if QUEUE_MDP_DEBUG
							std::cout << "[QMDP]   complete_job: busy_on already <= 0, no-op\n";
					#endif
							std::cout << "Cannot complete job in server manager" << std::endl;
						return;
					}

					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   complete_job: decrementing busy_on[" << k << "]["
						<< idx << "] from " << busy_on[(size_t)k][(size_t)idx] << " to "
						<< (busy_on[(size_t)k][(size_t)idx] - 1) << "\n";
					#endif

					// --- JOB-LEVEL COMPLETION: free all servers on this job type ---
					for (int64_t k_server = 0; k_server < (int64_t)busy_on.size(); ++k_server) {
						int idx2 = canServeIndex(*static_info, k_server, job);
						if (idx2 >= 0) {
					#if QUEUE_MDP_DEBUG
							if (busy_on[(size_t)k_server][(size_t)idx2] > 0) {
								std::cout << "[QMDP]   complete_job: setting busy_on[" << k_server
									<< "][" << idx2 << "] from "
									<< busy_on[(size_t)k_server][(size_t)idx2]
									<< " to 0\n";
							}
					#endif
							busy_on[(size_t)k_server][(size_t)idx2] = 0;
						}
					}
					
				}

				//returns the index of job in can_serve vector of server k, -1 if cannot serve
				static int canServeIndex(const std::vector<ServerStaticInfo>& S, int64_t k, int64_t job) {
					const auto& v = S[(size_t)k].can_serve;
					auto it = std::find(v.begin(), v.end(), job);
					return (it == v.end()) ? -1 : (int)std::distance(v.begin(), it);
				}

				
				//update total service rate
				void update_total_service_rate() {
					if (!static_info) {
						total_service_rate = 0.0;
						return;
					}
					total_service_rate = get_total_service_rate(*this, *static_info);
				}

				inline double get_total_service_rate(const ServerDynamicState& dyn,
					const std::vector<ServerStaticInfo>& S) {
					double r = 0.0;
					for (size_t k = 0; k < S.size(); ++k)
						for (size_t j = 0; j < S[k].can_serve.size(); ++j)
							r += dyn.busy_on[k][j] * S[k].mu_kj[j];
					return r;
				}

				inline bool assign_job(ServerDynamicState& dyn,
					const std::vector<ServerStaticInfo>& S,
					int64_t k, int64_t job) {
					int idx = canServeIndex(S, k, job);
					if (idx < 0) return false;
					if (dyn.busy_on[(size_t)k][(size_t)idx] >= S[(size_t)k].servers) return false;
					dyn.busy_on[(size_t)k][(size_t)idx] += 1;
					total_service_rate += S[(size_t)k].mu_kj[(size_t)idx];  // idx already computed above
					return true;
				}

				void print_actions() const {
					std::cout << "number of possible actions: ";
					std::cout << action_queue.size() << "\n";

					
					std::cout << "Possible actions: ";
					for (const auto& action : action_queue) {
						std::cout << "(k=" << action.server_index << ", job=" << action.job_type << ") ";
					}
					std::cout << "\n";
				}
				
				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vars;

					vars.Add("busy_on_rows", (int64_t)busy_on.size());
					for (size_t k = 0; k < busy_on.size(); ++k) {
						vars.Add("busy_on_" + std::to_string(k), busy_on[k]);
					}

					vars.Add("total_service_rate", total_service_rate);
					vars.Add("action_counter", action_counter);
					vars.Add("action_queue", action_queue);

					return vars;
				}

				explicit ServerDynamicState(const DynaPlex::VarGroup& vg) {
					int64_t rows = 0;
					vg.Get("busy_on_rows", rows);

					busy_on.clear();
					busy_on.reserve((size_t)rows);

					for (int64_t k = 0; k < rows; ++k) {
						std::vector<int64_t> row;
						vg.Get("busy_on_" + std::to_string(k), row);
						busy_on.push_back(std::move(row));
					}

					// service rate might be derived; only load if you truly store it
					if (vg.HasKey("total_service_rate"))
						vg.Get("total_service_rate", total_service_rate);
					else
						total_service_rate = 0.0;

					vg.Get("action_counter", action_counter);
					vg.Get("action_queue", action_queue);
				}
			};

			std::vector<ServerStaticInfo> server_static_info;

			int64_t n_jobs; // number of different types of jobs
			int64_t k_servers; // number of servers
			std::vector<double> arrival_rates; // arrival rates for each job type n
			double tick_rate; // tick rate
			std::vector<double> cost_rates; // service rates for each server type k
			std::vector<double> due_times; // rewards for completing each job type n
			double uniformization_rate;
			int64_t reward_type;      // 0=binary (FIL>D), 1=queue-lateness (default)
			int64_t max_queue_depth;  // tracked positions per job type: 1=FIL only (default)
			int64_t feature_queue_depth; // NN feature slots per job type (>= max_queue_depth; pads with 0)
			int64_t int_hash = 0;        // config hash — used by EvaluatePolicyRaw(Policy) to build type-erased states

			struct multi_queue {
				// waiting[n] = deque of waiting times for job type n, front = FIL (oldest).
				// Empty deque means no job of type n is currently waiting.
				int64_t max_queue_depth;                     // max tracked positions (from MDP config)
				std::vector<std::deque<int64_t>> waiting;    // waiting[n][0]=FIL, [1]=SIL, [2]=TIL, ...
				std::vector<double> arrival_rates;
				double total_tick_rate;
				double total_arrival_rate;

				multi_queue() = default;

				void initialize(int64_t n_jobs, double tick_rate,
				                std::vector<double> rates, int64_t depth = 1) {
					max_queue_depth   = depth;
					waiting.assign((size_t)n_jobs, std::deque<int64_t>{});
					total_tick_rate   = tick_rate;
					total_arrival_rate = 0.0;
					for (const auto& r : rates) total_arrival_rate += r;
					this->arrival_rates = std::move(rates);
				}

				// ---- Computed FIL shim (backward-compatible; used by generate_actions, RVI, etc.) ----
				std::vector<int64_t> get_FIL_waiting() const {
					std::vector<int64_t> out(waiting.size());
					for (size_t n = 0; n < waiting.size(); ++n)
						out[n] = waiting[n].empty() ? -1 : waiting[n].front();
					return out;
				}

				// ---- Direct FIL setter (tests and RVI state construction) ----
				// Clears any deeper positions for this type so callers that only know FIL stay correct.
				void set_fil(int64_t n, int64_t val) {
					waiting[(size_t)n].clear();
					if (val >= 0)
						waiting[(size_t)n].push_back(val);
					total_arrival_rate = get_total_arrival_rate(arrival_rates);
				}

				// ---- Clamp FIL to M for each type (RVI truncation) ----
				void clamp_fil(int64_t M) {
					for (auto& q : waiting)
						if (!q.empty())
							q.front() = std::min(q.front(), M);
				}

				// ---- Arrival: job of type n joins the back of the queue at waiting time 0 ----
				void arrival(int64_t n) {
					if (n < 0 || n >= static_cast<int64_t>(waiting.size()))
						throw std::runtime_error("arrival: invalid job type");
					if ((int64_t)waiting[(size_t)n].size() < max_queue_depth) {
						waiting[(size_t)n].push_back(0);
						// If this filled the last slot, remove from arrival process
						if ((int64_t)waiting[(size_t)n].size() == max_queue_depth)
							total_arrival_rate -= arrival_rates[(size_t)n];
					}
					// If already at max_queue_depth: event cannot fire (total_arrival_rate excludes it)
				}

				// ---- Tick: increment all tracked waiting times by 1 ----
				void tick() {
					for (auto& q : waiting)
						for (auto& t : q)
							++t;
				}

				// ---- complete_job: FIL served -> shift-up, Koole-sample new bottom position ----
				// Called during the FIL-refresh step (next_fil_job_type != -1).
				// For max_queue_depth==1: identical behaviour to the old code (Koole applied to FIL).
				// For max_queue_depth>1: SIL shifts to FIL deterministically; Koole applied at bottom.
				void complete_job(int64_t n, double uniform_draw) {
					if (n < 0 || n >= static_cast<int64_t>(waiting.size()))
						throw std::runtime_error("complete_job: invalid job type");
					auto& q = waiting[(size_t)n];
					if (q.empty())
						throw std::runtime_error("complete_job: queue is empty for type " + std::to_string(n));

					const bool    was_full   = ((int64_t)q.size() == max_queue_depth);
					const int64_t old_bottom = q.back();   // deepest tracked position before shift

					q.pop_front();  // FIL served; SIL (if any) becomes new FIL

					// Sample what fills the vacancy at the bottom of the tracked queue.
					// Semantics: given a job has been waiting old_bottom ticks, what is the
					// waiting time of the job behind it (if any)?  Same Koole formula as before.
					//
					// Special case: old_bottom == 0 means the served job had just arrived
					// (waiting time 0).  No job can be behind one that just arrived, so the
					// queue must be empty after the shift.  The Koole sampler returns 0 for
					// i<=0 (not -1), so we must handle this explicitly to avoid a spurious
					// zero-waiting-time job being pushed back.  (Matches the old code's
					// `else if (current_fil == 0) { FIL_waiting[n] = -1; }` guard and the
					// analytical NextFILDistribution which returns {-1, 1.0} for i==0.)
					int64_t new_bottom;
					if (old_bottom == 0) {
						new_bottom = -1;  // queue becomes empty; no job can be behind a just-arrived job
					} else {
						new_bottom = (int64_t)sample_next_fil_after_completion(
							(int)old_bottom, arrival_rates[(size_t)n], total_tick_rate, uniform_draw);
					}

					if (new_bottom >= 0)
						q.push_back(new_bottom);

					// If queue was full and is now not full, re-enable arrivals for this type
					if (was_full && (int64_t)q.size() < max_queue_depth)
						total_arrival_rate += arrival_rates[(size_t)n];
				}

				// ---- Rate helpers ----
				double get_total_arrival_rate(const std::vector<double>& rates) const {
					double total = 0.0;
					for (size_t n = 0; n < waiting.size(); ++n)
						if ((int64_t)waiting[n].size() < max_queue_depth)
							total += rates[n];
					return total;
				}

				void update_total_arrival_rate(const std::vector<double>& rates) {
					total_arrival_rate = get_total_arrival_rate(rates);
				}

				void update_total_tick_rate(double tick_rate) {
					total_tick_rate = compute_total_tick_rate(tick_rate);
				}

				double compute_total_tick_rate(double tick_rate) const {
					return tick_rate;
				}

				double get_max_tick_rate(const int64_t n_jobs, const double tick_rate) const {
					return n_jobs * tick_rate;
				}

				double get_max_arrival_rate(const std::vector<double>& rates) const {
					double total = 0.0;
					for (const auto& r : rates) total += r;
					return total;
				}

				// ---- Koole geometric sampler (unchanged logic) ----
				inline int sample_next_fil_after_completion(
					int i,
					double lambda,   // arrival rate for this job type
					double gamma,    // tick rate
					double uniform_draw
				) {
					if (i <= 0) return 0;
					const double denom = lambda + gamma;
					if (denom <= 0.0) return 0;
					const double alpha = lambda / denom;
					const double beta  = gamma  / denom;
					if (alpha <= 0.0) return 0;
					if (beta  <= 0.0) return i;
					double U = uniform_draw;
					if (U <= 0.0) U = std::numeric_limits<double>::min();
					if (U >= 1.0) U = std::nextafter(1.0, 0.0);
					const int H = static_cast<int>(std::floor(std::log(U) / std::log(beta)));
					if (H >= i) return -1;
					return i - H;
				}

				// ---- Serialization ----
				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vg;
					vg.Add("max_queue_depth", max_queue_depth);
					const int64_t n_types = (int64_t)waiting.size();
					vg.Add("n_types", n_types);
					for (int64_t n = 0; n < n_types; ++n)
						vg.Add("waiting_" + std::to_string(n),
						       std::vector<int64_t>(waiting[(size_t)n].begin(), waiting[(size_t)n].end()));
					vg.Add("total_tick_rate",    total_tick_rate);
					vg.Add("total_arrival_rate", total_arrival_rate);
					vg.Add("arrival_rates",      arrival_rates);
					return vg;
				}

				explicit multi_queue(const DynaPlex::VarGroup& vg) {
					vg.Get("max_queue_depth", max_queue_depth);
					int64_t n_types = 0;
					vg.Get("n_types", n_types);
					waiting.resize((size_t)n_types);
					for (int64_t n = 0; n < n_types; ++n) {
						std::vector<int64_t> q;
						vg.Get("waiting_" + std::to_string(n), q);
						waiting[(size_t)n] = std::deque<int64_t>(q.begin(), q.end());
					}
					vg.Get("total_tick_rate",    total_tick_rate);
					vg.Get("total_arrival_rate", total_arrival_rate);
					if (vg.HasKey("arrival_rates"))
						vg.Get("arrival_rates", arrival_rates);
				}
			};

			struct State {
				//using this is recommended:
				
				// vector of first in line waiting times jobs length n jobs
				//std::vector<int64_t> FIL_waiting;
				
				//servers servers();
				
				//initialize ServerDynamicState
				ServerDynamicState server_manager;
				multi_queue queue_manager;

				int64_t next_fil_job_type = -1;  // which queue needs refresh

				
				std::string last_event_category;

				DynaPlex::StateCategory cat;
				DynaPlex::VarGroup ToVarGroup() const;


			};
			//Event may also be struct or class like.
			struct Event_type {
				enum class Type { Arrival, Tick, JobCompletion, Nothing };
				Type type;

				int64_t arrival_index;   // Used only for arrival
				int64_t tick_index;      // Used only for tick
				int64_t server_index;    // Used only for job completion
				int64_t job_type;        // Used only for job completion

				// ----- Factory Methods -----

				static Event_type MakeArrival(int64_t job) {
					return Event_type{
						Type::Arrival,
						job,      // arrival_index
						-1,       // tick_index
						-1,       // server_index
						-1        // job_type
					};
				}

				static Event_type MakeTick() {
					return Event_type{
						Type::Tick,
						-1,       // arrival_index
						0,        // tick_index (if needed)
						-1,       // server_index
						-1
					};
				}

				static Event_type MakeCompletion(int64_t k, int64_t job) {
					return Event_type{
						Type::JobCompletion,
						-1,    // arrival_index
						-1,    // tick_index
						k,     // server_index
						job    // job_type
					};
				}

				static Event_type MakeNothing() {
					return Event_type{
						Type::Nothing,
						-1, -1, -1, -1
					};
				}
			};
			
			struct Event {
				double event_sample;
				double uniform_rate_next_fil;
			};

			struct nextStateProbability {
				MDP::State next_state;
				double probability;
			};





			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event&) const;



			Event_type GetEventType(const double event_sample, const State&) const;
			Event GetEvent(DynaPlex::RNG& rng) const;
			std::vector<std::tuple<Event,double>> EventProbabilities() const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State& state, int64_t action) const;			
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);
			
			
			static std::vector<std::pair<int64_t, double>> NextFILDistribution(int64_t i, double lambda, double gamma);
			std::vector<nextStateProbability> getNextStateProbability(const MDP::State& state, int64_t action) const;
			double GetImmediateCost(const State& state) const;
			double ComputeTickCost(const State& state) const;

			struct RVISolution {
				double g_star;  // optimal average cost per time unit
				int M;          // truncation level used
				std::unordered_map<uint64_t, int64_t> action_map;  // encoded state key -> optimal action
			};
			RVISolution runRVI(int M, int max_iter = 10000, bool silent = false) const;  // solve at fixed M
			RVISolution runRVI(double rel_tol = 1e-4, bool silent = false) const;       // auto-select M via heuristic + convergence check
			int64_t EvaluateRVIPolicy(const RVISolution& sol, const State& state) const;

			// ----------------------------------------------------------------
			// Continuous-time event-driven simulator
			// ----------------------------------------------------------------
			// Simulates the exact continuous-time system (exponential arrivals
			// and services, exact sojourn times) without the uniformised chain.
			// The policy is queried at every assignment decision point by
			// converting exact sojourn times tau_n to tick counts
			// f_n = floor(tau_n * tick_rate) and constructing a canonical
			// AwaitAction state.
			//
			// Returns:
			//   mean_cost_per_time  - time-averaged cost rate  (g*_time = g* x E[Lambda])
			//   mean_cost_per_event - cost per arrival/completion event
			//                         (comparable to the policy-comparer output)
			//   std_err_per_time    - standard error of mean_cost_per_time
			//   avg_events          - average events per trajectory
			//   avg_decisions       - average assignment decisions per trajectory
			// ----------------------------------------------------------------
			struct ContinuousSimResult {
				double mean_cost_per_time;
				double mean_cost_per_event;
				double std_err_per_time;
				long long avg_events;
				long long avg_decisions;
			};

			ContinuousSimResult SimulateContinuous(
				std::function<int64_t(const State&)> policy_fn,
				int    n_traj   = 100,
				double t_max    = 500000.0,
				double t_warmup =  50000.0) const;

			// ----------------------------------------------------------------
			// TraceContinuous
			// Runs one short trajectory of the continuous-time simulator and
			// prints a human-readable event log to std::cout.  Useful for
			// visually verifying that the simulator behaves as expected.
			//
			// Each line shows: physical time, event kind (ARRIVAL / COMPLETION /
			// TICK), updated queue sojourn times, server occupancy, and the
			// sequence of assignment decisions made by the policy.
			//
			// Parameters:
			//   policy_fn - raw lambda (same type as SimulateContinuous)
			//   t_trace   - print events in [0, t_trace]; default 20.0
			//   rng_seed  - RNG seed; default 42
			// ----------------------------------------------------------------
			void TraceContinuous(
				std::function<int64_t(const State&)> policy_fn,
				double  t_trace  = 20.0,
				int64_t rng_seed = 42) const;
		};

		/**
		 * Evaluates any DynaPlex::Policy by simulating the MDP for a fixed number of
		 * *uniformized steps* (every call to IncorporateEvent counts as one step,
		 * across all event streams).  The returned average cost per step is directly
		 * comparable to g_star from runRVI(), which is also expressed per uniformized
		 * step.  Works with FIFO, RVI_optimal, NN policies, or any other Policy object.
		 *
		 * @param mdp               DynaPlex::MDP adapter (from dp.GetMDP(config))
		 * @param policy            Any DynaPlex::Policy
		 * @param n_trajectories    Independent simulation runs          (default: 500)
		 * @param steps_per_traj    Uniformized steps in main phase      (default: 200000)
		 * @param warmup_steps      Uniformized steps discarded at start (default: 20000)
		 * @param rng_seed          Base seed; trajectory i gets offset i (default: 42)
		 * @return VarGroup with keys "mean", "std_error", "n_trajectories",
		 *                           "steps_per_trajectory", "warmup_steps"
		 */
		DynaPlex::VarGroup EvaluatePolicyPerStep(
			const DynaPlex::MDP&    mdp,
			const DynaPlex::Policy& policy,
			int64_t n_trajectories = 500,
			int64_t steps_per_traj = 200000,
			int64_t warmup_steps   = 20000,
			int64_t rng_seed       = 42);

		/**
		 * Raw policy evaluator that works directly on the concrete queue_mdp types,
		 * giving exact step-type counts that match the RVI's denominator.
		 *
		 * Counts three kinds of steps:
		 *   action_steps       - every IncorporateAction call (action=0 or action=1)
		 *   real_event_steps   - IncorporateEvent calls that are NOT a FIL refresh
		 *                        (tick, arrival, completion, nothing self-loop)
		 *   fil_refresh_steps  - IncorporateEvent calls that ARE a FIL refresh
		 *                        (triggered by action=1, no cost, no real event)
		 *
		 * The RVI denominator = action_steps + real_event_steps  (fil_refresh excluded)
		 * The comparer denominator = real_event_steps only (periods)
		 *
		 * Returns mean cost per RVI-equivalent step, plus breakdown for diagnosis.
		 */
		struct RawEvalResult {
			double  mean_cost_per_rvi_step;     // cost / (action_steps + real_event_steps)  [tick-event cost]
			double  mean_cost_per_rvi_step_rvi; // cost / (action_steps + real_event_steps)  [RVI-style: per-step at FIL>due_time]
			double  mean_cost_per_event;        // cost / real_event_steps  (matches comparer)
			double  mean_cost_per_step_gic;     // GetImmediateCost (non-FIL-refresh only) / rvi_steps -> matches g* from runRVI()
			double  std_error;                  // std error of mean_cost_per_rvi_step
			int64_t total_action_steps;
			int64_t total_real_event_steps;
			int64_t total_fil_refresh_steps;
		};

		RawEvalResult EvaluatePolicyRaw(
			const MDP&                                  mdp,
			std::function<int64_t(const MDP::State&)>   get_action,
			int64_t n_trajectories = 200,
			int64_t steps_per_traj = 100000,
			int64_t warmup_steps   = 10000,
			int64_t rng_seed       = 42);

		/// Convenience overload: wraps a DynaPlex::Policy into the std::function form above.
		/// Requires mdp.int_hash to be set (done automatically in MDP::MDP(VarGroup)).
		RawEvalResult EvaluatePolicyRaw(
			const MDP&                  mdp,
			const DynaPlex::Policy&     policy,
			int64_t n_trajectories = 200,
			int64_t steps_per_traj = 100000,
			int64_t warmup_steps   = 10000,
			int64_t rng_seed       = 42);

		/**
		 * Prints a console heatmap of a policy's job-type assignment decisions.
		 * X-axis: FIL_waiting[0], Y-axis: FIL_waiting[1].
		 * Cell: 0=serve type 0, 1=serve type 1, .=skip/idle, -=not visited.
		 * Samples canonical states: action_counter==0, both FIL>=0, exactly 1 server busy.
		 * Works with any DynaPlex::Policy (RVI, FIFO, NN, etc.).
		 */
		void PrintPolicyHeatmap(
			const DynaPlex::MDP&    fw_mdp,
			const DynaPlex::Policy& policy,
			int     max_fil   = 15,
			int64_t n_warmup  = 100000,
			int64_t n_samples = 2000000);

		/**
		 * Enumeration-based heatmap: directly constructs canonical AwaitAction states
		 * for every (FIL_0, FIL_1) cell and queries the policy without simulation.
		 * Eliminates '-' (not-visited) artifacts of the simulation-based version.
		 *
		 * Canonical state: pool 0 busy on type 0 (1 server busy), pool 1 idle,
		 * both job types waiting (FIL_0=f0, FIL_1=f1), action_counter=0.
		 *
		 * @param mdp        Concrete queue_mdp::MDP instance (not type-erased)
		 * @param policy_fn  Returns action (0=skip, 1=assign top) for a given State
		 * @param max_fil    Grid spans [0, max_fil] x [0, max_fil]
		 */
		void PrintEnumeratedHeatmap(
			const MDP& mdp,
			std::function<int64_t(const MDP::State&)> policy_fn,
			int max_fil = 15);

	}  // namespace queue_mdp
}  // namespace DynaPlex::Models

