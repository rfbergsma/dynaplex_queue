#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include <deque>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace DynaPlex::Models {
	namespace queue_mdp /*must be consistent everywhere for complete mdp definition and associated policies and states (if not defined inline).*/
	{		
		class MDP
		{			
		public:	
			double discount_factor;
			
			struct ServerStaticInfo {
				int64_t servers = 0;
				double mu_k = 0.0;
				std::vector<int64_t> can_serve; // types of jobs it can serve
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
							r += dyn.busy_on[k][j] * S[k].mu_k;
					return r;
				}

				inline bool assign_job(ServerDynamicState& dyn,
					const std::vector<ServerStaticInfo>& S,
					int64_t k, int64_t job) {
					int idx = canServeIndex(S, k, job);
					if (idx < 0) return false;
					if (dyn.busy_on[(size_t)k][(size_t)idx] >= S[(size_t)k].servers) return false;
					dyn.busy_on[(size_t)k][(size_t)idx] += 1;
					total_service_rate += S[(size_t)k].mu_k;
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

			struct multi_queue {
				// vector of first in line waiting times jobs length n jobs, -1 if empty
				std::vector<int64_t> FIL_waiting;
				std::vector <double> arrival_rates;
				double total_tick_rate;
				double total_arrival_rate;
				

				multi_queue() = default; 

				void initialize(int64_t n_jobs, double tick_rate, std::vector<double> arrival_rates) {
					FIL_waiting.resize((size_t)n_jobs, -1);
					total_tick_rate = tick_rate;
					total_arrival_rate = 0.0;
					for (const auto& rate : arrival_rates) {
						total_arrival_rate += rate;
					}
					this->arrival_rates = std::move(arrival_rates);
				}

				//arrival of new job (set FIL to 0 first, only allow arrivals if FIL<0)
				void arrival(int64_t n) {
					if (n < 0 || n >= static_cast<int64_t>(FIL_waiting.size())) throw("job cannot arrive, queu is not empty");
					if (FIL_waiting[(size_t)n] < 0) {
						FIL_waiting[(size_t)n] = 0;
						total_arrival_rate -= arrival_rates[(size_t)n];
					}
				}
				
				//tick event (increment FIL for all waiting jobs)
				void tick() {
					for (size_t n = 0; n < FIL_waiting.size(); ++n) {
						if (FIL_waiting[n] >= 0) {
							FIL_waiting[n] += 1;
						}
					}
				}

				//complete job n (set FIL to -1)
				void complete_job(int64_t n, double uniform_draw) {
					
					if (n < 0 || n >= static_cast<int64_t>(FIL_waiting.size())) throw("job cannot complete, invalid job type");
					
					int64_t current_fil = FIL_waiting[(size_t)n];
					
					if (current_fil == -1) throw("complete_job/pop called but queue is empty");

					if (current_fil == -1) {
						std::cout << "trying to complete job that has fil -1 " << std::endl;
					}

					else if (current_fil == 0) {
						FIL_waiting[(size_t)n] = -1;
						
						

					}
					else if (current_fil > 0) {
						FIL_waiting[(size_t)n] = sample_next_fil_after_completion(current_fil,
							arrival_rates[(size_t)n],
							total_tick_rate,
							uniform_draw);
						
						
					}

					int64_t next_fil = FIL_waiting[(size_t)n];
					if (next_fil == -1) {
						total_arrival_rate += arrival_rates[(size_t)n];
					}
					
				}

				std::vector<int64_t> get_FIL_waiting() const {
					return FIL_waiting;
				}

				// update total arrival rate
				void update_total_arrival_rate(const std::vector<double>& arrival_rates) {
					total_arrival_rate = get_total_arrival_rate(arrival_rates);
				}
				
				// update total tick rate
				void update_total_tick_rate(double tick_rate) {
					total_tick_rate = compute_total_tick_rate(tick_rate);
				}
				
				// function that calculates total tick rate
				double compute_total_tick_rate(double tick_rate) const {
					
					return tick_rate;
				}

				double get_total_arrival_rate(const std::vector<double>& arrival_rates) const {
					double total_rate = 0.0;
					for (size_t n = 0; n < FIL_waiting.size(); ++n) {
						if (FIL_waiting[n] < 0) {
							total_rate += arrival_rates[n];
						}
					}
					return total_rate;
				}
				
				//get maximum possible tick rate
				double get_max_tick_rate(const int64_t n_jobs,const double tick_rate) const {
					return n_jobs * tick_rate;
				}

				//get maximum possible arrival rate
				double get_max_arrival_rate(const std::vector<double>& arrival_rates) const {
					double total_rate = 0.0;
					for (const auto& rate : arrival_rates) {
						total_rate += rate;
					}
					return total_rate;
				}

				inline int sample_next_fil_after_completion(
					int i,
					double lambda, //arrival rate
					double gamma, //tick rate
					double uniform_draw
				) {
					if (i <= 0) return 0;

					const double denom = lambda + gamma;
					if (denom <= 0.0) return 0;

					// correct Koole probabilities
					double alpha = lambda / denom;   // arrival probability
					double beta = gamma / denom;   // tick probability

					if (alpha <= 0.0) return 0;      // no arrivals possible
					if (beta <= 0.0) return i;      // no ticks => all arrivals at same time => new FIL = i

					// Sample geometric H: #ticks before arrival
					double U = uniform_draw;
					if (U <= 0.0) U = std::numeric_limits<double>::min();
					if (U >= 1.0) U = std::nextafter(1.0, 0.0);

					// geometric: P(H = h) = beta^h * alpha
					int H = static_cast<int>(std::floor(std::log(U) / std::log(beta)));

					if (H >= i) return -1;  // queue empty
					return i - H;
				}

				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vg;
					vg.Add("FIL_waiting", FIL_waiting);
					vg.Add("total_tick_rate", total_tick_rate);
					vg.Add("total_arrival_rate", total_arrival_rate);
					// optional: store arrival_rates if you treat it as part of state (usually not needed)
					return vg;
				}

				explicit multi_queue(const DynaPlex::VarGroup& vg) {
					vg.Get("FIL_waiting", FIL_waiting);
					vg.Get("total_tick_rate", total_tick_rate);
					vg.Get("total_arrival_rate", total_arrival_rate);
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

			struct RVISolution {
				double g_star;  // optimal average cost per time unit
				int M;          // truncation level used
			};
			RVISolution runRVI(int M) const;  // solve at fixed M, verbose output
			RVISolution runRVI() const;       // auto-select M via heuristic + convergence check
		};
	}
}

