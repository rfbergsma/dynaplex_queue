#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"
#include <deque>



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
			};

			struct ServerDynamicState{
				// For each job type, number of idle servers that can serve it
				std::vector<int64_t> has_idle_capacity_per_job;
				// True if any job type has idle servers
				bool has_idle_capacity = false;
				// Reference to static info (not owned)
				const std::vector<ServerStaticInfo>* static_info = nullptr;
				std::vector<std::vector<int64_t>> busy_on;
				
				std::deque<Action> action_queue;
				int64_t action_counter;

				double total_service_rate;

				ServerDynamicState() = default;
				
				void initialize(const std::vector<ServerStaticInfo>* static_info_ptr, int64_t n_jobs) {
					static_info = static_info_ptr;
					busy_on.clear();
					if (!static_info) return;
					busy_on.resize(static_info->size());
					for (size_t k = 0; k < static_info->size(); ++k) {
						busy_on[k].resize((*static_info)[k].can_serve.size(), 0);
					}
					has_idle_capacity_per_job.assign(n_jobs, 0);
					has_idle_capacity = false;
					total_service_rate = 0.0;

					//initialize empty queue
					action_queue.clear();
					action_counter = 0;

					// Optionally, call update_idle_capacity() and update_total_service_rate() here
					update_idle_capacity();
				}
				
				void generate_actions(std::vector<int64_t> FIL_waiting) {
					// generate possible actions based on current busy_on and FIL_waiting
					// if FIL_waiting[n] >=0 and there is idle server for job n, add action
					action_queue.clear();
					
					for (size_t n = 0; n < FIL_waiting.size(); ++n) {
						if (FIL_waiting[n] >= 0) { // job n is waiting
							for (size_t k = 0; k < busy_on.size(); ++k) {
								int idx = canServeIndex(*static_info, static_cast<int64_t>(k), static_cast<int64_t>(n));
								
								if (idx >= 0) { // server k can serve job n
									if (busy_on[k][(size_t)idx] < (*static_info)[k].servers) {
										// there is idle server for job n
										action_queue.push_back(Action{ static_cast<int64_t>(k), static_cast<int64_t>(n) });
									}
								}

							}
						}
					}
				}


				// function takes action and modifies the server dynamic state accordingly
				// increases busy_on for the corresponding server and job type
				// removes actions from the queue that can no longer be taken
				// Reduces the action counter by number of jobs taken out of queue
				void take_action(Action taken_action) {
					assign_job(taken_action.server_index, taken_action.job_type);
					int64_t current_queue_length = action_queue.size();
					// remove actions that can no longer be taken
					action_queue.erase(std::remove_if(action_queue.begin(), action_queue.end(),
						[this](const Action& action) {
							int idx = canServeIndex(*static_info, action.server_index, action.job_type);
							if (idx < 0) return true;
							return busy_on[(size_t)action.server_index][(size_t)idx] >= (*static_info)[(size_t)action.server_index].servers;
						}), action_queue.end());
					int64_t new_queue_length = action_queue.size();
					action_counter = action_counter - (current_queue_length - new_queue_length);
				}


				void set_action_counter(int64_t count) {
					action_counter = count;
				}

				int64_t get_action_counter() const {
					return static_cast<int64_t>(action_queue.size());
				}

				void assign_job(int64_t k, int64_t job) {
					int idx = canServeIndex(*static_info, k, job);
					if (idx < 0) return;
					if (busy_on[(size_t)k][(size_t)idx] >= (*static_info)[(size_t)k].servers) return;
					busy_on[(size_t)k][(size_t)idx] += 1;

				}



				//complete job n at server k remove job from busy_on
				void complete_job(int64_t k, int64_t job) {
					int idx = canServeIndex(*static_info, k, job);
					if (idx < 0) return;
					if (busy_on[(size_t)k][(size_t)idx] <= 0) return;
					busy_on[(size_t)k][(size_t)idx] -= 1;
				}
				

				//returns the index of job in can_serve vector of server k, -1 if cannot serve
				static int canServeIndex(const std::vector<ServerStaticInfo>& S, int64_t k, int64_t job) {
					const auto& v = S[(size_t)k].can_serve;
					auto it = std::find(v.begin(), v.end(), job);
					return (it == v.end()) ? -1 : (int)std::distance(v.begin(), it);
				}

				// Update idle capacity info based on busy_on (passed in)
				void update_idle_capacity() {
					if (!static_info || static_info->empty() || busy_on.empty()) {
						has_idle_capacity_per_job.clear();
						has_idle_capacity = false;
						return;
					}
					size_t n_jobs = busy_on[0].size();
					has_idle_capacity_per_job.assign(n_jobs, 0);
					has_idle_capacity = false;
					for (size_t n = 0; n < n_jobs; ++n) {
						for (size_t k = 0; k < static_info->size(); ++k) {
							const auto& st = (*static_info)[k];
							if (std::find(st.can_serve.begin(), st.can_serve.end(), n) != st.can_serve.end()) {
								int64_t idle = st.servers - busy_on[k][n];
								if (idle > 0) {
									has_idle_capacity_per_job[n] += idle;
									has_idle_capacity = true;
								}
							}
						}
					}
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
					vars.Add("has_idle_capacity_per_job", has_idle_capacity_per_job);
					vars.Add("has_idle_capacity", has_idle_capacity);
					//vars.Add("busy_on", busy_on);
					for (size_t k = 0; k < busy_on.size(); ++k) {
						vars.Add("busy_on_" + std::to_string(k), busy_on[k]);
					}
					//vars.Add("busy_on_size", static_cast<int64_t>(busy_on.size()));
					vars.Add("total_service_rate", total_service_rate);
					
					return vars;
				}
				explicit ServerDynamicState(const DynaPlex::VarGroup& vg) {
					vg.Get("has_idle_capacity_per_job", has_idle_capacity_per_job);
					vg.Get("has_idle_capacity", has_idle_capacity);
					//vg.Get("busy_on", busy_on);
					busy_on.clear();
					size_t k = 0;
					while (true) {
						std::vector<int64_t> row;
						std::string key = "busy_on_" + std::to_string(k);
						if (!vg.HasKey(key)) break; // Stop if the key doesn't exist
						vg.Get(key, row);
						busy_on.push_back(std::move(row));
						++k;
					}
					
					vg.Get("total_service_rate", total_service_rate);
				}
			};

			std::vector<ServerStaticInfo> server_static_info;

			
			int64_t n_jobs; // number of different types of jobs
			int64_t k_servers; // number of servers
			std::vector<double> arrival_rates; // arrival rates for each job type n
			double tick_rate; // tick rate
			double uniformization_rate;

			struct multi_queue {
				// vector of first in line waiting times jobs length n jobs, -1 if empty
				std::vector<int64_t> FIL_waiting;
				std::vector < double> arrival_rates;
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
				void complete_job(int64_t n) {
					if (n < 0 || n >= static_cast<int64_t>(FIL_waiting.size())) throw("job cannot complete, invalid job type");
					if (FIL_waiting[(size_t)n] >= 0) {
						FIL_waiting[(size_t)n] = -1;
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
					double total = 0.0;
					for (auto w : FIL_waiting) if (w >= 0) total += tick_rate;
					return total;
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
					int i,              // current FIL (ticks)
					double lambda,      // arrival rate
					double gamma,       // tick rate
					DynaPlex::RNG& rng  // your RNG with Uniform01()
				) {
					if (i <= 0) return 0;

					const double denom = lambda + gamma;
					if (denom <= 0.0) return 0;        // no events at all => stays empty
					const double q = gamma / denom;    // in (0,1) unless lambda=0 or gamma=0
					const double p = 1.0 - q;

					// Edge cases
					if (p <= 0.0) return 0;            // lambda=0 -> no arrivals -> empties
					if (q <= 0.0) return i;            // gamma=0 -> arrival immediately -> J=i

					// Inverse-CDF geometric draw: H = floor( ln(U)/ln(q) )
					double U = rng.genUniform();
					// Guard U from hitting 0
					if (U <= 0.0) U = std::numeric_limits<double>::min();
					if (U >= 1.0) U = std::nextafter(1.0, 0.0);

					const double lnq = std::log(q); // negative
					int H = static_cast<int>(std::floor(std::log(U) / lnq));

					// Cap behavior: if first arrival happens after >= i ticks, queue empties
					if (H >= i) return 0;
					return i - H;
				}
				

				DynaPlex::VarGroup ToVarGroup() const {
					DynaPlex::VarGroup vars;
					vars.Add("FIL_waiting", FIL_waiting);
					vars.Add("total_tick_rate", total_tick_rate);
					vars.Add("total_arrival_rate", total_arrival_rate);

					return vars;
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

				DynaPlex::StateCategory cat;
				DynaPlex::VarGroup ToVarGroup() const;


			};
			//Event may also be struct or class like.
			struct Event {
				enum class Type { Arrival, Tick, JobCompletion, Nothing };
				Type type;
				int64_t arrival_index;    // For arrivals
				int64_t tick_index;       // For ticks
				int64_t server_index; // For completions
				int64_t job_type;        // For completions
			};

			double ModifyStateWithAction(State&, int64_t action) const;
			double ModifyStateWithEvent(State&, const Event& ) const;
			Event GetEvent(DynaPlex::RNG& rng, const State&) const;
			std::vector<std::tuple<Event,double>> EventProbabilities() const;
			DynaPlex::VarGroup GetStaticInfo() const;
			DynaPlex::StateCategory GetStateCategory(const State&) const;
			bool IsAllowedAction(const State& state, int64_t action) const;			
			State GetInitialState() const;
			State GetState(const VarGroup&) const;
			void RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>&) const;
			void GetFeatures(const State&, DynaPlex::Features&) const;
			explicit MDP(const DynaPlex::VarGroup&);
		};
	}
}

