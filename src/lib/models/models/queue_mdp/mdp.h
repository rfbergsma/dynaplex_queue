#pragma once
#include "dynaplex/dynaplex_model_includes.h"
#include "dynaplex/modelling/discretedist.h"

namespace DynaPlex::Models {
	namespace queue_mdp /*must be consistent everywhere for complete mdp definition and associated policies and states (if not defined inline).*/
	{		
		class MDP
		{			
		public:	
			double discount_factor;
			//any other mdp variables go here:


			/*
			//define a struct Gk, this contains: the amound of servers of type k, the service rate mu_k, and the types of job it can serve. Also y_n, the number of servers busy on each job n
			struct server_type {
				int64_t servers;
				double mu_k;
				std::vector<int64_t> can_serve; // types of jobs it can serve
				std::vector<int64_t> busy_on; // number of servers busy on each job n
			};


			//create a servers class with a couple of functions to manage the servers
			class servers {
				std::vector<server_type> server_types;
				bool has_idle_capacity;
				std::vector<int64_t> idle_capacity_per_job; // now stores counts

			public:
				servers()
					: server_types(),
					has_idle_capacity(false),
					idle_capacity_per_job() {
				}

				servers(const std::vector<server_type>& server_types_in)
					: server_types(server_types_in),
					has_idle_capacity(false),
					idle_capacity_per_job(server_types_in.empty() ? 0 : server_types_in[0].busy_on.size(), 0) {
				}

				// For each job type n, count idle servers that can serve n
				void update_idle_capacity() {
					if (server_types.empty() || server_types[0].busy_on.empty()) {
						idle_capacity_per_job.clear();
						has_idle_capacity = false;
						return;
					}
					size_t n_jobs = server_types[0].busy_on.size();
					idle_capacity_per_job.assign(n_jobs, 0);
					has_idle_capacity = false;
					for (size_t n = 0; n < n_jobs; ++n) {
						for (const auto& st : server_types) {
							// Only count servers that can serve job n
							if (std::find(st.can_serve.begin(), st.can_serve.end(), n) != st.can_serve.end()) {
								int64_t idle = st.servers - st.busy_on[n];
								if (idle > 0) {
									idle_capacity_per_job[n] += idle;
									has_idle_capacity = true;
								}
							}
						}
					}
				}

				void assign_job(int64_t k, int64_t n) {
					server_types[k].busy_on[n]++;
				}
				void remove_job(int64_t k, int64_t n) {
					server_types[k].busy_on[n]--;
				}
			};


			std::vector<server_type> initial_server_types;
			
			*/
			
			
			struct ServerStaticInfo {
				int64_t servers = 0;
				double mu_k = 0.0;
				std::vector<int64_t> can_serve; // types of jobs it can serve
			};


			struct ServerDynamicState{
				// For each job type, number of idle servers that can serve it
				std::vector<int64_t> has_idle_capacity_per_job;
				// True if any job type has idle servers
				bool has_idle_capacity = false;

				// Reference to static info (not owned)
				const std::vector<ServerStaticInfo>* static_info = nullptr;
				
				std::vector<std::vector<int64_t>> busy_on;
				
				double total_service_rate;

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
					// Optionally, call update_idle_capacity() and update_total_service_rate() here
					update_idle_capacity();
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
					return true;
				}

				
			};

			std::vector<ServerStaticInfo> server_static_info;

			
			int64_t n_jobs; // number of different types of jobs
			int64_t k_servers; // number of servers
			std::vector<double> arrival_rates; // arrival rates for each job type n
			double tick_rate = 1.0; // tick rate
			double uniformization_rate;

			struct multi_queue {
				// vector of first in line waiting times jobs length n jobs, -1 if empty
				std::vector<int64_t> FIL_waiting;
				double total_tick_rate;
				double total_arrival_rate;

				void initialize(int64_t n_jobs) {
					FIL_waiting.resize((size_t)n_jobs, -1);
					total_tick_rate = 0.0;
					total_arrival_rate = 0.0;
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

