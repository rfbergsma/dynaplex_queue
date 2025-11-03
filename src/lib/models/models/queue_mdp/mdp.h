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
				int64_t servers;
				double mu_k;
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

				// Update idle capacity info based on busy_on (passed in)
				void update_idle_capacity(const std::vector<std::vector<int64_t>>& busy_on) {
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

				double get_total_service_rate(const ServerDynamicState& dyn, const std::vector<ServerStaticInfo>& static_info) {
					double total_rate = 0.0;
					for (size_t k = 0; k < static_info.size(); ++k) {
						const auto& st = static_info[k];
						for (size_t j = 0; j < st.can_serve.size(); ++j) {
							// busy_on[k][j] is the number of servers of type k busy on job st.can_serve[j]
							total_rate += dyn.busy_on[k][j] * st.mu_k;
						}
					}
					return total_rate;
				}

				void assign_job(ServerDynamicState& dyn, const std::vector<ServerStaticInfo>& static_info, int64_t k, int64_t job_type) {
					auto it = std::find(static_info[k].can_serve.begin(), static_info[k].can_serve.end(), job_type);
					if (it != static_info[k].can_serve.end()) {
						size_t idx = std::distance(static_info[k].can_serve.begin(), it);
						dyn.busy_on[k][idx]++;
					}
				}
			};

			std::vector<ServerStaticInfo> server_static_info;

			
			int64_t n_jobs; // number of different types of jobs
			int64_t k_servers; // number of servers
			std::vector<double> arrival_rates; // arrival rates for each job type n
			double tick_rate = 1.0; // tick rate

			struct multi_queue {
				// vector of first in line waiting times jobs length n jobs, -1 if empty
				std::vector<int64_t> FIL_waiting;


				// function that calculates total tick rate
				double total_tick_rate(const double tick_rate) const {
					double total_rate = 0.0;
					for (size_t n = 0; n < FIL_waiting.size(); ++n) {
						if (FIL_waiting[n] >= 0) {
							total_rate += tick_rate;
						}
					}
					return total_rate;
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
			};


			struct State {
				//using this is recommended:
				
				// vector of first in line waiting times jobs length n jobs
				//std::vector<int64_t> FIL_waiting;
				
				//servers servers();
				
				//initialize ServerDynamicState
				ServerDynamicState server_manager;

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
		};
	}
}

