#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <deque>


namespace DynaPlex::Models {
	namespace queue_mdp /*keep this in line with id below and with namespace name in header*/
	{
		// helper for using config file
		VarGroup GetSubGroup(const VarGroup& group, const std::string& key) {
			VarGroup subgroup;
			group.Get(key, subgroup); // Assuming VarGroup has a Get method to retrieve subgroups
			return subgroup;
		}

		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			//Needs to update later:
			vars.Add("valid_actions", 1);
			vars.Add("discount_factor", discount_factor);

			return vars;
		}


		double MDP::ModifyStateWithEvent(State& state, const Event& event) const
		{
			if (event.type == Event::Type::JobCompletion) {
				//complete job at server
				state.server_manager.complete_job(event.server_index, event.job_type);
				state.queue_manager.complete_job(event.job_type); // job completed, remove from queue
				state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
				state.cat = StateCategory::AwaitAction();
				return 0.0; // no cost for job completion
			}
			else if (event.type == Event::Type::Arrival) {
				state.queue_manager.arrival(event.arrival_index);
				state.cat = StateCategory::AwaitAction();
				state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
				return 0.0; // cost for arrival
			}
			else if (event.type == Event::Type::Tick) {
				state.queue_manager.tick();
				state.cat = StateCategory::AwaitAction();
				// add cost rate for fils >  due time
				double cost = 0.0;
				for (size_t n = 0; n < state.queue_manager.FIL_waiting.size(); ++n) {
					if (state.queue_manager.FIL_waiting[n] >= 0) {
						if (state.queue_manager.FIL_waiting[n] > due_times[n]) {
							// add cost
							// (could be more sophisticated, e.g., linear in waiting time - due time)
							cost += cost_rates[n];
						}
					}
				}
				return cost; // cost for tick
			}
			else {
				// No event
				return 0.0;
			}
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{
			
			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());


			state.server_manager.take_action(action);

			if (state.server_manager.get_action_counter() >= static_cast<int64_t>(state.server_manager.action_queue.size())) {
				// all actions processed
				state.server_manager.set_action_counter(0);
				state.cat = StateCategory::AwaitEvent();
			}
			
			//implement change to state. 
			// do not forget to update state.cat. 
			//returns costs. 
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			//add any other state variables. 
			return vars;
		}

		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			State state{};		
			vars.Get("cat", state.cat);
			//initiate any other state variables. 
			return state;
		}

		MDP::State MDP::GetInitialState() const
		{			
			State state{};
			state.cat = StateCategory::AwaitEvent();//or AwaitAction(), depending on logic
			
			state.server_manager.initialize(&server_static_info,n_jobs);			
			state.queue_manager.initialize(n_jobs,tick_rate, arrival_rates);

			return state;
		}

		MDP::MDP(const VarGroup& config)
		{
			//In principle, state variables should be initiated as follows:
			//config.Get("name_of_variable",name_of_variable); 
			
			//we may also have config arguments that are not mandatory, and the internal value takes on 
			// a default value if not provided. Use sparingly. 
			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;
		
			config.Get("k_servers", k_servers);
			config.Get("n_jobs", n_jobs);
			config.Get("arrival_rates", arrival_rates);
			config.Get("tick_rate", tick_rate);
			config.Get("cost_rates", cost_rates);
			config.Get("due_times", due_times);

			//initialize server manager
			server_static_info.clear();
			server_static_info.resize((size_t)k_servers);

			for (int64_t i = 0; i < k_servers; ++i) {
				VarGroup serverConfig = GetSubGroup(config, "server_type_" + std::to_string(i));
				serverConfig.Get("servers", server_static_info[i].servers);
				serverConfig.Get("service_rate", server_static_info[i].mu_k);
				serverConfig.Get("can_serve", server_static_info[i].can_serve);
			}

			uniformization_rate = tick_rate;
			// + sum of arrival rates
			for (const auto& rate : arrival_rates) {
				uniformization_rate += rate;
			}
			// sum of all service rates
			for (int64_t i = 0; i < k_servers; ++i) {
				uniformization_rate += server_static_info[i].mu_k;
			}

		}


		MDP::Event MDP::GetEvent(RNG& rng, const State& state) const {
			double event_sample = rng.genUniform() * uniformization_rate;
			//consume random number to avoid unused parameter warning
			
			//std::cout << "uniformization rate " << uniformization_rate << "\n";
			//std::cout << "tota1 arrival rate: " << state.queue_manager.total_arrival_rate << "\n";
			//std::cout << "total service rate: " << state.server_manager.total_service_rate << "\n";
			//std::cout << "total tick rate: " << state.queue_manager.total_tick_rate << "\n";

			if (event_sample < state.queue_manager.total_arrival_rate) {
				// Arrival event
				double cumulative_rate = 0.0;
				for (int64_t n = 0; n < n_jobs; ++n) {
					if (state.queue_manager.FIL_waiting[n] == -1) {
						cumulative_rate += arrival_rates[(size_t)n];

						if (event_sample < cumulative_rate) {
							return Event{ Event::Type::Arrival, n };
						}
					}
				}
			}
			else if (event_sample < state.queue_manager.total_arrival_rate + state.queue_manager.total_tick_rate) {
				// Tick event
				return Event{ Event::Type::Tick, -1 };
			}
			else {
				// Job completion event
				double cumulative_rate = state.queue_manager.total_arrival_rate + state.queue_manager.total_tick_rate;
				for (int64_t k = 0; k < k_servers; ++k) {
					for (size_t j = 0; j < server_static_info[(size_t)k].can_serve.size(); ++j) {
						cumulative_rate += state.server_manager.busy_on[(size_t)k][j] * server_static_info[(size_t)k].mu_k;
						if (event_sample < cumulative_rate) {
							return Event{ Event::Type::JobCompletion, static_cast<int64_t>(server_static_info[k].can_serve[j])};
						}
					}
				}
			}
			return Event{ Event::Type::Nothing, -1 }; // Should not reach here
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const
		{
			//This is optional to implement. You only need to implement it if you intend to solve versions of your problem
			//using exact methods that need access to the exact event probabilities.
			//Note that this is typically only feasible if the state space if finite and not too big, i.e. at most a few million states.
			throw DynaPlex::NotImplementedError();
		}


		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {
			
			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());
			//t64_t server_index = current_action.server_index;	
			features.Add(current_action.server_index);
			features.Add(current_action.job_type);
			features.Add(state.queue_manager.FIL_waiting);

			
			//Features.Add(state.);
			
			throw DynaPlex::NotImplementedError();
		}
		
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{//Here, we register any custom heuristics we want to provide for this MDP.	
		 //On the generic DynaPlex::MDP constructed from this, these heuristics can be obtained
		 //in generic form using mdp->GetPolicy(VarGroup vars), with the id in var set
		 //to the corresponding id given below.
			registry.Register<FIFOPolicy>("FIFO policy",
				"First in first out policy, always assigns a job to a server.");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			//this typically works, but state.cat must be kept up-to-date when modifying states. 
			return state.cat;
		}	

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());
			if (action == 1) {
				// assign job
				return state.server_manager.can_assign_job(current_action.server_index, current_action.job_type);
			}
			else if (action == 0) {
				// do not assign job
				return true;
			}
			else {
				return false; // invalid action
			}
		}


		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel("queue_mdp", "mdp for analyzing queueing models", registry);
			//To use this MDP with dynaplex, register it like so, setting name equal to namespace and directory name
			// and adding appropriate description. 
			//DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel(
			//	"<id of mdp goes here, and should match namespace name and directory name>",
			//	"<description goes here>",
			//	registry); 
		}
	}
}

