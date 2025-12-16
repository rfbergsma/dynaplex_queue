#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"
#include <deque>
#include <iostream>


namespace DynaPlex::Models {
	namespace queue_mdp /*keep this in line with id below and with namespace name in header*/
	{
		// ===== DEBUG UTILITIES FOR queue_mdp  =====
		#define QUEUE_MDP_DEBUG 0

		#if QUEUE_MDP_DEBUG

		static void DebugPrintFIL(const MDP::State& state, const char* prefix)
		{
			std::cout << prefix << " FIL_waiting = [";
			const auto& FIL = state.queue_manager.FIL_waiting;
			for (size_t i = 0; i < FIL.size(); ++i) {
				std::cout << FIL[i];
				if (i + 1 < FIL.size()) std::cout << ",";
			}
			std::cout << "]\n";
		}

		static void DebugPrintBusyOn(const MDP::State& state, const char* prefix)
		{
			std::cout << prefix << " busy_on = ";
			const auto& busy_on = state.server_manager.busy_on;
			for (size_t k = 0; k < busy_on.size(); ++k) {
				std::cout << "[";
				for (size_t j = 0; j < busy_on[k].size(); ++j) {
					std::cout << busy_on[k][j];
					if (j + 1 < busy_on[k].size()) std::cout << ",";
				}
				std::cout << "]";
				if (k + 1 < busy_on.size()) std::cout << " ";
			}
			std::cout << "\n";
		}

		static void DebugPrintActionQueue(const MDP::State& state, const char* prefix)
		{
			const auto& SM = state.server_manager;
			const auto& AQ = SM.action_queue;
			const auto& cnt = SM.action_counter;

			std::cout << prefix << " action_queue size = " << AQ.size()
				<< ", action_counter = " << cnt << "\n";

			std::cout << prefix << " actions = [ ";
			for (size_t i = 0; i < AQ.size(); ++i) {
				std::cout << "(" << AQ[i].server_index << "," << AQ[i].job_type << ")";
				if (i + 1 < AQ.size()) std::cout << ", ";
			}
			std::cout << " ]\n";
		}

		#endif // QUEUE_MDP_DEBUG
		// ===== END DEBUG UTILITIES =====
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
			vars.Add("valid_actions", 2);
			vars.Add("discount_factor", discount_factor);


			return vars;
		}

		double MDP::ModifyStateWithEvent(State& state, const Event& event) const
			{
				#if QUEUE_MDP_DEBUG
				std::cout << "\n[QMDP] ModifyStateWithEvent called\n";
				DebugPrintFIL(state, "[QMDP]   BEFORE event");
				DebugPrintBusyOn(state, "[QMDP]   BEFORE event");
				DebugPrintActionQueue(state, "[QMDP]   BEFORE event");
				#endif

			
				double event_sample = event.event_sample;
				double uniform_rate_next_fil = event.uniform_rate_next_fil;
			
				Event_type event_type = GetEventType(event_sample, state);
			
			
				if (event_type.type == Event_type::Type::JobCompletion)
				{
				
				#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   COMPLETION for server=" << event_type.server_index
						<< " job_type=" << event_type.job_type << "\n";
					DebugPrintBusyOn(state, "[QMDP]   BEFORE complete_job");
				#endif
					
					state.last_event_category = "completion";
				
					//complete job at server
					state.server_manager.complete_job(event_type.server_index, event_type.job_type);
				#if QUEUE_MDP_DEBUG
					DebugPrintBusyOn(state, "[QMDP]   AFTER complete_job");
				#endif
				
					state.queue_manager.complete_job(event_type.job_type, uniform_rate_next_fil); // job completed, remove from queue
					
					state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
					if (!state.server_manager.action_queue.empty()) {
						state.cat = StateCategory::AwaitAction();
					}
					else {
						state.cat = StateCategory::AwaitEvent();
					}
				
					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   AFTER event handling\n";
					DebugPrintFIL(state, "[QMDP]   AFTER event");
					DebugPrintBusyOn(state, "[QMDP]   AFTER event");
					DebugPrintActionQueue(state, "[QMDP]   AFTER event");
					#endif

					return 0.0; // no cost for job completion
				}
				else if (event_type.type == Event_type::Type::Arrival) {
					state.last_event_category = "arrival";


				
					state.queue_manager.arrival(event_type.arrival_index);
				
				
					state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());

					// if there are pending actions, go to await action
					if (!state.server_manager.action_queue.empty()) {
						state.cat = StateCategory::AwaitAction();
					}
					else {
						state.cat = StateCategory::AwaitEvent();
					}

					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   ARRIVAL EVENT\n";
					std::cout << "[QMDP]   AFTER event handling\n";
					DebugPrintFIL(state, "[QMDP]   AFTER event");
					DebugPrintBusyOn(state, "[QMDP]   AFTER event");
					DebugPrintActionQueue(state, "[QMDP]   AFTER event");
					#endif

					return 0.0; // cost for arrival
				}
				else if (event_type.type == Event_type::Type::Tick) {
					state.last_event_category = "tick";

				
					state.queue_manager.tick();
				
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
				
					if (!state.server_manager.action_queue.empty()) {
						state.cat = StateCategory::AwaitAction();
					}
					else {
						state.cat = StateCategory::AwaitEvent();
					}
				
					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   TICK EVENT\n";
					std::cout << "[QMDP]   AFTER event handling\n";
					DebugPrintFIL(state, "[QMDP]   AFTER event");
					DebugPrintBusyOn(state, "[QMDP]   AFTER event");
					DebugPrintActionQueue(state, "[QMDP]   AFTER event");
					#endif

					return cost; // cost for tick
				}
				else {
					state.last_event_category = "nothing";

					#if QUEUE_MDP_DEBUG
					std::cout << "[QMDP]   AFTER event handling\n";
					DebugPrintFIL(state, "[QMDP]   AFTER event");
					DebugPrintBusyOn(state, "[QMDP]   AFTER event");
					DebugPrintActionQueue(state, "[QMDP]   AFTER event");
					#endif

					if (!state.server_manager.action_queue.empty()) {
						state.cat = StateCategory::AwaitAction();
					}
					else {
						state.cat = StateCategory::AwaitEvent();
					}

					// No event
					return 0.0;
				}
			}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{
		#if QUEUE_MDP_DEBUG
			std::cout << "\n[QMDP] ModifyStateWithAction called\n";
			std::cout << "[QMDP]   incoming action = " << action << "\n";
			DebugPrintFIL(state, "[QMDP]   BEFORE action");
			DebugPrintBusyOn(state, "[QMDP]   BEFORE action");
			DebugPrintActionQueue(state, "[QMDP]   BEFORE action");
		#endif

			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());
		
		#if QUEUE_MDP_DEBUG
			std::cout << "[QMDP]   current_action = ("
				<< current_action.server_index << ","
				<< current_action.job_type << ")\n";
		#endif


			state.server_manager.take_action(action);

		#if QUEUE_MDP_DEBUG
			std::cout << "[QMDP]   AFTER take_action\n";
			DebugPrintFIL(state, "[QMDP]   AFTER action");
			DebugPrintBusyOn(state, "[QMDP]   AFTER action");
			DebugPrintActionQueue(state, "[QMDP]   AFTER action");
		#endif

		
			if (state.server_manager.get_action_counter() >= static_cast<int64_t>(state.server_manager.action_queue.size())) {
				// all actions processed
				state.server_manager.set_action_counter(0);
				state.cat = StateCategory::AwaitEvent();
			}
		#if QUEUE_MDP_DEBUG
			std::cout << "[QMDP]   All actions processed -> set cat=AwaitEvent, reset counter to 0\n";
		#endif
			return 0.0; 

		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);

			//add any other state variables. 
			vars.Add("FIL_waiting", queue_manager.FIL_waiting);
			vars.Add("action_counter", server_manager.action_counter);
			vars.Add("action_queue", server_manager.action_queue);
			
			// row-by-row storage instead of vector<vector<int64_t>>
			for (size_t k = 0; k < server_manager.busy_on.size(); ++k) {
				vars.Add("busy_on_" + std::to_string(k), server_manager.busy_on[k]);
			}
			vars.Add("last_event_category", last_event_category);

			return vars;
		}

		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			// Start from a valid, fully initialized state
			State state = GetInitialState();

			// Basic fields
			vars.Get("cat", state.cat);
			vars.Get("last_event_category", state.last_event_category);

			// Queue manager
			vars.Get("FIL_waiting", state.queue_manager.FIL_waiting);

			// Recompute rates (or call your helper methods)
			state.queue_manager.total_arrival_rate = 0.0;
			for (double r : arrival_rates) {
				state.queue_manager.total_arrival_rate += r;
			}
			state.queue_manager.total_tick_rate = tick_rate;

			// Server manager: action info
			vars.Get("action_counter", state.server_manager.action_counter);
			vars.Get("action_queue", state.server_manager.action_queue);

			// Server manager: busy_on rows
			state.server_manager.busy_on.clear();
			for (int64_t k = 0; k < k_servers; ++k) {
				std::vector<int64_t> row;
				std::string key = "busy_on_" + std::to_string(k);
				if (vars.HasKey(key)) {
					vars.Get(key, row);
				}
				else {
					// fallback: zero row with correct length
					row.assign(server_static_info[(size_t)k].can_serve.size(), 0);
				}
				state.server_manager.busy_on.push_back(std::move(row));
			}

			// Restore pointer & derived data
			state.server_manager.static_info = &server_static_info;
			state.server_manager.update_idle_capacity();
			state.server_manager.update_total_service_rate();

			return state;
		}
		/*
		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			State state{};		
			vars.Get("cat", state.cat);
			//initiate any other state variables. 
			vars.Get("FIL_waiting",state.queue_manager.FIL_waiting);
			vars.Get("action_counter", state.server_manager.action_counter);
			vars.Get("action_queue", state.server_manager.action_queue);
			for (size_t k = 0; k < state.server_manager.busy_on.size(); ++k) {
				vars.Get("busy_on_" + std::to_string(k), state.server_manager.busy_on[k]);
			}
			
			vars.Get("last_event_category", state.last_event_category);
			return state;
		}
		*/
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
				uniformization_rate += server_static_info[i].mu_k * server_static_info[i].servers;
			}

		#if QUEUE_MDP_DEBUG
			std::cout << "[QMDP]   INITIALIZATION \n";
			std::cout << "[QMDP]   Uniformization rate:  " << uniformization_rate <<std::endl;
		#endif


		}

		MDP::Event  MDP::GetEvent(RNG& rng) const {
			// Sample event based on rates
			double U = rng.genUniform();
			double event_sample = U * uniformization_rate;
			double uniform_rate_next_fil = rng.genUniform();
			MDP::Event event;
			event.event_sample = event_sample;
			event.uniform_rate_next_fil = uniform_rate_next_fil;
			return event;
		
		}

		MDP::Event_type MDP::GetEventType(const double event_sample, const State& state) const {
			
			#if QUEUE_MDP_DEBUG
				// --- Compute arrival & tick rates ---
			double arrival_rate = state.queue_manager.total_arrival_rate;
			double tick_rate = state.queue_manager.total_tick_rate;

			// --- Compute completion rate from busy_on and mu_k ---
			double completion_rate = 0.0;
			for (int64_t k = 0; k < k_servers; ++k) {
				double mu_k = server_static_info[(size_t)k].mu_k;
				for (size_t j = 0; j < server_static_info[(size_t)k].can_serve.size(); ++j) {
					int64_t n_busy = state.server_manager.busy_on[(size_t)k][j];
					completion_rate += n_busy * mu_k;
				}
			}

			double total_rate = arrival_rate + tick_rate + completion_rate;

			// "Nothing" rate only makes sense if you sample in [0,1)
			// and total_rate <= 1.0. Otherwise this will be <= 0.
			double nothing_rate = uniformization_rate - total_rate;

			std::cout << "\n[QMDP] GetEventType called\n";
			std::cout << "[QMDP]   event_sample      = " << event_sample << "\n";
			std::cout << "[QMDP]   arrival_rate      = " << arrival_rate << "\n";
			std::cout << "[QMDP]   tick_rate         = " << tick_rate << "\n";
			std::cout << "[QMDP]   completion_rate   = " << completion_rate << "\n";
			std::cout << "[QMDP]   total_rate        = " << total_rate << "\n";
			std::cout << "[QMDP]   nothing_rate      = " << nothing_rate << " (1 - total_rate)\n";

			// Optional: print boundaries in the same units as your comparisons
			std::cout << "[QMDP]   arrival boundary  = [0, "
				<< arrival_rate << ")\n";
			std::cout << "[QMDP]   tick boundary     = ["
				<< arrival_rate << ", "
				<< arrival_rate + tick_rate << ")\n";
			std::cout << "[QMDP]   completion start  = "
				<< (arrival_rate + tick_rate) << "\n";
		#endif
			if (event_sample < state.queue_manager.total_arrival_rate) {
				// Arrival event
				double cumulative_rate = 0.0;
				for (int64_t n = 0; n < n_jobs; ++n) {
					if (state.queue_manager.FIL_waiting[n] == -1) {
						cumulative_rate += arrival_rates[(size_t)n];

						if (event_sample < cumulative_rate) {
							return Event_type::MakeArrival(n);
						}
					}
				}
			}
			
			else if (event_sample < state.queue_manager.total_arrival_rate + state.queue_manager.total_tick_rate) {
				// Tick event
				return Event_type::MakeTick();
			}
			
			else {
				// Job completion event
				double cumulative_rate = state.queue_manager.total_arrival_rate + state.queue_manager.total_tick_rate;
				for (int64_t k = 0; k < k_servers; ++k) {
					for (size_t j = 0; j < server_static_info[(size_t)k].can_serve.size(); ++j) {
						cumulative_rate += state.server_manager.busy_on[(size_t)k][j] * server_static_info[(size_t)k].mu_k;
						if (event_sample < cumulative_rate) {							
							int64_t job_type = server_static_info[k].can_serve[j];
							return Event_type::MakeCompletion(k, job_type);
						}
					}
				}
			}
			
			return Event_type::MakeNothing(); // Should not reach here
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const
		{
			//This is optional to implement. You only need to implement it if you intend to solve versions of your problem
			//using exact methods that need access to the exact event probabilities.
			//Note that this is typically only feasible if the state space if finite and not too big, i.e. at most a few million states.
			throw DynaPlex::NotImplementedError();
		}

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {
			if (state.server_manager.action_queue.empty() ||
				state.server_manager.get_action_counter() >= (int64_t)state.server_manager.action_queue.size())
			{
				features.Add(-1); // no server
				features.Add(-1); // no job
				features.Add(state.queue_manager.FIL_waiting);
				return;
			}

			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());
			features.Add(current_action.server_index);
			features.Add(current_action.job_type);
			features.Add(state.queue_manager.FIL_waiting);
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
		#if QUEUE_MDP_DEBUG
			std::cout << "\n[QMDP] IsAllowedAction called\n";
			std::cout << "[QMDP]   requested action = " << action << "\n";
			DebugPrintActionQueue(state, "[QMDP]   ");
		#endif
			
			Action current_action = state.server_manager.action_queue.at(state.server_manager.get_action_counter());
			if (action == 1) {
				bool ok = state.server_manager.can_assign_job(current_action.server_index, current_action.job_type);
		#if QUEUE_MDP_DEBUG
				std::cout << "[QMDP]   current pair = ("
					<< current_action.server_index << ","
					<< current_action.job_type << ")\n";
				std::cout << "[QMDP]   IsAllowedAction -> " << (ok ? "true" : "false") << " for action=1\n";
		#endif
				
				// assign job
				return ok;
			}
			else if (action == 0) {
		#if QUEUE_MDP_DEBUG
				std::cout << "[QMDP]   IsAllowedAction -> true for action=0 (skip)\n";
		#endif		
				// do not assign job
				return true;
			}
			else {
		#if QUEUE_MDP_DEBUG
				std::cout << "[QMDP]   IsAllowedAction -> false for invalid action\n";
		#endif
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

