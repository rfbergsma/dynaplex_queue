#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"


namespace DynaPlex::Models {
	namespace queue_mdp /*keep this in line with id below and with namespace name in header*/
	{
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
			return vars;
		}


		double MDP::ModifyStateWithEvent(State& state, const Event& event) const
		{
			throw DynaPlex::NotImplementedError();
			//implement change to state
			// do not forget to update state.cat. 
			//returns costs.
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{
			throw DynaPlex::NotImplementedError();
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
			//initiate other variables. 
			// initialize server manager state
			state.server_manager.busy_on.resize(server_static_info.size());
			for (size_t k = 0; k < server_static_info.size(); ++k) {
				state.server_manager.busy_on[k].resize(server_static_info[k].can_serve.size(), 0);
			}
			// Link static info
			state.server_manager.static_info = &server_static_info;
			// Update idle capacity
			state.server_manager.update_idle_capacity(state.server_manager.busy_on);

			
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

			//initialize server manager
			std::vector<ServerStaticInfo> server_static_info;

			for (int64_t i = 0; i < k_servers; ++i) {
				VarGroup serverConfig = GetSubGroup(config, "server_type_" + std::to_string(i));
				serverConfig.Get("n_servers", server_static_info[i].servers);
				serverConfig.Get("service_rate", server_static_info[i].mu_k);
				serverConfig.Get("can_serve", server_static_info[i].can_serve);
			}
			
		}


		MDP::Event MDP::GetEvent(RNG& rng) const {
			throw DynaPlex::NotImplementedError();
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const
		{
			//This is optional to implement. You only need to implement it if you intend to solve versions of your problem
			//using exact methods that need access to the exact event probabilities.
			//Note that this is typically only feasible if the state space if finite and not too big, i.e. at most a few million states.
			throw DynaPlex::NotImplementedError();
		}


		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {
			throw DynaPlex::NotImplementedError();
		}
		
		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{//Here, we register any custom heuristics we want to provide for this MDP.	
		 //On the generic DynaPlex::MDP constructed from this, these heuristics can be obtained
		 //in generic form using mdp->GetPolicy(VarGroup vars), with the id in var set
		 //to the corresponding id given below.
			registry.Register<EmptyPolicy>("empty_policy",
				"This policy is a place-holder, and throws a NotImplementedError when asked for an action. ");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			//this typically works, but state.cat must be kept up-to-date when modifying states. 
			return state.cat;
		}	

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			throw DynaPlex::NotImplementedError();
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

