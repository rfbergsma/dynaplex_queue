#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "policies.h"


namespace DynaPlex::Models {
	namespace exact_test_case /*keep this in line with id below and with namespace name in header*/
	{
		VarGroup MDP::GetStaticInfo() const
		{
			VarGroup vars;
			//Needs to update later:
			vars.Add("valid_actions", 5);
			vars.Add("horizon_type", "finite");
			return vars;
		}


		double MDP::ModifyStateWithEvent(State& state, const Event& event) const
		{
			if (event == 0) {
				state.cat = StateCategory::Final();
			}
			else {
				state.cat = StateCategory::AwaitAction();
			}

			return 10.0;
		}

		double MDP::ModifyStateWithAction(MDP::State& state, int64_t action) const
		{
			//auto index = state.cat.Index();
			//if (index == 0)
			//{
			//	state.cat = StateCategory::AwaitAction(1);
			//}
			//else
			//{
			//	if (index == 1)
			//	{
			//		state.cat = StateCategory::AwaitAction(2);
			//	}
			//	else {
			//		if (index == 2)
			//		{
			//			state.cat = StateCategory::AwaitAction(3);
			//		}
			//		else 
			//		{/*
			//			if (state.period == 0) {
			//				state.cat = StateCategory::Final();
			//			}
			//			else {*/
			//			state.cat = StateCategory::AwaitEvent();
			//			//}
			//		}
			//	}
			//}

			if (state.currentNode != 3) {
				state.currentNode++;
			}
			else {
				state.currentNode = 0;
				state.cat = StateCategory::AwaitEvent();

			}

			return 0.0;
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const
		{
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("currentNode", currentNode);
			//add any other state variables. 
			return vars;
		}

		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			State state{};			
			vars.Get("cat", state.cat);
			vars.Get("currentNode", state.currentNode);
			//initiate any other state variables. 
			return state;
		}

		MDP::State MDP::GetInitialState() const
		{			
			State state{};
			state.cat = StateCategory::AwaitAction(0);//or AwaitAction(), depending on logic
			
			//initiate other variables. 
			state.currentNode = 0;
			return state;
		}

		MDP::MDP(const VarGroup& config)
		{
			dist = DiscreteDist::GetCustomDist({ 0.5, 0.5 });
	
			if (config.HasKey("discount_factor"))
				config.Get("discount_factor", discount_factor);
			else
				discount_factor = 1.0;
		
		}


		MDP::Event MDP::GetEvent(RNG& rng) const {
			return dist.GetSample(rng);
		}

		std::vector<std::tuple<MDP::Event, double>> MDP::EventProbabilities() const
		{
			return dist.QuantityProbabilities();
		}


		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const {
			//features.Add(state.cat);
			features.Add(state.currentNode);
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
			return state.cat;
		}

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
			return true;
		}


		void Register(DynaPlex::Registry& registry)
		{
			DynaPlex::Erasure::MDPRegistrar<MDP>::RegisterModel("exact_test_case", "nonempty description", registry);
			
		}
	}
}

