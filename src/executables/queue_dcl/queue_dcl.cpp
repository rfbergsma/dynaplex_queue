#include <iostream>
#include "dynaplex/dynaplexprovider.h"

	using namespace DynaPlex;
	int main() {

		auto& dp = DynaPlexProvider::Get();


		DynaPlex::VarGroup config;
		//retrieve MDP registered under the id string "lost_sales":
		//config.Add("id", "lost_sales");
		//add other parameters required by that MDP:
	
	
		std::string config_name = "mdp_config_0.json";
		auto path_to_json = dp.FilePath({ "mdp_config_examples", "queue_mdp" }, config_name);
		config = VarGroup::LoadFromFile(path_to_json);
		
		DynaPlex::MDP mdp = dp.GetMDP(config);

		DynaPlex::Policy fifo_policy = mdp->GetPolicy("random");

		DynaPlex::VarGroup demonstrator_config;
		demonstrator_config.Add("max_period_count", 10); //increase default number of periods


	
		auto demonstrator = dp.GetDemonstrator(demonstrator_config);
	
		auto trace = demonstrator.GetTrace(mdp, fifo_policy);

		/*
		for (auto v : trace) {
			std::cout << v.Dump(2) << std::endl;
		}
		*/
	
		DynaPlex::VarGroup nn_training{
					   {"early_stopping_patience",3}
		};
		DynaPlex::VarGroup nn_architecture{
				{"type","mlp"},
				{"hidden_layers",DynaPlex::VarGroup::Int64Vec{	128,64 }}
		};
		int64_t num_gens = 1;
		bool train = true;
		bool test = true;

		DynaPlex::VarGroup dcl_config{
			//just for illustration, so we collect only little data, so DCL will run fast but will not perform well.
			{"N",1000},
			{"num_gens",num_gens},
			{"M",100},
			{"nn_architecture",nn_architecture},
			{"nn_training",nn_training},
			{"retrain_lastgen_only",false},
			{"H",100}

		};



		auto dcl = dp.GetDCL(mdp, fifo_policy,dcl_config);
		dcl.TrainPolicy();

		auto policies = dcl.GetPolicies();


		return 0;
	}
