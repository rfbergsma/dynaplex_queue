#include "dynaplex/dynaplexprovider.h"
#include "dynaplex/modelling/queue.h"
#include <vector>
#include <sstream>
#include <variant>
#include <numeric>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>
#include <cmath>
#include "../../../lib/models/models/queue_mdp/mdp.h"


using namespace DynaPlex;
using DynaPlex::VarGroup;
using DynaPlex::DynaPlexProvider;
using DynaPlex::Models::queue_mdp::MDP;


int main() {

	auto& dp = DynaPlexProvider::Get();


	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	auto path_to_json = dp.FilePath({ "mdp_config_examples", "queue_mdp" }, "mdp_config_0.json");
	//also possible to use a standard configuration file:
	config = VarGroup::LoadFromFile(path_to_json);

	//MDP mdp(config);


	// Optional: read id field if present
	if (config.HasKey("id")) {
		std::string id;
		config.Get("id", id);
		std::cout << "MDP id: " << id << "\n";
	}


	
	DynaPlex::Models::queue_mdp::MDP mdp(config);
	
	
	std::cout << "Server types: " << mdp.server_static_info.size() << "\n";
	for (size_t k = 0; k < mdp.server_static_info.size(); ++k) {
		const DynaPlex::Models::queue_mdp::MDP::ServerStaticInfo& st = mdp.server_static_info[k];
		std::cout << "  k=" << k
			<< "  n_servers=" << st.servers
			<< "  mu=" << st.mu_k
			<< "  can_serve=[";
		for (size_t j = 0; j < st.can_serve.size(); ++j) {
			std::cout << st.can_serve[j] << (j + 1 < st.can_serve.size() ? "," : "");
		}
		std::cout << "]\n";
	}

	auto state = mdp.GetInitialState();
	
	state.server_manager.update_idle_capacity();

	std::cout << "Has idle? " << (state.server_manager.has_idle_capacity ? "yes" : "no") << "\n";


	if (!mdp.server_static_info.empty() && !mdp.server_static_info[0].can_serve.empty()) {
		
		int64_t k = 0;
		int64_t job = mdp.server_static_info[0].can_serve[0];

		// Assign one job to server-type 0 on its first supported job
		state.server_manager.assign_job(state.server_manager, mdp.server_static_info, k, job);

		state.server_manager.update_idle_capacity();

		// Print busy row for k=0
		std::cout << "busy_on[k=0]: [";
		for (size_t j = 0; j < state.server_manager.busy_on[0].size(); ++j) {
			std::cout << state.server_manager.busy_on[0][j]
				<< (j + 1 < state.server_manager.busy_on[0].size() ? "," : "");
		}
		std::cout << "]\n";

		std::cout << "Has idle? " << (state.server_manager.has_idle_capacity ? "yes" : "no") << "\n";

		// idle capacity per server
		std::cout << "Idle capacity per job: [";
		for (size_t n = 0; n < state.server_manager.has_idle_capacity_per_job.size(); ++n) {
			std::cout << state.server_manager.has_idle_capacity_per_job[n]
				<< (n + 1 < state.server_manager.has_idle_capacity_per_job.size() ? "," : "");
		}
		std::cout << "]\n";
		//mdp.GetEvent(DynaPlex::RNG, state);



	}

	std::cout << "Uniformization rate: " << mdp.uniformization_rate << "\n";
	// initialize 
	//initialize server manager
	
	
	//server_static_info.clear();
	//server_static_info.resize((size_t)k_servers);

	try
	{
		//DynaPlex::MDP mdp = dp.GetMDP(config);
	}
	catch (const std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
	return 0;
}
