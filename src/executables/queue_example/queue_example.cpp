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
using DynaPlex::RNG;



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

		state.queue_manager.arrival(job); // add job of type 0 to queue

		// Assign one job to server-type 0 on its first supported job
		state.server_manager.assign_job(state.server_manager, mdp.server_static_info, k, job);
		
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
		


		// construct an RNG object:
		DynaPlex::RNG rng(true, 123456789, 0, 0, 1);
		// get an event:
		auto event = mdp.GetEvent(rng, state);
		
		
		int n_samples = 10000;
		std::vector<int> event_counts(4, 0); // Assuming 4 event types
		std::vector<std::vector<double>> event_rates(4); // Store rates for each event type at each step

		for (int i = 0; i < n_samples; ++i) {
			auto ev = mdp.GetEvent(rng, state);
			switch (ev.type) {
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Arrival:
				event_counts[0]++;
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Tick:
				event_counts[1]++;
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::JobCompletion:
				event_counts[2]++;
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Nothing:
				event_counts[3]++;
				break;
			default:
				break;
			}

			// Calculate rates at this step and store them
			for (size_t j = 0; j < event_counts.size(); ++j) {
				event_rates[j].push_back(static_cast<double>(event_counts[j]) / (i + 1));
			}
		}

		// Calculate mean and standard deviation for each event type
		std::cout << "Empirical event probabilities after " << n_samples << " samples:\n";
		for (size_t j = 0; j < event_counts.size(); ++j) {
			double mean = static_cast<double>(event_counts[j]) / n_samples;

			// Calculate variance
			double variance = 0.0;
			for (const auto& rate : event_rates[j]) {
				variance += (rate - mean) * (rate - mean);
			}
			variance /= n_samples;

			double stddev = std::sqrt(variance);

			// Print results
			std::cout << "  Event type " << j << ":\n";
			std::cout << "    Mean: " << mean << "\n";
			std::cout << "    Standard deviation: " << stddev << "\n";
		}
		
		std::cout << "Check actions: \n";

		int64_t n_events = 10;


		auto state = mdp.GetInitialState();


		//print starting FIL waiting
		std::cout << " Initial FIL waiting: ";
		auto FIL_waiting = state.queue_manager.get_FIL_waiting();
		for (size_t n = 0; n < FIL_waiting.size(); ++n) {
			std::cout << FIL_waiting[n] << (n + 1 < FIL_waiting.size() ? "," : "");
		}
		std::cout << "\n";

		std::cout << " Initial busy on: ";
		for (size_t k = 0; k < state.server_manager.busy_on.size(); ++k) {
			std::cout << "[";
			for (size_t j = 0; j < state.server_manager.busy_on[k].size(); ++j) {
				std::cout << state.server_manager.busy_on[k][j]
					<< (j + 1 < state.server_manager.busy_on[k].size() ? "," : "");
			}
			std::cout << "] ";
		}

		std::cout << "\n";

		for (int64_t i = 0; i < n_events; ++i) {
			auto ev = mdp.GetEvent(rng, state);
			
			switch (ev.type) {
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Arrival:
				std::cout << " Event " << i << ": type= Arrival \n";
				std::cout << "  arrival index: " << ev.arrival_index << "\n";
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Tick:
				std::cout << " Event " << i << ": type= Tick \n";
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::JobCompletion:
				std::cout << " Event " << i << ": type= Job completion \n";
				std::cout << "  server index: " << ev.server_index << "\n";
				std::cout << "  job type: " << ev.job_type << "\n";
				break;
			case DynaPlex::Models::queue_mdp::MDP::Event::Type::Nothing:
				std::cout << " Event " << i << ": type= Nothing \n";
				break;
			}
			mdp.ModifyStateWithEvent(state, ev);
			
			std::cout << " FIL waiting: ";
			auto FIL_waiting = state.queue_manager.get_FIL_waiting();
			for (size_t n = 0; n < FIL_waiting.size(); ++n) {
				std::cout << FIL_waiting[n] << (n + 1 < FIL_waiting.size() ? "," : "");
			}
			std::cout << "\n";

			std::cout << " Busy on: ";
			for (size_t k = 0; k < state.server_manager.busy_on.size(); ++k) {
				std::cout << "[";
				for (size_t j = 0; j < state.server_manager.busy_on[k].size(); ++j) {
					std::cout << state.server_manager.busy_on[k][j]
						<< (j + 1 < state.server_manager.busy_on[k].size() ? "," : "");
				}
				std::cout << "] ";
			}

			std::cout << "\n";

			std::cout << "  Generated actions: ";
			
			state.server_manager.print_actions();
			
			std::cout << "\n";
		}

		//event.type

		
	}
	
	
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
