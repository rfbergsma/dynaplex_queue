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



double expected_next_fil(int i, double lambda, double gamma) {
	if (i <= 0) return 0.0;
	double denom = lambda + gamma;
	if (denom <= 0.0) return 0.0;
	double alpha = lambda / denom;
	double beta = gamma / denom;

	double EX = 0.0;
	for (int j = 1; j <= i; ++j) {
		double p = std::pow(beta, i - j) * alpha;
		EX += j * p;
	}
	return EX;
}


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



		std::cout << "Action update checker: \n";
		std::cout << "Case 1: server 0 free, server 1 has 1 capacity left, jobs in queue 0,1,2,: \n";


		auto new_state = mdp.GetInitialState();
		new_state.queue_manager.arrival(0); // add job of type 0 to queue
		new_state.queue_manager.arrival(1); // add job of type 1 to queue
		new_state.queue_manager.arrival(2); // add job of type 2 to queue
		new_state.server_manager.assign_job(new_state.server_manager, mdp.server_static_info, 1, 1);


		new_state.server_manager.generate_actions(new_state.queue_manager.get_FIL_waiting());
		new_state.server_manager.set_action_counter(3);
		new_state.server_manager.print_actions();
		
		std::cout << "action counter = " << new_state.server_manager.get_action_counter() << " \n";
		new_state.server_manager.take_action(1);
		std::cout << "After taking current action: \n";
		new_state.server_manager.print_actions();
		std::cout << "action counter = " << new_state.server_manager.get_action_counter() << " \n";


		std::cout << "\nCase 2: server 0 has capacity 1, both servers free, jobs 0,1,2 in queue:\n";

		{
			auto state = mdp.GetInitialState();

			// Jobs in queue: 0, 1, 2
			state.queue_manager.arrival(0);
			//state.queue_manager.arrival(1);
			state.queue_manager.arrival(1); // second job type 1
			state.queue_manager.arrival(2);

			// No servers busy initially
			state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());

			// Expected action list (conceptually): [0,0], [0,1], [1,1], [1,2]
			state.server_manager.set_action_counter(1);  // execute [0,1]
			std::cout << "Before taking action:\n";
			state.server_manager.print_actions();
			std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";

			// Take current action (server 0 does job 1)
			state.server_manager.take_action(1);

			std::cout << "After taking current action:\n";
			state.server_manager.print_actions();
			std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";
			std::cout << "Expected: new queue ~ [1,1], [1,2], action_counter = 0\n";
		}

		std::cout << "\nCase 3: server 0 (capacity 1) already full, server 1 free, jobs 1 and 2 in queue:\n";

		{
			auto state = mdp.GetInitialState();

			// Make server 0 busy so it cannot be used in actions
			// (job type here doesn't matter much, just something server 0 can serve)
			state.queue_manager.arrival(0);
			//state.server_manager.assign_job(state.server_manager, mdp.server_static_info, 0, 0);

			// Now put jobs 1 and 2 in the queue
			state.queue_manager.arrival(1);
			state.queue_manager.arrival(2);

			//state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());

			state.server_manager.action_queue.push_back(DynaPlex::Models::queue_mdp::MDP::Action { 0, 0 });
			state.server_manager.action_queue.push_back(DynaPlex::Models::queue_mdp::MDP::Action{ 1, 1 });
			state.server_manager.action_queue.push_back(DynaPlex::Models::queue_mdp::MDP::Action{ 0, 1 });
			state.server_manager.action_queue.push_back(DynaPlex::Models::queue_mdp::MDP::Action{ 1, 2 });
			
			// Expected action list conceptually: [1,1], [1,2]
			state.server_manager.set_action_counter(2);  // execute [1,1]
			std::cout << "Before taking action:\n";
			state.server_manager.print_actions();
			std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";

			// Take current action (server 1 does job 1)
			state.server_manager.take_action(1);

			std::cout << "After taking current action:\n";
			state.server_manager.print_actions();
			std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";
			std::cout << "Expected: new queue ~ [1,1],[1,2], action_counter = 0\n";
		}
	}


	std::cout << "\nCase 4: server 1 has capacity 2, both servers free, multiple jobs of type 1:\n";

	{
		auto state = mdp.GetInitialState();

		// Jobs 0, 1, 1, 2 in queue:
		state.queue_manager.arrival(0);
		state.queue_manager.arrival(1);
		state.queue_manager.arrival(2);

		// Both servers free; server 1 has capacity 2 (from your static info)
		state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());

		// Expected action order conceptually: [0,0], [0,1], [1,1], [1,2]
		state.server_manager.set_action_counter(2);  // execute [1,1]
		std::cout << "Before taking action:\n";
		state.server_manager.print_actions();
		std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";

		// Take current action (server 1 does job 1)
		state.server_manager.take_action(1);

		std::cout << "After taking current action:\n";
		state.server_manager.print_actions();
		std::cout << "action counter = " << state.server_manager.get_action_counter() << "\n";
		std::cout << "Expected: queue unchanged, action_counter = 3\n";
	}


	// write a short script that samples the next fil and compares to expected value.
	
	
	
	int64_t number_of_samples = 10000;
	std::vector<int64_t> sampled_fils;
	sampled_fils.resize(number_of_samples);
	double arrival_rate = 0.2;
	double tick_rate = 1;
	int64_t max_current_fil = 10;
	DynaPlex::RNG rng(true, 123456789, 0, 0, 1);
	

	for (int64_t fil = 0; fil < max_current_fil; fil++) {
		
		for (int64_t i = 0; i < number_of_samples; ++i) {
			int64_t next_fil = state.queue_manager.sample_next_fil_after_completion(fil, arrival_rate, tick_rate, rng);
			sampled_fils.at(i) = next_fil;
		}

		std::cout << "Current FIL: " << fil << "\n";
		// compute empirical mean
		double empirical_mean = std::accumulate(sampled_fils.begin(), sampled_fils.end(), 0.0) / number_of_samples;
		std::cout << "Empirical mean of sampled next FIL: " << empirical_mean << "\n";
		std::cout << "Expected mean of next FIL: " << expected_next_fil(fil, arrival_rate, tick_rate) << "\n";
		//percentage deviation
		std::cout << "percentage difference: " << std::abs(empirical_mean - expected_next_fil(fil, arrival_rate, tick_rate)) / expected_next_fil(fil, arrival_rate, tick_rate) * 100.0 << " %\n";
	
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
