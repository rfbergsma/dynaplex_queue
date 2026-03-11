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
#include <cassert>
#include <queue>
#include <unordered_map>
#include <array>
#include <limits>


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

static void CheckDistribution(
	const std::vector<DynaPlex::Models::queue_mdp::MDP::nextStateProbability>& dist,
	const std::string& label,
	double tol = 1e-12)
{
	if (dist.empty()) {
		std::cout << "[FAIL] " << label << ": distribution is empty\n";
		std::abort();
	}

	double sum = 0.0;
	double minp = 1.0, maxp = 0.0;
	int neg = 0;

	for (const auto& e : dist) {
		sum += e.probability;
		minp = std::min(minp, e.probability);
		maxp = std::max(maxp, e.probability);
		if (e.probability < -tol) neg++;
	}

	std::cout << "[CHECK] " << label
		<< "  n=" << dist.size()
		<< "  sum=" << std::setprecision(17) << sum
		<< "  min=" << minp
		<< "  max=" << maxp
		<< "  neg=" << neg
		<< "\n";

	if (neg > 0 || std::abs(sum - 1.0) > 1e-9) {
		std::cout << "[FAIL] " << label << ": probabilities invalid\n";
		std::abort();
	}
}

int main() {

	auto& dp = DynaPlexProvider::Get();


	DynaPlex::VarGroup config;
	//retrieve MDP registered under the id string "lost_sales":
	auto path_to_json = dp.FilePath({ "mdp_config_examples", "queue_mdp" }, "mdp_config_0.json");
	//also possible to use a standard configuration file:
	config = VarGroup::LoadFromFile(path_to_json);



	// Optional: read id field if present
	if (config.HasKey("id")) {
		std::string id;
		config.Get("id", id);
		std::cout << "MDP id: " << id << "\n";
	}

	try
	{
		//test 1 (check initial state)
		DynaPlex::Models::queue_mdp::MDP mdp(config);
		auto state = mdp.GetInitialState();
		auto dist = mdp.getNextStateProbability(state, /*action*/ 0); // action ignored in AwaitEvent
		CheckDistribution(dist, "Initial state (AwaitEvent)");

		//test 2

		// Force a waiting job type 0
		{
			state.queue_manager.FIL_waiting[0] = 5;
			state.queue_manager.update_total_arrival_rate(mdp.arrival_rates);
			// if you keep tick rate state-dependent, update it too:
			state.queue_manager.update_total_tick_rate(mdp.tick_rate);

			state.server_manager.generate_actions(state.queue_manager.FIL_waiting);
			state.server_manager.set_action_counter(0);
			state.cat = state.server_manager.action_queue.empty()
				? DynaPlex::StateCategory::AwaitEvent()
				: DynaPlex::StateCategory::AwaitAction();

			if (state.cat == DynaPlex::StateCategory::AwaitAction()) {
				auto d0 = mdp.getNextStateProbability(state, 0);
				CheckDistribution(d0, "AwaitAction action=0");

				auto d1 = mdp.getNextStateProbability(state, 1);
				CheckDistribution(d1, "AwaitAction action=1");
			}
			else {
				std::cout << "No feasible actions created in forced state.\n";
			}
		}
		//test 3
		{
			auto dist = mdp.getNextStateProbability(state, 1);
			for (size_t idx = 0; idx < dist.size(); ++idx) {
				const auto& ns = dist[idx].next_state;

				// Example invariants:
				if (ns.cat == DynaPlex::StateCategory::AwaitAction()) {
					// should have an action queue and counter in range
					if (ns.server_manager.get_action_counter() < 0 ||
						ns.server_manager.get_action_counter() > (int64_t)ns.server_manager.action_queue.size()) {
						std::cout << "[FAIL] counter out of range in next state " << idx << "\n";
						std::abort();
					}
				}

				// Always true: FIL either -1 or >=0
				for (auto fil : ns.queue_manager.FIL_waiting) {
					if (fil < -1) {
						std::cout << "[FAIL] FIL < -1 in next state " << idx << "\n";
						std::abort();
					}
				}
			}
		}


		//step 4
		{
			std::mt19937_64 gen(123);
			std::uniform_real_distribution<double> U(0.0, 1.0);

			auto cur = state;
			for (int t = 0; t < 100; ++t) {
				int64_t a = 0;
				if (cur.cat == DynaPlex::StateCategory::AwaitAction()) {
					a = (U(gen) < 0.5) ? 0 : 1;
				}

				auto dist = mdp.getNextStateProbability(cur, a);
				CheckDistribution(dist, "step " + std::to_string(t));

				// sample next state
				double u = U(gen), cum = 0.0;
				size_t pick = 0;
				for (; pick < dist.size(); ++pick) {
					cum += dist[pick].probability;
					if (u <= cum) break;
				}
				if (pick >= dist.size()) pick = dist.size() - 1;
				cur = dist[pick].next_state;
			}
		}

		// ---- Run RVI (auto-selects M) ----
		auto sol = mdp.runRVI();
		std::cout << "\nFinal result: g* = " << std::setprecision(12) << sol.g_star
			      << "  (M=" << sol.M << ")\n";
	}
	catch (const std::exception& e)
	{
		std::cout << "exception: " << e.what() << std::endl;
	}
	return 0;
}
