#include "mdp.h"
#include "dynaplex/erasure/mdpregistrar.h"
#include "dynaplex/retrievestate.h"
#include "dynaplex/parallel_execute.h"
#include "policies.h"
#include <deque>
#include <iostream>
#include <iomanip>
#include <functional>
#include <span>
#include <thread>


namespace DynaPlex::Models {
	namespace queue_mdp /*keep this in line with id below and with namespace name in header*/
	{
		// ===== DEBUG UTILITIES FOR queue_mdp  =====
		#define QUEUE_MDP_DEBUG 0

		#if QUEUE_MDP_DEBUG

		static void DebugPrintFIL(const MDP::State& state, const char* prefix)
		{
			std::cout << prefix << " FIL_waiting = [";
			const auto FIL = state.queue_manager.get_FIL_waiting();
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
				//std::cout << "ModifyStateWithEvent callced" << std::endl;

				#if QUEUE_MDP_DEBUG
				std::cout << "\n[QMDP] ModifyStateWithEvent called\n";
				DebugPrintFIL(state, "[QMDP]   BEFORE event");
				DebugPrintBusyOn(state, "[QMDP]   BEFORE event");
				DebugPrintActionQueue(state, "[QMDP]   BEFORE event");
				#endif

			
				double event_sample = event.event_sample;
				double uniform_rate_next_fil = event.uniform_rate_next_fil;
				
				if (state.next_fil_job_type != -1) {
					const int64_t n = state.next_fil_job_type;

					// Pop head / reveal next head using your existing sampler
					state.queue_manager.complete_job(n, uniform_rate_next_fil);

					// clear pending refresh
					state.next_fil_job_type = -1;

					// regenerate actions because FIL changed
					state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
					state.server_manager.set_action_counter(0);
					state.stochastic_draws = event.stochastic_draws;

					// go back to decision-making
					state.cat = state.server_manager.action_queue.empty()
						? StateCategory::AwaitEvent()
						: StateCategory::AwaitAction();

					state.last_event_category = "fil_refresh";
					return 0.0;
				}


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
				
					//state.queue_manager.complete_job(event_type.job_type, uniform_rate_next_fil); // job completed, remove from queue
					
					state.server_manager.generate_actions(state.queue_manager.get_FIL_waiting());
					state.server_manager.set_action_counter(0);
					state.stochastic_draws = event.stochastic_draws;
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
					state.server_manager.set_action_counter(0);
					state.stochastic_draws = event.stochastic_draws;

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
				
					// Charge tick cost using the configurable reward function (post-tick FIL)
					double cost = ComputeTickCost(state);
				
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
				// Safety: if action_queue is empty, nothing to do
				if (state.server_manager.action_queue.empty()) {
					state.server_manager.set_action_counter(0);
					state.cat = StateCategory::AwaitEvent();
					return 0.0;
				}

				const int64_t acnt = state.server_manager.get_action_counter();
				if (acnt < 0 || acnt >= (int64_t)state.server_manager.action_queue.size()) {
					// Counter out of range: reset and go to event
					state.server_manager.set_action_counter(0);
					state.cat = StateCategory::AwaitEvent();
					return 0.0;
				}

				Action current_action = state.server_manager.action_queue.at((size_t)acnt);

				if (action == 1) {
					// Assign: trigger FIL refresh; ModifyStateWithEvent completes it
					state.server_manager.take_action(1);
					state.next_fil_job_type = current_action.job_type;
					state.cat = StateCategory::AwaitEvent();
					return 0.0;
				}

				// action == 0: skip this candidate, advance to next
				int64_t next_cnt = acnt + 1;
				if (next_cnt >= (int64_t)state.server_manager.action_queue.size()) {
					// Exhausted all candidates: idle for this epoch
					state.server_manager.set_action_counter(0);
					state.cat = StateCategory::AwaitEvent();
				} else {
					state.server_manager.set_action_counter(next_cnt);
					// cat stays AwaitAction — present next candidate
				}
				return 0.0;
			}

			std::vector<std::pair<int64_t, double>>
				MDP::NextFILDistribution(int64_t i, double lambda, double gamma)
			{
				std::vector<std::pair<int64_t, double>> dist;

				if (i < 0) { // already empty; shouldn't happen on completion
					dist.push_back({ -1, 1.0 });
					return dist;
				}
				if (i == 0) { // completion makes it empty deterministically in your code
					dist.push_back({ -1, 1.0 });
					return dist;
				}

				const double denom = lambda + gamma;
				if (denom <= 0.0) { // no arrivals/ticks; safest deterministic
					dist.push_back({ -1, 1.0 });
					return dist;
				}

				const double alpha = lambda / denom;
				const double beta = gamma / denom;

				// j = 1..i
				for (int64_t j = 1; j <= i; ++j) {
					const int64_t h = i - j;                // h >= 0
					const double p = std::pow(beta, (double)h) * alpha;
					if (p > 0.0) dist.push_back({ j, p });
				}

				// empty outcome
				const double p_empty = std::pow(beta, (double)i);
				if (p_empty > 0.0) dist.push_back({ -1, p_empty });

				// (Optional) normalize for numeric safety
				double sum = 0.0;
				for (auto& kv : dist) sum += kv.second;
				if (sum > 0.0) for (auto& kv : dist) kv.second /= sum;

				return dist;
		}
	
		std::vector<MDP::nextStateProbability> MDP::getNextStateProbability(const MDP::State& state, int64_t action) const{
			
			std::vector<MDP::nextStateProbability> out;   // <-- define it

			if (state.cat == StateCategory::AwaitAction()) {
				MDP::State modified_state = state;

				int64_t action_counter = state.server_manager.get_action_counter();
				Action current_action = state.server_manager.action_queue.at(action_counter);
				int64_t current_job_type = current_action.job_type;
				int64_t current_server_type = current_action.server_index;

				//modified the action queue and counter

				
				if (action == 1) {
					// Generate vector of states, each equal to modified state, but with the FIL of the current action
					modified_state.server_manager.take_action(action);

					// RVI is FIL-projected: read and write only the FIL (position 0) for this type.
					const int64_t i = state.queue_manager.get_FIL_waiting()[(size_t)current_job_type];
					const double lambda = arrival_rates[current_job_type];
					const double gamma = state.queue_manager.total_tick_rate;

					auto fil_dist = NextFILDistribution(i, lambda, gamma);

					for (auto [next_fil, p_fil] : fil_dist) {
						MDP::State s2 = modified_state;

						// Set new FIL for this job type (clears any deeper positions — FIL-projected)
						s2.queue_manager.set_fil(current_job_type, next_fil);

						// Update derived rates to reflect new FIL
						s2.queue_manager.update_total_arrival_rate(arrival_rates);

						// Regenerate action queue from scratch using new FIL values
						s2.server_manager.generate_actions(s2.queue_manager.get_FIL_waiting());
						s2.server_manager.set_action_counter(0);
						s2.cat = s2.server_manager.action_queue.empty()
							? StateCategory::AwaitEvent()
							: StateCategory::AwaitAction();

						out.push_back({ std::move(s2), p_fil });
					}
					
				
				}
				else {
					// action == 0: skip this candidate, advance counter to next
					int64_t next_cnt = action_counter + 1;
					if (next_cnt >= (int64_t)modified_state.server_manager.action_queue.size()) {
						// Exhausted all candidates: idle
						modified_state.server_manager.set_action_counter(0);
						modified_state.cat = StateCategory::AwaitEvent();
					} else {
						modified_state.server_manager.set_action_counter(next_cnt);
						// cat stays AwaitAction — present next candidate
					}
					out.push_back({ std::move(modified_state), 1 });
				}
				return out;
				
			}
			// if awaitevent
			else {
				const double Lambda = uniformization_rate;

				// 1) ARRIVALS (only if queue not full for that type)
				double A = 0.0;
				for (int64_t n = 0; n < n_jobs; ++n) {
					if (state.queue_manager.waiting[(size_t)n].empty()) {
						const double r = arrival_rates[(size_t)n];

						MDP::State s2 = state;
						// apply arrival deterministically:
						s2.queue_manager.arrival(n);
						// after arrival, tick-rate depends on #waiting:
						s2.queue_manager.update_total_tick_rate(tick_rate);
						s2.queue_manager.update_total_arrival_rate(arrival_rates);

						s2.server_manager.generate_actions(s2.queue_manager.get_FIL_waiting());
						s2.server_manager.set_action_counter(0);
						
						s2.cat = s2.server_manager.action_queue.empty()
							? StateCategory::AwaitEvent()
							: StateCategory::AwaitAction();

						out.push_back({ std::move(s2), r / Lambda });
						A += r;
					}
				}

				// 2) TICK
				const double T = state.queue_manager.total_tick_rate;
				if (T > 0.0) {
					MDP::State s2 = state;
					s2.queue_manager.tick();
					// after tick, nothing changes about who is empty/full,
					// but if you treat total_tick_rate as depending on #waiting, it stays the same here.
					out.push_back({ std::move(s2), T / Lambda });
				}

				// 3) COMPLETIONS / DEPARTURES
				// Scan the same buckets as GetEventType: (k, j in can_serve)
				double C = 0.0;
				for (int64_t k = 0; k < k_servers; ++k) {
					for (size_t j = 0; j < server_static_info[(size_t)k].can_serve.size(); ++j) {
						MDP::State s2 = state;

						const int64_t busy = state.server_manager.busy_on[(size_t)k][j];
						if (busy <= 0) continue;

						const int64_t job = server_static_info[(size_t)k].can_serve[j];
						const double mu = server_static_info[(size_t)k].mu_kj[j];
						const double r = mu * (double)busy;
						const double p_event = r / Lambda;

						// completion changes FIL(job) stochastically
						s2.server_manager.complete_job(k, job);

						s2.server_manager.generate_actions(s2.queue_manager.get_FIL_waiting());
						s2.server_manager.set_action_counter(0);
						s2.cat = s2.server_manager.action_queue.empty()
							? StateCategory::AwaitEvent()
							: StateCategory::AwaitAction();


						out.push_back({ std::move(s2), p_event});

						C += r;
						

					}
				}

				// 4) NOTHING / SELF-LOOP (uniformization leftover)
				const double R = A + T + C;
				const double r_nothing = std::max(0.0, Lambda - R);
				if (r_nothing > 0.0) {
					out.push_back({ state, r_nothing / Lambda }); // self-loop
				}


			}
			return out;
		}

		double MDP::ComputeTickCost(const State& state) const {
			double cost = 0.0;
			for (size_t n = 0; n < (size_t)n_jobs; ++n) {
				const auto& q = state.queue_manager.waiting[n];
				if (q.empty()) continue;

				if (reward_type == 0) {
					// Binary: only FIL matters (flat cost if FIL > deadline)
					if (q.front() > (int64_t)due_times[n])
						cost += cost_rates[n];
				}
				else {
					// Queue-lateness: exact excess summed over all tracked positions,
					// plus a Koole tail approximation for untracked positions beyond max_queue_depth.

					// Exact contribution from tracked positions
					for (const int64_t t : q) {
						const double excess = (double)t - due_times[n];
						if (excess > 0.0)
							cost += cost_rates[n] * excess;
					}

					// Tail approximation for jobs beyond max_queue_depth:
					// Same Koole formula as the old single-position formula, but now anchored
					// at the deepest tracked position (q.back()) instead of the FIL.
					// At max_queue_depth==1, q.back()==q.front()==FIL, so this is identical
					// to the original formula — full backward compatibility.
					const double bottom_excess = (double)q.back() - due_times[n];
					if (bottom_excess > 0.0) {
						const double tail = (arrival_rates[n] / tick_rate)
						                  * std::max(0.0, bottom_excess - 1.0) * bottom_excess / 2.0;
						cost += cost_rates[n] * tail;
					}
				}
			}
			return cost;
		}

		double MDP::GetImmediateCost(const State& state) const {
			if (state.cat != StateCategory::AwaitEvent()) return 0.0;
			return (tick_rate / uniformization_rate) * ComputeTickCost(state);
		}

		DynaPlex::VarGroup MDP::State::ToVarGroup() const {
			DynaPlex::VarGroup vars;
			vars.Add("cat", cat);
			vars.Add("last_event_category", last_event_category);

			vars.Add("server", server_manager.ToVarGroup());
			vars.Add("queue", queue_manager.ToVarGroup());
			vars.Add("next_fil_job_type", next_fil_job_type);
			return vars;
		}


		
		MDP::State MDP::GetState(const VarGroup& vars) const
		{
			
			State state = GetInitialState(); // ensures static_info + busy_on shape exist

			/*
			std::cout << "[QMDP] GetState called\n";
			std::cout << "[QMDP] expected servers = " << k_servers
				<< " server_static_info.size()=" << server_static_info.size()
				<< " busy_on.size() currently=" << state.server_manager.busy_on.size()
				<< "\n";
			*/
			vars.Get("cat", state.cat);
			vars.Get("last_event_category", state.last_event_category);
			vars.Get("next_fil_job_type", state.next_fil_job_type);

			VarGroup qvg, svg;
			vars.Get("queue", qvg);
			vars.Get("server", svg);
			
			state.queue_manager = multi_queue(qvg);
			state.server_manager = ServerDynamicState(svg);


			// Re-attach pointer + recompute derived fields (ServerDynamicState ctor can�t know this pointer)
			state.server_manager.static_info = &server_static_info;

			state.server_manager.update_total_service_rate();

			// Restore arrival_rates (MDP parameter; not always stored in VarGroup)
			if (state.queue_manager.arrival_rates.empty())
				state.queue_manager.arrival_rates = arrival_rates;
			// Recompute derived queue rates
			state.queue_manager.update_total_arrival_rate(arrival_rates);
			state.queue_manager.update_total_tick_rate(tick_rate);

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
			state.queue_manager.initialize(n_jobs, tick_rate, arrival_rates, max_queue_depth);

			state.next_fil_job_type = -1;
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

			// Normalise so that callers express cost_rates and due_times in real-time units,
			// making tick_rate a pure granularity parameter with no effect on cost levels:
			//   cost_rates[n]  : cost per unit real time  →  stored as cost per tick (/ tick_rate)
			//   due_times[n]   : deadline in real seconds  →  stored in tick units   (* tick_rate)
			// At tick_rate = 1 this is a no-op, so existing configs are unaffected.
			for (auto& c : cost_rates) c /= tick_rate;
			for (auto& d : due_times)  d *= tick_rate;

			if (config.HasKey("reward_type"))
				config.Get("reward_type", reward_type);
			else
				reward_type = 1;  // default: queue-lateness formula

			// max_queue_depth: number of queue positions tracked per job type (default 1 = FIL only)
			if (config.HasKey("max_queue_depth"))
				config.Get("max_queue_depth", max_queue_depth);
			else
				max_queue_depth = 1;

			// feature_queue_depth: NN feature slots per job type (default = max_queue_depth)
			// Set > max_queue_depth to zero-pad for cross-deployment experiments
			if (config.HasKey("feature_queue_depth"))
				config.Get("feature_queue_depth", feature_queue_depth);
			else
				feature_queue_depth = max_queue_depth;

			//initialize server manager
			server_static_info.clear();
			server_static_info.resize((size_t)k_servers);

			for (int64_t i = 0; i < k_servers; ++i) {
				VarGroup serverConfig = GetSubGroup(config, "server_type_" + std::to_string(i));
				serverConfig.Get("servers", server_static_info[i].servers);
				// Accept both "service_rate" (scalar, same rate for all job types) and
				// "service_rates" (vector, one rate per job type).
				// Check for the scalar key WITHOUT fuzzy-match warnings (warn=false).
				if (serverConfig.HasKey("service_rate", false)) {
					double scalar_rate;
					serverConfig.Get("service_rate", scalar_rate);
					server_static_info[i].mu_kj.assign((size_t)n_jobs, scalar_rate);
				} else {
					serverConfig.Get("service_rates", server_static_info[i].mu_kj);
				}
				serverConfig.Get("can_serve", server_static_info[i].can_serve);
			}

			uniformization_rate = tick_rate;
			// + sum of arrival rates
			for (const auto& rate : arrival_rates) {
				uniformization_rate += rate;
			}
			// sum of all service rates (upper bound: max mu per pool × servers)
			for (int64_t i = 0; i < k_servers; ++i) {
				double max_mu = *std::max_element(server_static_info[i].mu_kj.begin(),
				                                  server_static_info[i].mu_kj.end());
				uniformization_rate += max_mu * server_static_info[i].servers;
			}

		int_hash = config.Int64Hash();

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
			// Draw one uniform per candidate slot for StochasticFIFOPolicy.
			// Upper bound on candidates: k_servers * n_jobs.
			int64_t max_candidates = k_servers * n_jobs;
			event.stochastic_draws.resize((size_t)max_candidates);
			for (auto& d : event.stochastic_draws)
				d = rng.genUniform();
			return event;

		}

		MDP::Event_type MDP::GetEventType(const double event_sample, const State& state) const {
			
			#if QUEUE_MDP_DEBUG
				// --- Compute arrival & tick rates ---
			double arrival_rate = state.queue_manager.total_arrival_rate;
			double tick_rate = state.queue_manager.total_tick_rate;

			// --- Compute completion rate from busy_on and mu_kj ---
			double completion_rate = 0.0;
			for (int64_t k = 0; k < k_servers; ++k) {
				for (size_t j = 0; j < server_static_info[(size_t)k].can_serve.size(); ++j) {
					int64_t n_busy = state.server_manager.busy_on[(size_t)k][j];
					completion_rate += n_busy * server_static_info[(size_t)k].mu_kj[j];
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
				// Arrival event — fires for types whose queue is not yet full
				double cumulative_rate = 0.0;
				for (int64_t n = 0; n < n_jobs; ++n) {
					if ((int64_t)state.queue_manager.waiting[(size_t)n].size() < state.queue_manager.max_queue_depth) {
						cumulative_rate += arrival_rates[(size_t)n];
						if (event_sample < cumulative_rate)
							return Event_type::MakeArrival(n);
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
						cumulative_rate += state.server_manager.busy_on[(size_t)k][j] * server_static_info[(size_t)k].mu_kj[j];
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

		/*
		void MDP::GetFeatures(const State& state, DynaPlex::Features& features)const {
			//std::cout << "[QMDP] GetFeatures called\n";

			
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
		*/

		void MDP::GetFeatures(const State& state, DynaPlex::Features& features) const
		{
			// ----- (1) Event category as an integer -----
			// arrival=0, tick=1, completion=2, nothing=3, unknown=-1
			int64_t last_evt = -1;
			if (state.last_event_category == "arrival")     last_evt = 0;
			else if (state.last_event_category == "tick")        last_evt = 1;
			else if (state.last_event_category == "completion")  last_evt = 2;
			else if (state.last_event_category == "nothing")     last_evt = 3;
			features.Add(last_evt);

			// ----- (2) Queue state -----
			// feature_queue_depth slots per job type (zero-padded for untracked positions).
			// When feature_queue_depth == max_queue_depth: exact values.
			// When feature_queue_depth > max_queue_depth: extra slots are 0
			//   so a policy trained here can be deployed on a deeper-queue MDP.
			const auto& waiting = state.queue_manager.waiting;
			for (size_t n = 0; n < (size_t)n_jobs; ++n) {
				for (int64_t d = 0; d < feature_queue_depth; ++d) {
					const int64_t val = (d < (int64_t)waiting[n].size()) ? waiting[n][(size_t)d] : 0;
					features.Add(val);
				}
			}

			// Include derived rates (these DO change with state in your implementation)
			features.Add(state.queue_manager.total_arrival_rate);
			features.Add(state.queue_manager.total_tick_rate);

			// ----- (3) Server state (busy_on flattened) -----
			// busy_on[k][j] where j indexes can_serve slots for server type k
			const auto& busy_on = state.server_manager.busy_on;

			int64_t total_busy_all = 0;
			for (size_t k = 0; k < busy_on.size(); ++k) {
				int64_t total_busy_k = 0;
				for (size_t j = 0; j < busy_on[k].size(); ++j) {
					features.Add(busy_on[k][j]);
					total_busy_k += busy_on[k][j];
				}
				// Helpful aggregates per server type
				features.Add(total_busy_k);
				total_busy_all += total_busy_k;
			}
			// Global aggregate
			features.Add(total_busy_all);

			// Also include the server-side derived rate (you maintain this)
			features.Add(state.server_manager.total_service_rate);

			// ----- (4) Action-list state (this is CRUCIAL for learning "defer") -----
			const int64_t qsize = static_cast<int64_t>(state.server_manager.action_queue.size());
			const int64_t acnt = state.server_manager.get_action_counter();

			features.Add(qsize);
			features.Add(acnt);
			// normalized progress through the list (avoid div-by-0)
			features.Add(qsize > 0 ? static_cast<double>(acnt) / static_cast<double>(qsize) : 0.0);

			// ----- (5) Current candidate (server, job) -----
			if (qsize <= 0 || acnt < 0 || acnt >= qsize) {
				features.Add(-1); // no server
				features.Add(-1); // no job
			}
			else {
				const Action& current_action = state.server_manager.action_queue.at(static_cast<size_t>(acnt));
				features.Add(current_action.server_index);
				features.Add(current_action.job_type);
			}
		}


		void MDP::RegisterPolicies(DynaPlex::Erasure::PolicyRegistry<MDP>& registry) const
		{//Here, we register any custom heuristics we want to provide for this MDP.	
		 //On the generic DynaPlex::MDP constructed from this, these heuristics can be obtained
		 //in generic form using mdp->GetPolicy(VarGroup vars), with the id in var set
		 //to the corresponding id given below.
			registry.Register<FIFOPolicy>("FIFO policy",
				"First in first out policy, always assigns a job to a server.");
			registry.Register<RVI_optimal>("RVI_optimal",
				"Optimal average-cost policy via Relative Value Iteration.");
			registry.Register<StochasticFIFOPolicy>("stochastic_FIFO",
				"FIFO with probabilistic skipping: draws[acnt] < threshold -> skip (action=0), "
				"else assign (action=1).  threshold=0.0 equals plain FIFO.");
			registry.Register<CmuPolicy>("cmu",
				"c-mu scheduling policy: assigns the highest c*mu job type for the current server. "
				"Skips a candidate if a remaining candidate for the same server has strictly higher c*mu. "
				"Ties broken by FIL (FIFO order).");
		}

		DynaPlex::StateCategory MDP::GetStateCategory(const State& state) const
		{
			//this typically works, but state.cat must be kept up-to-date when modifying states. 
			return state.cat;
		}	

		bool MDP::IsAllowedAction(const State& state, int64_t action) const {
		//std::cout << "[QMDP] IsAllowedAction called action=" << action << "\n";

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


		// -----------------------------------------------------------------------
		// EvaluatePolicyPerStep
		// Counts every call to IncorporateEvent as one uniformized step so that
		// the returned average cost per step matches g_star from runRVI().
		// -----------------------------------------------------------------------
		DynaPlex::VarGroup EvaluatePolicyPerStep(
			const DynaPlex::MDP&    mdp,
			const DynaPlex::Policy& policy,
			int64_t n_trajectories,
			int64_t steps_per_traj,
			int64_t warmup_steps,
			int64_t rng_seed)
		{
			std::vector<double> avg_costs(n_trajectories);

			for (int64_t i = 0; i < n_trajectories; ++i)
			{
				DynaPlex::Trajectory traj;
				// Give each trajectory its own independent RNG stream
				traj.RNGProvider.SeedEventStreams(true, rng_seed, 0, i);

				auto single = std::span<DynaPlex::Trajectory>(&traj, 1);
				mdp->InitiateState(single);


				// --- warm-up phase: count every MDP transition (action OR event) as one step ---
				// This matches g_star's denominator: total transitions = n_actions + n_events.
				for (int64_t s = 0; s < warmup_steps; ++s)
				{
					if (traj.Category.IsAwaitAction())
						mdp->IncorporateAction(single, policy);
					else
						mdp->IncorporateEvent(single);
				}
				double baseline = traj.CumulativeReturn;

				// --- main evaluation phase: same mixed action/event counting ---
				for (int64_t s = 0; s < steps_per_traj; ++s)
				{
					if (traj.Category.IsAwaitAction())
						mdp->IncorporateAction(single, policy);
					else
						mdp->IncorporateEvent(single);
				}

				avg_costs[i] = (traj.CumulativeReturn - baseline)
				               / static_cast<double>(steps_per_traj);
			}

			// Compute mean and standard error across trajectories
			double mean = 0.0;
			for (double c : avg_costs) mean += c;
			mean /= static_cast<double>(n_trajectories);

			double var = 0.0;
			for (double c : avg_costs) var += (c - mean) * (c - mean);
			if (n_trajectories > 1)
				var /= static_cast<double>(n_trajectories - 1);
			double std_error = std::sqrt(var / static_cast<double>(n_trajectories));

			DynaPlex::VarGroup result;
			result.Add("mean",                 mean);
			result.Add("std_error",            std_error);
			result.Add("n_trajectories",       n_trajectories);
			result.Add("steps_per_trajectory", steps_per_traj);
			result.Add("warmup_steps",         warmup_steps);
			return result;
		}

		// -----------------------------------------------------------------------
		// EvaluatePolicyRaw
		// Simulates the policy at the raw MDP level so we can classify every step
		// as action / real_event / fil_refresh exactly as the RVI sees them.
		// Accepts a std::function so the caller can inject any policy logic.
		// -----------------------------------------------------------------------
		RawEvalResult EvaluatePolicyRaw(
			const MDP&                                    mdp,
			std::function<int64_t(const MDP::State&)>     get_action,
			int64_t n_trajectories,
			int64_t steps_per_traj,
			int64_t warmup_steps,
			int64_t rng_seed)
		{
			std::vector<double> costs_per_rvi_step(n_trajectories, 0.0);
			std::vector<double> costs_per_rvi_step_rvi(n_trajectories, 0.0);
			std::vector<double> costs_per_step_gic(n_trajectories, 0.0);
			int64_t grand_action_steps      = 0;
			int64_t grand_real_event_steps  = 0;
			int64_t grand_fil_refresh_steps = 0;

			for (int64_t i = 0; i < n_trajectories; ++i)
			{
				DynaPlex::RNGProvider rng_provider;
				rng_provider.SeedEventStreams(true, rng_seed, 0, i);

				MDP::State state = mdp.GetInitialState();

				int64_t action_steps      = 0;
				int64_t real_event_steps  = 0;
				int64_t fil_refresh_steps = 0;
				double  cumcost           = 0.0;
				double  cumcost_rvi       = 0.0;
				double  cumcost_gic       = 0.0;

				// --- warm-up phase ---
				for (int64_t s = 0; s < warmup_steps; ++s) {
					if (state.cat == DynaPlex::StateCategory::AwaitAction()) {
						int64_t a = get_action(state);
						mdp.ModifyStateWithAction(state, a);
					}
					else {
						MDP::Event evt = mdp.GetEvent(rng_provider.GetEventRNG(0));
						mdp.ModifyStateWithEvent(state, evt);
					}
				}

				// --- main phase ---
				cumcost = 0.0;

				for (int64_t s = 0; s < steps_per_traj; ++s) {
					if (state.cat == DynaPlex::StateCategory::AwaitAction()) {
						int64_t a = get_action(state);
						mdp.ModifyStateWithAction(state, a);
						++action_steps;
					}
					else {
						bool is_fil_refresh = (state.next_fil_job_type != -1);
						// RVI-style: charge per-step at AwaitEvent states with FIL > due_time
						// (before applying the event, using the current state's FIL — same as RVI)
						if (!is_fil_refresh) {
							double rvi_step_cost = 0.0;
							for (int64_t n = 0; n < mdp.n_jobs; ++n)
								if (!state.queue_manager.waiting[(size_t)n].empty() &&
								    state.queue_manager.waiting[(size_t)n].front() > (int64_t)mdp.due_times[(size_t)n])
									rvi_step_cost += mdp.cost_rates[(size_t)n];
							cumcost_rvi += (mdp.tick_rate / mdp.uniformization_rate) * rvi_step_cost;
						}
						// GetImmediateCost-based charge: matches the RVI Bellman cost exactly.
						// Only charged at NON-FIL-refresh AwaitEvent states, because the RVI
						// chain has no FIL-refresh states (action=1 uses NextFILDistribution
						// to jump directly to the post-FIL state without an extra chain step).
						// Divide by rvi_steps (= action + real_event) — not all_steps — for the
						// same reason: the denominator of g* is the number of RVI chain steps.
						if (!is_fil_refresh)
							cumcost_gic += mdp.GetImmediateCost(state);
						MDP::Event evt = mdp.GetEvent(rng_provider.GetEventRNG(0));
						double cost = mdp.ModifyStateWithEvent(state, evt);
						cumcost += cost;
						if (is_fil_refresh)
							++fil_refresh_steps;
						else
							++real_event_steps;
					}
				}

				int64_t rvi_steps = action_steps + real_event_steps;
				costs_per_rvi_step[i] = (rvi_steps > 0)
					? cumcost / static_cast<double>(rvi_steps)
					: 0.0;
				costs_per_rvi_step_rvi[i] = (rvi_steps > 0)
					? cumcost_rvi / static_cast<double>(rvi_steps)
					: 0.0;
				// Divide by rvi_steps (action + real_event, same as RVI chain denominator) -> matches g*
				costs_per_step_gic[i] = (rvi_steps > 0)
					? cumcost_gic / static_cast<double>(rvi_steps)
					: 0.0;

				grand_action_steps      += action_steps;
				grand_real_event_steps  += real_event_steps;
				grand_fil_refresh_steps += fil_refresh_steps;
			}

			// mean and std error across trajectories
			double mean = 0.0;
			for (double c : costs_per_rvi_step) mean += c;
			mean /= static_cast<double>(n_trajectories);

			double mean_rvi = 0.0;
			for (double c : costs_per_rvi_step_rvi) mean_rvi += c;
			mean_rvi /= static_cast<double>(n_trajectories);

			double var = 0.0;
			for (double c : costs_per_rvi_step) var += (c - mean) * (c - mean);
			if (n_trajectories > 1)
				var /= static_cast<double>(n_trajectories - 1);
			double std_error = std::sqrt(var / static_cast<double>(n_trajectories));

			// mean cost per event (what the comparer measures)
			double total_cost = 0.0;
			for (double c : costs_per_rvi_step) total_cost += c;   // approximate
			double mean_cost_per_event = (grand_real_event_steps > 0)
				? (mean * static_cast<double>(grand_action_steps + grand_real_event_steps))
				  / static_cast<double>(grand_real_event_steps)
				: 0.0;

			double mean_gic = 0.0;
			for (double c : costs_per_step_gic) mean_gic += c;
			mean_gic /= static_cast<double>(n_trajectories);

			RawEvalResult result;
			result.mean_cost_per_rvi_step     = mean;
			result.mean_cost_per_rvi_step_rvi = mean_rvi;
			result.mean_cost_per_event        = mean_cost_per_event;
			result.mean_cost_per_step_gic     = mean_gic;
			result.std_error                  = std_error;
			result.total_action_steps         = grand_action_steps;
			result.total_real_event_steps     = grand_real_event_steps;
			result.total_fil_refresh_steps    = grand_fil_refresh_steps;
			return result;
		}

		// -----------------------------------------------------------------------
		// EvaluatePolicyRaw (DynaPlex::Policy overload)
		// Bridges a DynaPlex::Policy into the std::function interface above.
		//
		// Key implementation notes:
		//  1. The Trajectory is heap-allocated to avoid stack-size issues when
		//     called deep in a call chain (e.g. after DCL training).
		//  2. RNGProvider must be seeded before any SetAction call; without seeding,
		//     policies that use GetAction(state, RNG&) throw "empty provider" errors.
		//  3. The concrete MDP::State is overwritten in-place on every call so we
		//     avoid a heap allocation per step while keeping the state adapter valid.
		// -----------------------------------------------------------------------
		RawEvalResult EvaluatePolicyRaw(
			const MDP&              mdp,
			const DynaPlex::Policy& policy,
			int64_t n_trajectories,
			int64_t steps_per_traj,
			int64_t warmup_steps,
			int64_t rng_seed)
		{
			// Heap-allocated; seeded so GetPolicyRNG() doesn't throw for random-type policies.
			auto action_traj = std::make_unique<DynaPlex::Trajectory>();
			action_traj->RNGProvider.SeedEventStreams(false, rng_seed);
			action_traj->Category = DynaPlex::StateCategory::AwaitAction();
			action_traj->Reset(
				std::make_unique<DynaPlex::Erasure::StateAdapter<MDP::State>>(
					mdp.int_hash, mdp.GetInitialState()));

			auto get_action = [&policy, &action_traj](const MDP::State& s) -> int64_t {
				// Overwrite the stored state in-place — no per-call allocation.
				auto* adapter = static_cast<DynaPlex::Erasure::StateAdapter<MDP::State>*>(
					action_traj->GetState().get());
				adapter->state = s;
				action_traj->Category = DynaPlex::StateCategory::AwaitAction();
				policy->SetAction(std::span<DynaPlex::Trajectory>(action_traj.get(), 1));
				return action_traj->NextAction;
			};

			return EvaluatePolicyRaw(mdp, std::function<int64_t(const MDP::State&)>(get_action),
				n_trajectories, steps_per_traj, warmup_steps, rng_seed);
		}

	// -----------------------------------------------------------------------
		// EvaluatePolicyRawParallel
		// Parallel version: splits n_trajectories across num_threads using
		// DynaPlex::Parallel::parallel_compute (std::jthread-based, no OpenMP).
		//
		// Each thread owns one heap-allocated Trajectory for action queries,
		// created once per thread and reused across all trajectories in its slice.
		// The per-trajectory simulation RNG is seeded by global trajectory index i,
		// so the statistical results are equivalent to the serial version.
		// -----------------------------------------------------------------------
		RawEvalResult EvaluatePolicyRawParallel(
			const MDP&              mdp,
			const DynaPlex::Policy& policy,
			int64_t n_trajectories,
			int64_t steps_per_traj,
			int64_t warmup_steps,
			int64_t rng_seed,
			int64_t num_threads)
		{
			if (num_threads <= 0)
				num_threads = (int64_t)std::thread::hardware_concurrency();

			// Per-trajectory result collected inside the parallel region.
			struct PerTrajResult {
				double  cost_rvi     = 0.0;   // cumcost / rvi_steps
				double  cost_rvi2    = 0.0;   // cumcost_rvi / rvi_steps
				double  cost_gic     = 0.0;   // cumcost_gic / rvi_steps
				int64_t action_steps = 0;
				int64_t event_steps  = 0;
				int64_t fil_steps    = 0;
			};
			std::vector<PerTrajResult> results((size_t)n_trajectories);

			// work(span, offset): called once per thread.
			// offset = global index of span[0]; span.size() = number of trajectories for this thread.
			auto work = [&](std::span<PerTrajResult> span, int64_t offset) {

				// One action trajectory per thread — heap-allocated, seeded once.
				auto action_traj = std::make_unique<DynaPlex::Trajectory>();
				action_traj->RNGProvider.SeedEventStreams(false, rng_seed, 0, offset);
				action_traj->Category = DynaPlex::StateCategory::AwaitAction();
				action_traj->Reset(
					std::make_unique<DynaPlex::Erasure::StateAdapter<MDP::State>>(
						mdp.int_hash, mdp.GetInitialState()));

				auto get_action = [&policy, &action_traj](const MDP::State& s) -> int64_t {
					auto* adapter = static_cast<DynaPlex::Erasure::StateAdapter<MDP::State>*>(
						action_traj->GetState().get());
					adapter->state = s;
					action_traj->Category = DynaPlex::StateCategory::AwaitAction();
					policy->SetAction(std::span<DynaPlex::Trajectory>(action_traj.get(), 1));
					return action_traj->NextAction;
				};

				for (int64_t j = 0; j < (int64_t)span.size(); ++j) {
					int64_t i = offset + j;   // global trajectory index

					// Per-trajectory simulation RNG — keyed on i for reproducibility.
					DynaPlex::RNGProvider rng_provider;
					rng_provider.SeedEventStreams(true, rng_seed, 0, i);

					MDP::State state = mdp.GetInitialState();

					// --- warm-up ---
					for (int64_t s = 0; s < warmup_steps; ++s) {
						if (state.cat == DynaPlex::StateCategory::AwaitAction())
							mdp.ModifyStateWithAction(state, get_action(state));
						else {
							MDP::Event evt = mdp.GetEvent(rng_provider.GetEventRNG(0));
							mdp.ModifyStateWithEvent(state, evt);
						}
					}

					// --- main phase ---
					int64_t action_steps = 0, real_event_steps = 0, fil_refresh_steps = 0;
					double cumcost = 0.0, cumcost_rvi = 0.0, cumcost_gic = 0.0;

					for (int64_t s = 0; s < steps_per_traj; ++s) {
						if (state.cat == DynaPlex::StateCategory::AwaitAction()) {
							mdp.ModifyStateWithAction(state, get_action(state));
							++action_steps;
						} else {
							bool is_fil_refresh = (state.next_fil_job_type != -1);
							if (!is_fil_refresh) {
								double rvi_step_cost = 0.0;
								for (int64_t n = 0; n < mdp.n_jobs; ++n)
									if (!state.queue_manager.waiting[(size_t)n].empty() &&
									    state.queue_manager.waiting[(size_t)n].front() > (int64_t)mdp.due_times[(size_t)n])
										rvi_step_cost += mdp.cost_rates[(size_t)n];
								cumcost_rvi += (mdp.tick_rate / mdp.uniformization_rate) * rvi_step_cost;
								cumcost_gic += mdp.GetImmediateCost(state);
							}
							MDP::Event evt = mdp.GetEvent(rng_provider.GetEventRNG(0));
							double cost = mdp.ModifyStateWithEvent(state, evt);
							cumcost += cost;
							if (is_fil_refresh) ++fil_refresh_steps;
							else               ++real_event_steps;
						}
					}

					int64_t rvi_steps = action_steps + real_event_steps;
					span[j] = {
						rvi_steps > 0 ? cumcost     / (double)rvi_steps : 0.0,
						rvi_steps > 0 ? cumcost_rvi / (double)rvi_steps : 0.0,
						rvi_steps > 0 ? cumcost_gic / (double)rvi_steps : 0.0,
						action_steps, real_event_steps, fil_refresh_steps
					};
				}
			};

			DynaPlex::Parallel::parallel_compute<PerTrajResult>(
				results,
				std::function<void(std::span<PerTrajResult>, int64_t)>(work),
				num_threads);

			// --- reduce ---
			int64_t grand_action_steps      = 0;
			int64_t grand_real_event_steps  = 0;
			int64_t grand_fil_refresh_steps = 0;
			for (auto& r : results) {
				grand_action_steps      += r.action_steps;
				grand_real_event_steps  += r.event_steps;
				grand_fil_refresh_steps += r.fil_steps;
			}

			double mean = 0.0, mean_rvi = 0.0, mean_gic = 0.0;
			for (auto& r : results) { mean += r.cost_rvi; mean_rvi += r.cost_rvi2; mean_gic += r.cost_gic; }
			mean     /= (double)n_trajectories;
			mean_rvi /= (double)n_trajectories;
			mean_gic /= (double)n_trajectories;

			double var = 0.0;
			for (auto& r : results) var += (r.cost_rvi - mean) * (r.cost_rvi - mean);
			if (n_trajectories > 1) var /= (double)(n_trajectories - 1);
			double std_error = std::sqrt(var / (double)n_trajectories);

			double mean_cost_per_event = (grand_real_event_steps > 0)
				? (mean * (double)(grand_action_steps + grand_real_event_steps))
				  / (double)grand_real_event_steps
				: 0.0;

			RawEvalResult result;
			result.mean_cost_per_rvi_step     = mean;
			result.mean_cost_per_rvi_step_rvi = mean_rvi;
			result.mean_cost_per_event        = mean_cost_per_event;
			result.mean_cost_per_step_gic     = mean_gic;
			result.std_error                  = std_error;
			result.total_action_steps         = grand_action_steps;
			result.total_real_event_steps     = grand_real_event_steps;
			result.total_fil_refresh_steps    = grand_fil_refresh_steps;
			return result;
		}

		// -----------------------------------------------------------------------
		// PrintPolicyHeatmap
		// Simulation-based: samples canonical AwaitAction states (action_counter==0,
		// both job types waiting, exactly 1 server busy) and records which job type
		// the policy assigns.  Works for any DynaPlex::Policy (RVI, FIFO, NN, ...).
		// -----------------------------------------------------------------------
		void PrintPolicyHeatmap(
			const DynaPlex::MDP&    fw_mdp,
			const DynaPlex::Policy& policy,
			int     max_fil,
			int64_t n_warmup,
			int64_t n_samples)
		{
			// -2 = not visited, -1 = skip/idle, 0 = type 0, 1 = type 1
			std::vector<std::vector<int>> grid(
				max_fil + 1, std::vector<int>(max_fil + 1, -2));

			DynaPlex::Trajectory traj;
			traj.RNGProvider.SeedEventStreams(true, 42, 0, 0);
			auto single = std::span<DynaPlex::Trajectory>(&traj, 1);
			fw_mdp->InitiateState(single);

			// Warm-up
			for (int64_t s = 0; s < n_warmup; ++s) {
				if (traj.Category.IsAwaitAction())
					fw_mdp->IncorporateAction(single, policy);
				else
					fw_mdp->IncorporateEvent(single);
			}

			// Sample collection
			for (int64_t s = 0; s < n_samples; ++s) {
				if (traj.Category.IsAwaitAction()) {
					MDP::State& sp = DynaPlex::RetrieveState<MDP::State>(traj.GetState());

					if (sp.server_manager.action_counter == 0) {
						const auto fil = sp.queue_manager.get_FIL_waiting();
						int64_t f0 = fil[0];
						int64_t f1 = fil[1];

						if (f0 >= 0 && f1 >= 0 && f0 <= max_fil && f1 <= max_fil
							&& grid[f0][f1] == -2)
						{
							// Canonical filter: exactly 1 server busy across all pools
							int busy = 0;
							for (const auto& row : sp.server_manager.busy_on)
								for (int64_t b : row) busy += (int)b;

							if (busy == 1) {
								policy->SetAction(single);   // sets traj.NextAction
								int64_t action = traj.NextAction;
								if (action == 1) {
									grid[f0][f1] = (int)sp.server_manager.action_queue[0].job_type;
								} else {
									grid[f0][f1] = -1;  // skipped highest-FIL job
								}
								fw_mdp->IncorporateAction(single);  // apply already-set action
								continue;
							}
						}
					}
					fw_mdp->IncorporateAction(single, policy);
				} else {
					fw_mdp->IncorporateEvent(single);
				}
			}

			// Print header
			std::cout << "\n      FIL_1:";
			for (int f1 = 0; f1 <= max_fil; ++f1)
				std::cout << std::setw(3) << f1;
			std::cout << "\nFIL_0\n";

			for (int f0 = 0; f0 <= max_fil; ++f0) {
				std::cout << std::setw(7) << f0 << " :";
				for (int f1 = 0; f1 <= max_fil; ++f1) {
					int v = grid[f0][f1];
					if      (v == -2) std::cout << "  -";
					else if (v == -1) std::cout << "  .";
					else              std::cout << "  " << v;
				}
				std::cout << "\n";
			}
			std::cout << "\nLegend: 0=serve type 0, 1=serve type 1, .=skip/idle, -=not visited\n";
			std::cout << "FIFO boundary: diagonal\n";
			std::cout << "  Below diagonal (FIL_0>FIL_1): FIFO serves type 0  -> expect '0'\n";
			std::cout << "  Above diagonal (FIL_1>FIL_0): FIFO serves type 1  -> expect '1'\n";
			std::cout << "  Deviations from diagonal pattern = RVI != FIFO\n";
		}

		// -----------------------------------------------------------------------
		// PrintEnumeratedHeatmap
		// Enumeration-based: directly constructs the canonical AwaitAction state
		// for every (FIL_0, FIL_1) cell and queries the supplied policy function.
		// No simulation -> no '-' (not-visited) artifacts.
		//
		// Canonical state: server pool 0 is busy on job type 0 (1 server busy),
		// pool 1 idle, both job types have a job waiting (FIL_0=f0, FIL_1=f1),
		// action_counter=0.  This matches the states a simulation would sample
		// most often when recording the first scheduling decision.
		// -----------------------------------------------------------------------
		void PrintEnumeratedHeatmap(
			const MDP& mdp,
			std::function<int64_t(const MDP::State&)> policy_fn,
			int max_fil)
		{
			// grid[f0][f1]:  -2 = no valid action (shouldn't occur)
			//                -1 = skip/idle (action=0, '.')
			//                 0 = serve type 0
			//                 1 = serve type 1
			std::vector<std::vector<int>> grid(
				max_fil + 1, std::vector<int>(max_fil + 1, -2));

			for (int f0 = 0; f0 <= max_fil; ++f0) {
				for (int f1 = 0; f1 <= max_fil; ++f1) {
					// ---- Construct canonical AwaitAction state ----
					MDP::State s;

					// Queue: FIL_0 = f0, FIL_1 = f1
					s.queue_manager.initialize(
						mdp.n_jobs, mdp.tick_rate, mdp.arrival_rates, mdp.max_queue_depth);
					s.queue_manager.set_fil(0, (int64_t)f0);
					s.queue_manager.set_fil(1, (int64_t)f1);

					// Servers: pool 0 busy on type 0, pool 1 idle
					s.server_manager.initialize(&mdp.server_static_info, mdp.n_jobs);
					// busy_on[pool][canServeIndex(pool,job)]:
					// for fully-flexible pools, can_serve={0,1} -> index 0 = job type 0
					s.server_manager.busy_on[0][0] = 1;

					// Generate the action queue and reset counter
					s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
					s.server_manager.set_action_counter(0);
					s.server_manager.update_total_service_rate();

					s.next_fil_job_type = -1;
					s.cat = DynaPlex::StateCategory::AwaitAction();

					if (s.server_manager.action_queue.empty()) {
						// No idle capacity -> nothing to assign (mark with '*')
						grid[f0][f1] = -2;
						continue;
					}

					// ---- Query policy ----
					int64_t action = policy_fn(s);

					if (action == 0) {
						grid[f0][f1] = -1;  // skip top candidate
					} else {
						// action == 1: assign action_queue[0]; record its job type
						grid[f0][f1] = (int)s.server_manager.action_queue[0].job_type;
					}
				}
			}

			// ---- Print header ----
			std::cout << "\n      FIL_1:";
			for (int f1 = 0; f1 <= max_fil; ++f1)
				std::cout << std::setw(3) << f1;
			std::cout << "\nFIL_0\n";

			for (int f0 = 0; f0 <= max_fil; ++f0) {
				std::cout << std::setw(7) << f0 << " :";
				for (int f1 = 0; f1 <= max_fil; ++f1) {
					int v = grid[f0][f1];
					if      (v == -2) std::cout << "  *";   // no valid action
					else if (v == -1) std::cout << "  .";   // skip
					else              std::cout << "  " << v;
				}
				std::cout << "\n";
			}
			std::cout << "\nLegend: 0=serve type 0, 1=serve type 1, .=skip/idle, *=no valid action\n";
			std::cout << "FIFO boundary: diagonal\n";
			std::cout << "  Below diagonal (FIL_0>FIL_1): FIFO serves type 0  -> expect '0'\n";
			std::cout << "  Above diagonal (FIL_1>FIL_0): FIFO serves type 1  -> expect '1'\n";
			std::cout << "  Deviations from FIFO pattern = RVI overrides (type-0 priority)\n";
		}

		// -----------------------------------------------------------------------
		// PrintEnumeratedGapHeatmap
		// For every (FIL_0, FIL_1) canonical state, prints the action-value gap
		//   gap(s) = |Q(s,a=0) - Q(s,a=1)|
		// stored in sol.gap_map.  Each cell shows floor(log10(gap)) so the range
		// of the table spans ~6 decades in two characters per cell:
		//   -6 .. -1 : near-tie  (decision unreliable, potential numerical noise)
		//    0 ..  2  : decisive  (one action is clearly better)
		//    ?        : gap not available (state not in map)
		// -----------------------------------------------------------------------
		void PrintEnumeratedGapHeatmap(
			const MDP&              mdp,
			const MDP::RVISolution& sol,
			int                     max_fil)
		{
			// gap_grid[f0][f1] stores the gap value (-1.0 = not available)
			std::vector<std::vector<double>> gap_grid(
				max_fil + 1, std::vector<double>(max_fil + 1, -1.0));

			for (int f0 = 0; f0 <= max_fil; ++f0) {
				for (int f1 = 0; f1 <= max_fil; ++f1) {
					// Build the same canonical state as PrintEnumeratedHeatmap
					MDP::State s;
					s.queue_manager.initialize(
						mdp.n_jobs, mdp.tick_rate, mdp.arrival_rates, mdp.max_queue_depth);
					s.queue_manager.set_fil(0, (int64_t)f0);
					s.queue_manager.set_fil(1, (int64_t)f1);
					s.server_manager.initialize(&mdp.server_static_info, mdp.n_jobs);
					s.server_manager.busy_on[0][0] = 1;
					s.server_manager.generate_actions(s.queue_manager.get_FIL_waiting());
					s.server_manager.set_action_counter(0);
					s.server_manager.update_total_service_rate();
					s.next_fil_job_type = -1;
					s.cat = DynaPlex::StateCategory::AwaitAction();

					if (s.server_manager.action_queue.empty()) continue;

					gap_grid[f0][f1] = mdp.EvaluateRVIGap(sol, s);
				}
			}

			// Compute max gap for reference
			double max_gap = 0.0;
			for (auto& row : gap_grid)
				for (double g : row)
					if (g > max_gap) max_gap = g;

			// Print header
			std::cout << "\n  Action-value gap heatmap  |Q(s,0)-Q(s,1)|"
			          << "  (cell = floor(log10(gap)), '?' = not available)\n";
			std::cout << "  Max gap = " << std::scientific << std::setprecision(3)
			          << max_gap << "\n";
			std::cout << "\n      FIL_1:";
			for (int f1 = 0; f1 <= max_fil; ++f1)
				std::cout << std::setw(3) << f1;
			std::cout << "\nFIL_0\n";

			for (int f0 = 0; f0 <= max_fil; ++f0) {
				std::cout << std::setw(7) << f0 << " :";
				for (int f1 = 0; f1 <= max_fil; ++f1) {
					double g = gap_grid[f0][f1];
					if (g < 0.0) {
						std::cout << "  ?";
					} else if (g == 0.0) {
						std::cout << " -∞";
					} else {
						int lg = (int)std::floor(std::log10(g));
						// Clamp to [-9, +9] for display
						if (lg < -9) lg = -9;
						if (lg >  9) lg =  9;
						std::cout << std::setw(3) << lg;
					}
				}
				std::cout << "\n";
			}
			std::cout << "\nLegend: cell = floor(log10(|Q(s,0)-Q(s,1)|))\n"
			          << "  Large negative (e.g. -5): near-tie, decision may be noise\n"
			          << "  Close to 0 or positive  : decisive, decision is robust\n"
			          << "  ?                       : gap not stored (state outside BFS)\n";
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

