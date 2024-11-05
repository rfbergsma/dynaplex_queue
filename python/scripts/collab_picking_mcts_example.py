import json
from time import time

import numpy as np

from dp import dynaplex

from planning.mcts_stochastic import MCTS

# Example of using MCTS for the collaborative picking environment
if __name__ == '__main__':

    folder_name = "collaborative_picking"
    mdp_version_number = 8
    path_to_json = dynaplex.filepath("mdp_config_examples", folder_name, f"mdp_config_{mdp_version_number}.json")

    try:
        with open(path_to_json, "r") as input_file:
            vars = json.load(input_file)  # vars can be initialized manually with something like
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path_to_json} not found. Please make sure the file exists and try again.")
    except:
        raise Exception(
            "Something went wrong when loading the json file. Have you checked the json file does not contain any comment?")

    mdp = dynaplex.get_mdp(**vars)

    num_actions = mdp.get_static_info()['valid_actions']

    print("Environment created")

    baseline_policy = mdp.get_policy(id="closest_pair")  # Policy to compare against MCTS
    rollout_policy = mdp.get_policy(id="random")  # Policy to be used for MCTS rollouts

    traj = dynaplex.get_trajectory(42)  # Main trajectory object for the MCTS experiment

    search_traj = dynaplex.get_trajectory()  # Trajectory object for the MCTS search
    search_traj.seed_rngprovider()

    rewards = []
    startup_rewards = []
    heur_rewards = []
    search_times = []
    dec_times = []

    start_time = time()

    n_episodes = 10
    for ep in range(n_episodes):

        print(f"Episode {ep}")

        seed = ep

        # Create trajectory object
        traj.seed_rngprovider(seed)

        # Initiate state in the trajectory object
        mdp.initiate_state(traj)

        # Apply startup period to get to a meaningful state in the MDP
        startup_duration = vars['startup_duration']
        print(f"INITIAL STATE: {traj.state_as_dict()}")
        mdp.incorporate_until_nontrivial_action(traj)  # Necessary if the mdp starts in await_event state
        while traj.period_count < (startup_duration / vars['timestep_delta']):
            mdp.incorporate_action(traj, baseline_policy)  # Apply greedy heuristic policy
            mdp.incorporate_until_nontrivial_action(traj)
        print(f"POST STARTUP STATE: {traj.state_as_dict()}")

        startup_rewards.append(-traj.cumulative_return)

        heur_traj = mdp.deep_copy(traj)  # Trajectory object for the baseline heuristic experiment

        mcts = None
        last_action = None
        n_step = 0

        max_period_count = 2200  # timestep_delta = 10, 30 min + 6.66 min setup

        # To run at the same time MCTS and the baseline heuristic policy, we need to keep track of two separated period
        # counters, since taking different actions in the two trajectories can lead to different period counts.
        while traj.period_count < max_period_count or heur_traj.period_count < max_period_count:

            start_time_dec = time()

            if traj.period_count < max_period_count:
                obs = {'obs': mdp.get_features(traj)[0],
                       'mask': mdp.get_mask(traj)[0]}

                if mcts is None:
                    gamma = 0.99
                    mcts = MCTS(root_obs=obs, root=None, gamma=gamma, search_env=mdp,
                                rollout_env=mdp, search_traj=search_traj)
                    mcts.set_random_seed(seed)

                if n_step > 0:
                    mcts.forward(last_action, obs)

                n_chance_nodes = 10
                n_mcts = 250
                c = 0.25

                max_len = 30

                start_search_time = time()

                start_period = traj.period_count
                start_return = traj.cumulative_return

                mcts.search(n_mcts=n_mcts, n_chance_nodes=n_chance_nodes, c=c, policy=rollout_policy,
                            start_period=start_period, start_state=traj, start_return=start_return,
                            max_len=max_len)

                search_times.append(time() - start_search_time)

                pi, _ = mcts.return_results(temp=1.0)       # Extract probability distribution over actions
                child_action_idx = np.argmax(pi)            # Get id of action with highest probability, considering the mask
                action = mcts.root.child_actions[child_action_idx].index    # Get the global id of the action

                traj.next_action = action       # Set the next action in the trajectory
                mdp.incorporate_action(traj)    # Incorporate the action in the MDP

                last_action = child_action_idx  # Save the action id for the next iteration

            if heur_traj.period_count < max_period_count:
                mdp.incorporate_action(heur_traj, baseline_policy)  # Apply the baseline heuristic policy

            dec_times.append(time() - start_time_dec)

            n_step += 1

            # Evolve the trajectories, if episode is not finished
            if traj.period_count < max_period_count:
                mdp.incorporate_until_nontrivial_action(traj, max_period_count=max_period_count)
            if heur_traj.period_count < max_period_count:
                mdp.incorporate_until_nontrivial_action(heur_traj, max_period_count=max_period_count)

        rewards.append(-traj.cumulative_return)
        heur_rewards.append(-heur_traj.cumulative_return)

        state = traj.state_as_dict()
        heur_state = heur_traj.state_as_dict()

    eval_time = time() - start_time

    print()
    print(f"Total time: {eval_time}")

    print(f"Average baseline heuristic reward: {np.mean(heur_rewards).round(2)} +- {np.std(heur_rewards).round(2)}")
    print(f"Average MCTS reward: {np.mean(rewards).round(2)} +- {np.std(rewards).round(2)}")

    print()
    print(f"Average startup reward: {np.mean(startup_rewards).round(2)} +- {np.std(startup_rewards).round(2)}")

    useful_heur_rewards = np.array(heur_rewards) - np.array(startup_rewards)
    useful_rewards = np.array(rewards) - np.array(startup_rewards)

    print("Performance gain considering startup period: ",
          ((np.mean(useful_rewards) - np.mean(useful_heur_rewards)) / np.mean(useful_heur_rewards)).round(2))

    # print(f"Average search time: {np.mean(search_times)} +- {np.std(search_times)}, n searches: {len(search_times)}")
    # print(f"Average decision time: {np.mean(dec_times)} +- {np.std(dec_times)}, n decisions: {len(dec_times)}")