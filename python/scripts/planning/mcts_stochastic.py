import random

import numpy as np

from scripts.planning.helpers import stable_normalizer


class State():
    ''' State object '''

    def __init__(self, obs, r, terminal, parent_action):
        ''' Initialize a new state '''
        self.obs = obs      # State observation containing the features
        self.r = r          # Reward upon arriving in this state
        self.terminal = terminal  # Whether the domain terminated in this state
        self.n = 0   # Number of visits to the state

        self.parent_action = parent_action

        # Child actions
        self.allowed_actions = np.where(self.obs['mask'])[0]
        self.child_actions = []

        self.V = None

    def add_child_actions(self, q_init):
        self.child_actions = [Action(a, parent_state=self, Q_init=q_init) for a in self.allowed_actions]

        return self

    def select(self, c=1.5):
        ''' Select one of the child actions based on UCT rule '''

        child_Q_values = np.array([child_action.Q for child_action in self.child_actions])

        n_values = np.array([child_action.n for child_action in self.child_actions])
        sqrt_terms = np.sqrt((2 * np.log(self.n + 1)) / (n_values + 1))
        UCT = child_Q_values + c * sqrt_terms

        winner = UCT.argmax()

        return self.child_actions[winner]

    def evaluate(self, rollout_env, traj, policy, max_len):
        ''' Bootstrap the state value '''

        if self.terminal:
            self.V = 0.0
            return

        rollout_env.run_until_period_count(policy, traj, max_len)

        self.V = - traj.cumulative_return

    def update(self):
        ''' update count on backward pass '''
        self.n += 1


class ChanceNode():
    ''' Chance Node object '''
    ''' Contains the rng seed for the trajectory that will be followed thereafter '''
    ''' and pointers to the parent action and future state '''

    def __init__(self, rng_seed, parent_action):
        self.rng_seed = rng_seed
        self.parent_action = parent_action
        self.child_state = None

    def add_child_state(self, obs, r: float, terminal: bool):
        state = State(obs, r, terminal, parent_action=self.parent_action)
        self.child_state = state
        return state


class Action():
    ''' Action object '''

    def __init__(self, index, parent_state, Q_init=0.0):
        self.index = index  # Index of the action in the Discrete action space
        self.W = 0.0        # Total accumulated reward during the search
        self.n = 0          # Number of visits to the action node
        self.Q = Q_init     # Initial Q value of the action

        # Actions can only have one parent state
        self.parent_state = parent_state

        # Actions can have child states or child chance nodes, depending on whether the after applying it the environment
        # evolves stochastically or not (leads directly to another action without any random events happening)
        self.child_state = None
        self.child_nodes = {}

    def add_child_state(self, obs, r: float, terminal: bool):
        state = State(obs, r, terminal, parent_action=self)
        self.child_state = state
        return state

    def add_child_node(self, rng_seed: int):
        chance_node = ChanceNode(rng_seed, parent_action=self)
        self.child_nodes[rng_seed] = chance_node
        return chance_node

    def update(self, R):
        self.W += R
        self.n += 1
        self.Q = self.W / self.n


class MCTS():
    ''' MCTS object '''

    def __init__(self, root, root_obs, gamma, search_env, search_traj, rollout_env=None):
        self.root = None
        self.root_obs = root_obs
        self.gamma = gamma

        self.action_q_init = np.finfo(np.float32).max

        self.cum_return = 0.0

        self.search_env = search_env
        if rollout_env is not None:
            self.rollout_env = rollout_env
        else:
            self.rollout_env = self.search_env

        self.search_traj = search_traj

        self.n_other_actions = 0

    def set_random_seed(self, seed):
        random.seed(seed)

    def search(self, n_mcts, n_chance_nodes, c, policy, start_state, start_period, start_return, max_len=10000):
        ''' Perform the MCTS search from the root '''

        n_it = 0

        self.n_other_actions = 0
        for i in range(n_mcts):

            n_it += 1

            if self.root is None:
                # Initialize new root state

                r = - (start_return - self.cum_return)

                self.root = State(self.root_obs, r=r, terminal=False, parent_action=None)
                self.root.add_child_actions(self.action_q_init)  # Initialize root actions
            else:
                # Continue from current root state
                self.root.parent_action = None
                self.root.parent_state = None
            if self.root.terminal:
                raise (ValueError("Can't do tree search from a terminal state"))

            state = self.root  # reset to root for new trace

            # start_state is the outside trajectory
            self.search_env.initiate_state(self.search_traj, start_state)

            last_period = self.search_traj.period_count
            last_cum_return = self.cum_return

            while True:

                done = False

                if not done:
                    action = state.select(c=c)

                    last_action = action

                    self.search_traj.next_action = action.index
                    self.search_env.incorporate_action(self.search_traj)

                else:  # Done
                    break

                # Select child branch if max number of chance nodes is reached
                if len(last_action.child_nodes) == n_chance_nodes:
                    idx = random.randint(0, len(last_action.child_nodes) - 1)
                    node = list(last_action.child_nodes.values())[idx]
                    state = node.child_state

                    # Change rng_seed in the trajectory to the selected one and evolve
                    self.search_traj.seed_rngprovider(node.rng_seed)
                    self.search_env.incorporate_until_action(self.search_traj)

                else:
                    # Expand
                    # Evolve the trajectory based on a new random seed
                    rng_seed = random.randint(0, 1000000)
                    self.search_traj.seed_rngprovider(rng_seed)
                    self.search_env.incorporate_until_action(self.search_traj)

                    # Extract information from the environment
                    obs = {'obs': self.search_env.get_features(self.search_traj)[0],
                           'mask': self.search_env.get_mask(self.search_traj)[0]}

                    actual_cum_return = self.search_traj.cumulative_return
                    r = - (actual_cum_return - last_cum_return)

                    # Add chance node to last_action.child_nodes
                    node = last_action.add_child_node(rng_seed)

                    # Add new state as child of chance node
                    state = node.add_child_state(obs, r, done)

                    state = state.add_child_actions(self.action_q_init)
                    state.evaluate(self.rollout_env, self.search_traj, policy, max_len)

                    break

                # Not reached if a state was expanded, so no repetition
                actual_cum_return = self.search_traj.cumulative_return
                last_cum_return = actual_cum_return

            # Back-up
            if state.V is None:
                state = state.add_child_actions(self.action_q_init)
                state.evaluate(self.rollout_env, self.search_traj, policy, max_len)
            R = state.V

            while state.parent_action is not None:  # loop back-up until root is reached
                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()

    def return_results(self, temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])

        Q = np.array([child_action.Q for child_action in self.root.child_actions])

        if counts.sum() != 0:
            pi_target = stable_normalizer(counts, temp)
            V_target = np.sum((counts / np.sum(counts)) * Q)[None]
        else:
            pi_target = self.root.obs['mask'][self.root.allowed_actions] / len(self.root.allowed_actions)
            try:
                V_target = np.sum(pi_target * Q)[None]
            except:
                V_target = np.array([0.0])  # It should not be used anyway

        return pi_target, V_target

    def forward(self, a, s1):
        ''' Move the root forward '''

        self.cum_return += -self.root.r

        if self.root.child_actions[a].child_state is None:
            self.root = None
            self.root_obs = s1
        else:
            s = self.root.child_actions[a].child_state
            diff = np.linalg.norm(s.obs['obs'] - s1['obs']) + \
                   np.linalg.norm(s.obs['mask'] ^ s1['mask'])
            if diff < 0.001:
                self.root = self.root.child_actions[a].child_state
                if len(self.root.child_actions) == 0:
                    self.root.add_child_actions(self.action_q_init)  # Initialize root actions
            else:
                self.root = None
                self.root_obs = s1
