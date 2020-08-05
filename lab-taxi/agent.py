import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.discount = 0.8
        self.alpha = 0.07
        self.epsilon = 0.01

    def epsilon_greedy_probs(self,Q_s):
    #""" obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = self.epsilon
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action_types = [i for i in range(0, self.nA)]
        policy_s = self.epsilon_greedy_probs(self.Q[state])
        # pick action A
        action = np.random.choice(np.arange(self.nA), p=policy_s) if state in self.Q else random.choice(action_types)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(next_state)
        self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + (self.discount * self.Q[next_state][next_action]) - self.Q[state][action]))
        state = next_state
        action = next_action
