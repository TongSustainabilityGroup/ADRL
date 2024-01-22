import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

class PolicyOpt:
    def __init__(self, env, param_file):

        self.num_states = env._num_states
        self.num_actions = env._num_actions
        self.q = np.zeros((self.num_states, self.num_actions))
        self.pg = np.zeros((self.num_states, self.num_actions))
        self.policy = np.ones((self.num_states, self.num_actions)) / self.num_actions
        self.env = env
        self.gamma = self.env.gamma
        self.mu = self.env.mu
        self.param_file = param_file
        self.u0 = None

    def update_policy(self):
        pass

    def ctd(self):
        pass

    def generate_trajectory(self, state_action_j, length, tau=0, use_tqdm=False):
        trajectory = []
        state_idx, action_idx, j = state_action_j
        ssa_pair = None
        timesteps = range(length) if (not use_tqdm) else tqdm(range(length))
        
        for t in timesteps:
            if t > 0:
                action_idx = self.make_action(state_idx, sig=0)
            next_state_idx, reward, self.u0 = self.env.generate_experience(
                state_idx, action_idx, int(j*length+t), self.u0)
            if tau > 0:
                reward -= tau * \
                    (entropy(self.policy[state_idx, :]) +
                     np.log(self.num_actions))
            if ssa_pair is not None:
                pair = ssa_pair + [action_idx]
                trajectory.append(pair)
            ssa_pair = [state_idx, action_idx, reward, next_state_idx]

            state_idx = next_state_idx
        return trajectory

    def prox_update(self, pi, G, eta, tau):
        logits = (np.log(pi) - eta * G) / ((self.mu + tau) * eta + 1)
        logits -= logits.max()
        logits = np.clip(logits, -32, 32)
        p = np.nan_to_num(np.exp(logits), nan=0) + 1e-15
        p /= np.sum(p)
        p = np.nan_to_num(p, nan=0)
        return p

    def make_action(self, state_idx, sig):
        A = self.policy[state_idx]
        action_idx = np.random.choice(np.arange(self.num_actions), p=A)  
        return action_idx

    def read_params(self):
        params = self.param_file
        return params


class PMD(PolicyOpt):
    def ctd(self, T=1, alpha=1, lr=1.0, tau=0, **kwargs):
        state_idx = 1098 if self.u0 is None else self.last_state_idx
        action_idx = self.make_action(state_idx, sig=0)
        trajectory = self.generate_trajectory((state_idx, action_idx, 0), T * alpha, tau, use_tqdm=False)

        for t, (state_idx, action_idx, reward, next_state_idx, next_action_idx) in enumerate(trajectory):
            if t % alpha != 0:
                continue
            self.q[state_idx, action_idx] = self.q[state_idx, action_idx] - lr * \
                (self.q[state_idx, action_idx] - reward -
                 self.gamma * self.q[next_state_idx, next_action_idx])
            self.last_state_idx = next_state_idx

    def update_policy(self, eta=1.0, tau=0.0, **kwargs):
        for state_idx in range(self.num_states):
            self.policy[state_idx, :] = self.prox_update(
                self.policy[state_idx, :], self.q[state_idx, :], eta, tau)
    
    def optimize(self, writer=None):
        params = self.read_params()
        steps = params['steps']
        eval_name = params['eval_method']
        methods = {'ctd': self.ctd,}
        eval_method = methods[eval_name]

        for step in range(1, steps + 1):
            self.params = params
            self.env.generate_disturbance()
            eval_method(**params)
            self.update_policy(**params)