import os
import warnings
import pickle

import pandas as pd
import numpy as np
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F
from torch import nn

from .environ import Adapter


class ReplayBuffer:

    def __init__(self, max_len=1e6):
        self.max_len = max_len
        self.ptr = 0
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def add(self, transition):
        keys = {'state', 'action', 'reward', 'next_state', 'terminated'}
        assert set(transition.keys()) == keys
        if len(self) == self.max_len:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_len
        else:
            self.storage.append(transition)

    def sample(self, batch_size):

        sample = np.random.randint(0, len(self.storage), size=batch_size)

        batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminated = [], [], [], [], []

        for idx in sample:
            batch_states.append(self.storage[idx]['state'])
            batch_next_states.append(self.storage[idx]['next_state'])
            batch_actions.append(self.storage[idx]['action'])
            batch_rewards.append(self.storage[idx]['reward'])
            batch_terminated.append(self.storage[idx]['terminated'])

        return (
            np.vstack(batch_states),
            np.vstack(batch_next_states),
            np.vstack(batch_actions),
            np.vstack(batch_rewards),
            np.vstack(batch_terminated),
        )


class Value(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout=0.):

        super().__init__()

        self.dropout = dropout

        self.conv_1 = dglnn.GraphConv(state_dim, hidden_dim, allow_zero_in_degree=True)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.conv_2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.lin = nn.Linear(hidden_dim, action_dim)

    def forward(self, graph, feat):

        X = self.conv_1(graph, feat)
        X = F.dropout(self.bn_1(X).relu(), self.dropout)
        X = self.conv_2(graph, X)
        X = F.dropout(self.bn_2(X).relu(), self.dropout)
        X = self.lin(X)

        return X


class ValueAdvantage(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, dropout=0., agg='mean'):

        super().__init__()

        self.dropout = dropout
        self.agg = agg

        self.conv_1 = dglnn.GraphConv(state_dim, hidden_dim, allow_zero_in_degree=True)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.conv_2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.bn_2 = nn.BatchNorm1d(hidden_dim)
        self.adv_fn = nn.Linear(hidden_dim, action_dim)
        self.val_fn = nn.Linear(hidden_dim, 1)

    def forward(self, graph, feat):

        X = self.conv_1(graph, feat)
        X = F.dropout(self.bn_1(X).relu(), self.dropout)
        X = self.conv_2(graph, X)
        X = F.dropout(self.bn_2(X).relu(), self.dropout)

        adv = self.adv_fn(X)
        val = self.val_fn(X)
        if self.agg == 'mean':
            adv_agg = adv.mean(dim=1, keepdim=True)
        elif self.agg == 'max':
            adv_agg = adv.amax(dim=1, keepdim=True)
        else:
            raise ValueError

        Q = val + (adv - adv_agg)

        return Q


class DQN:

    def __init__(self, adapter, *,
                 hidden_dim=256, mem_size=1e6, eta=1e-3):

        super().__init__()

        self.adapter = adapter
        self.hidden_dim = hidden_dim
        self.mem_size = mem_size
        self.eta = eta

        # self.value_fn = Value(self.adapter.input_dim, self.adapter.output_dim, hidden_dim=self.hidden_dim)
        # self.value_fn_target = Value(self.adapter.input_dim, self.adapter.output_dim, hidden_dim=self.hidden_dim)
        self.value_fn = ValueAdvantage(self.adapter.input_dim, self.adapter.output_dim, hidden_dim=self.hidden_dim)
        self.value_fn_target = ValueAdvantage(self.adapter.input_dim, self.adapter.output_dim, hidden_dim=self.hidden_dim)
        self.value_fn_target.load_state_dict(self.value_fn.state_dict())

        self.value_fn_optimizer = torch.optim.Adam(self.value_fn.parameters(), lr=self.eta)

        self.reset_replay_buffer()

    @property
    def adapter(self):
        return self._adapter

    @adapter.setter
    def adapter(self, value):
        assert isinstance(value, Adapter)
        self._adapter = value

    def reset_replay_buffer(self, size=None):
        if size is None:
            size = self.mem_size
        self.replay_buffer = ReplayBuffer(max_len=size)

    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        Q_values = self.adapter.predict(self.value_fn, state)
        return Q_values.detach().numpy().argmax(1)

    def train(self, iterations, batch_size=64, gamma=0.99, tau=1., update_freq=3):

        for itr in range(iterations):

            sample = self.replay_buffer.sample(batch_size)
            state, next_state, action, reward, terminated = sample

            self.value_fn_optimizer.zero_grad()
            current_Qs = self.adapter.predict_batch(self.value_fn, state)
            target_Qs = self.adapter.predict_batch(self.value_fn_target, next_state)

            terminated = torch.tensor(terminated[:,None,:])
            reward = torch.tensor(reward[:,None,:])
            action = torch.tensor(action[:,:,None])

            target_Qs = target_Qs.amax(-1, keepdim=True).detach()
            target_Qs = reward + ((1-terminated) * gamma * target_Qs)
            current_Qs = torch.gather(current_Qs, dim=-1, index=action)
            Q_loss = ((current_Qs-target_Qs)**2).mean(0).sum()

            Q_loss.backward()
            self.value_fn_optimizer.step()

            if itr % update_freq == 0:
                for param, target_param in zip(self.value_fn.parameters(), self.value_fn_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau) * target_param.data)

    def save(self, directory):

        filename = 'agent'

        torch.save(self.value_fn.state_dict(), f'{directory}/{filename}_value_fn.pth')

        with open(f'{directory}/{filename}_replay_buffer.pkl', 'wb') as outfile:
            pickle.dump({
                'kwargs': {
                    'hidden_dim': self.hidden_dim,
                    'mem_size': self.mem_size,
                    'eta': self.eta,
                },
                'buffer': {
                    'storage': self.replay_buffer.storage,
                    'ptr': self.replay_buffer.ptr,
                }
            }, outfile)

    def load(self, directory):

        filename = 'agent'

        self.value_fn.load_state_dict(torch.load(f'{directory}/{filename}_value_fn.pth'))
        self.value_fn_target.load_state_dict(self.value_fn.state_dict())

        with open(f'{directory}/{filename}_replay_buffer.pkl', 'rb') as infile:
            info = pickle.load(infile)

        self.replay_buffer.storage = info['buffer']['storage']
        self.replay_buffer.ptr = info['buffer']['ptr']
        self.hidden_dim = info['kwargs']['hidden_dim']
        self.mem_size = info['kwargs']['mem_size']
        self.eta = info['kwargs']['eta']


def noise_decay(eta, step, decay=1e-4, min_rate=1e-4):
    rate = eta * 1/(decay*step + 1)
    return max(rate, min_rate)


def train(env, agent, *,
          train_steps=9, update_freq=3, batch_size=16, gamma=0.99, tau=0.99, exploration_rate=0.99,
          exploration_decay=5e-5, num_episodes=1000, burn_in=50, verbose=True, save_path=None):

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    history = pd.DataFrame(columns=['episode', 'exploration', 'reward', 'converged', 'modularity'])
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        history.to_csv(f'{save_path}/history.csv', index=False, header=True)

    for episode in range(1, num_episodes):
        try:

            state, info = env.reset()
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):

                if np.random.random() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                agent.replay_buffer.add({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'terminated': int(terminated),
                })

                state = next_state
                steps += 1

                mod = env.modularity
                history = pd.concat([
                    history,
                    pd.DataFrame([{
                        'episode': episode,
                        'exploration': exploration_rate,
                        'reward': reward,
                        'converged': terminated,
                        'modularity': mod,
                    }]),
                ], axis=0, ignore_index=True)
                vprint('|', end='')

            vprint('', episode, mod, exploration_rate)

            if len(agent.replay_buffer) > burn_in:
                agent.train(train_steps, batch_size=batch_size, gamma=gamma, tau=tau, update_freq=update_freq)
                exploration_rate = noise_decay(exploration_rate, episode, exploration_decay)

            if save_path is not None:
                history[history['episode']==episode].to_csv(f'{save_path}/history.csv', mode='a', index=False, header=False)
                hist_mod = history[history['converged']].groupby('episode')['modularity'].max()
                if (mod >= hist_mod).all():
                    agent.save(save_path)

        except KeyboardInterrupt:
            break

    return history


def eval(env, agent, *, num_episodes=1000, verbose=True):

    if env.stochastic:
        warnings.warn('Stochastic environment is unstable for evaluation.')

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    history = pd.DataFrame(columns=['episode', 'reward', 'converged', 'modularity'])
    preds = []

    for episode in range(num_episodes):
        try:

            state, info = env.reset()
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                state = next_state
                steps += 1

                mod = env.modularity
                history = pd.concat([
                    history,
                    pd.DataFrame([{
                        'episode': episode,
                        'reward': reward,
                        'converged': terminated,
                        'modularity': mod,
                    }]),
                ], axis=0, ignore_index=True)
                vprint('|', end='')

            vprint('', episode, mod)

            preds.append(env.partition)

        except KeyboardInterrupt:
            break

    return preds, history
