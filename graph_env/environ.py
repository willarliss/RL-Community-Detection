import torch
import gymnasium as gym
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.sparse import spmm


def torch_one_hot(x, k):

    if k is None:
        k = x.max()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    return torch.nn.functional.one_hot(x, num_classes=k).float()


def compute_modularity(graph, partition, num_communities):

    A = graph.adj()
    P = torch_one_hot(partition, num_communities)
    d = (graph.in_degrees()/2 + graph.out_degrees()/2)[:, None]
    m = graph.num_edges()

    Q = 1/(2*m) * (P.T.mm(spmm(A, P)) - P.T.mm(d.mm(d.T)).mm(P)/(2*m)).trace()

    # boost = (np.unique(partition).size / (num_communities + 1)) ** 2
    # penalty = ((num_communities - np.unique(partition).size) / num_communities) ** 2

    return Q.item()


class GraphEnv(gym.Env):

    metadata = {}

    def __init__(self,
                 graph: dgl.DGLGraph,
                 num_communities: int, *,
                 initial_partition: np.ndarray = None,
                 stochastic: bool = False,
                 max_iter: int = 100,
                 n_iter_no_change: int = 10,
                 min_change: float = 0.003):

        super().__init__()

        self.graph = graph
        self.num_communities = num_communities
        self.stochastic = stochastic
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.min_change = min_change
        self.num_nodes = graph.num_nodes()

        self.action_space = gym.spaces.Box(
            low=0, high=self.num_communities-1, shape=(self.num_nodes, ), dtype=int,
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=self.num_communities-1, shape=(self.num_nodes, ), dtype=int,
        )

        self.propagate = GraphConv(
            in_feats=self.num_communities,
            out_feats=self.num_communities,
            norm='both',
            # norm='left',
            weight=False,
            bias=False,
            activation=None,
            allow_zero_in_degree=True,
        ).eval()

        self.reset(initial_partition=initial_partition)

    def _validate_partition(self, partition):

        assert isinstance(partition, np.ndarray)
        partition = partition.astype(int)

        assert partition.ndim == 1 and partition.shape[0] == self.num_nodes
        assert partition.min() >= 0 and partition.max() <= self.num_communities-1

        return partition

    # def _transition(self, partition_pred):

    #     partition_pred = torch_one_hot(partition_pred, self.num_communities)
    #     probs = self.propagate(self.graph, partition_pred)

    #     if self.stochastic:
    #         probs /= probs.sum(1, keepdim=True)
    #         next_partition = torch.distributions.Categorical(probs=probs).sample()
    #     else:
    #         next_partition = probs.argmax(1)

    #     return next_partition.numpy()

    def _transition(self, partition_pred):

        no_change = partition_pred == self._current_partition

        partition_pred = torch_one_hot(partition_pred, self.num_communities)
        probs = self.propagate(self.graph, partition_pred)

        if self.stochastic:
            probs /= probs.sum(1, keepdim=True)
            next_partition = torch.distributions.Categorical(probs=probs).sample()
        else:
            next_partition = probs.argmax(1)

        # next_partition[no_change] = self.current_partition[no_change]
        return np.where(no_change, self._current_partition, next_partition.numpy())

    def _info(self):
        return {'stochastic': self.stochastic, 'modularity': self.modularity}

    @property
    def modularity(self):
        return float(self._current_modularity)

    @property
    def partition(self):
        return self._current_partition.copy()

    @property
    def stochastic(self):
        return bool(self._stochastic)

    @stochastic.setter
    def stochastic(self, value):
        self._stochastic = bool(value)

    def get_adapter(self, features=None):
        return Adapter(self.graph, self.num_communities, features=features)

    def compute_modularity(self, partition=None):

        if partition is None:
            partition = self._current_partition
        else:
            partition = self._validate_partition(partition)

        return compute_modularity(self.graph, partition, self.num_communities)

    def reset(self, initial_partition=None, *, seed=None, options=None):

        super().reset(seed=seed)

        assert isinstance(self.graph, dgl.DGLGraph)
        if initial_partition is None:
            initial_partition = self.observation_space.sample()
        else:
            initial_partition = self._validate_partition(initial_partition)

        self._current_partition = initial_partition
        self._current_modularity = self.compute_modularity(self._current_partition)
        self._modularity_trace = [self._current_modularity]

        observation = self._current_partition.copy()
        info = self._info()

        return observation, info

    def step(self, new_partition):

        new_partition = self._validate_partition(new_partition)
        new_modularity = self.compute_modularity(new_partition)
        d_modularity = (new_modularity - self._current_modularity) / np.abs(self._current_modularity).clip(1e-9, None)

        self._current_partition = self._transition(new_partition)
        self._current_modularity = self.compute_modularity(self._current_partition)
        self._modularity_trace.append(self._current_modularity)

        terminated = truncated = False
        if len(self._modularity_trace) >= self.max_iter:
            truncated = True
        elif len(self._modularity_trace) >= self.n_iter_no_change:
            diffs = np.diff(self._modularity_trace[-self.n_iter_no_change:])
            if (np.abs(diffs) < self.min_change).all():
                terminated = True

        if terminated:
            reward = 5 + 10*new_modularity
        elif truncated:
            reward = 10*new_modularity
        else:
            reward = d_modularity

        observation = self._current_partition.copy()
        info = self._info()

        return observation, reward, terminated, truncated, info


class Adapter:

    def __init__(self, graph, num_communities, features=None):

        self.graph = graph
        self.num_communities = num_communities

        if features is None:
            self.features = torch.zeros((self.graph.num_nodes(), 0))
        else:
            assert features.ndim == 2 and features.shape[0] == self.graph.num_nodes()
            self.features = features

    @property
    def input_dim(self):
        return self.features.shape[1] + self.num_communities

    @property
    def output_dim(self):
        return self.num_communities

    def _model_call(self, model, state):
        feats = torch_one_hot(state.flatten(), self.num_communities)
        feats = torch.cat([self.features, feats], dim=1).float()
        return model(self.graph, feats)

    def predict(self, model, state):
        assert state.ndim == 1 and state.shape[0] == self.graph.num_nodes()
        return self._model_call(model, state)

    def predict_batch(self, model, state):
        assert state.ndim == 2 and state.shape[1] == self.graph.num_nodes()
        return torch.stack([
            self._model_call(model, state[i, :]) for i in range(state.shape[0])
        ], dim=0)
