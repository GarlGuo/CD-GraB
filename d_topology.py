import networkx as nx
import torch
import numpy as np
import abc
import math


class Graph(abc.ABC):
    def __init__(self, node_cnt, rank, world) -> None:
        self.node_cnt = node_cnt
        self._rank = rank
        self._world = world
        self._n_cuda_per_process = 1  # TODO: currently only supports 1

    @abc.abstractmethod
    def get_W(self):
        raise NotImplementedError()

    def get_P(self):
        return self.get_W() - torch.ones(self.node_cnt, self.node_cnt) / self.node_cnt

    def get_neighborhood(self):
        row = self.get_W()[self._rank]
        return {c: v for c, v in zip(range(len(row)), row) if v != 0}

    def get_neighbor_list(self):
        ans = []
        raw = self.get_neighborhood()
        for i in raw.keys():
            if i != self._rank:
                ans.append(i)
        return ans

    @property
    def world(self):  # unused for now due to we only have 1 cuda for each node
        assert self._world is not None
        self._world_list = self._world.split(",")
        assert self.node_cnt * \
            self._n_cuda_per_process <= len(self._world_list)
        return [int(l) for l in self._world_list]

    @property
    def device(self):  # unused for now due to we only have 1 cuda for each node
        # return a string
        return self.world[
            self._rank * self._n_cuda_per_process: (self._rank + 1) * self._n_cuda_per_process
        ]

    @property
    def rank(self):
        return self._rank

    @property
    def ranks(self):
        return list(range(self.node_cnt))


class Ring(Graph):
    def __init__(self, node_cnt, rank, world) -> None:
        super().__init__(node_cnt, rank, world)

    def get_W(self):
        if self.node_cnt == 1:
            return torch.ones(1, 1)
        W = torch.zeros(self.node_cnt, self.node_cnt, dtype=torch.float32)
        value = 1/3 if self.node_cnt >= 3 else 1/2
        W.fill_diagonal_(value)
        W[1:].fill_diagonal_(value)
        W[:, 1:].fill_diagonal_(value)
        W[0, self.node_cnt - 1] = value
        W[self.node_cnt - 1, 0] = value
        return W


class CentralizedGraph(Graph):
    def __init__(self, node_cnt, rank, world) -> None:
        super().__init__(node_cnt, rank, world)

    def get_W(self):
        return torch.ones((self.node_cnt, self.node_cnt)) / self.node_cnt


class DisconnectedGraph(Graph):
    def __init__(self, node_cnt, rank, world) -> None:
        super().__init__(node_cnt, rank, world)

    def get_W(self):
        return torch.eye(self.node_cnt)


class Torus(Graph):
    def __init__(self, node_cnt, rank, world) -> None:
        super().__init__(node_cnt, rank, world)

    def get_W(self):
        G = nx.generators.lattice.grid_2d_graph(
            int(math.sqrt(self.node_cnt)), int(math.sqrt(self.node_cnt)), periodic=True)
        W = torch.from_numpy(nx.adjacency_matrix(G).toarray())
        W.fill_diagonal_(1)
        W = W / 5
        return W


class FixedRhoTopology(Graph):
    def __init__(self, node_cnt, rho, rank, world) -> None:
        super().__init__(node_cnt, rank, world)
        self.rho = rho

    def get_W(self):
        AVG = torch.ones(self.node_cnt, self.node_cnt) / self.node_cnt
        u = torch.zeros(self.node_cnt, 1)

        for k in range(self.node_cnt // 2):
            u[k, 0] = 1 / math.sqrt(self.node_cnt)

        for k in range(self.node_cnt // 2, self.node_cnt):
            u[k, 0] = - 1 / math.sqrt(self.node_cnt)

        P = u @ u.T
        return AVG + self.rho * P


class SocialNetworkGraph(Graph):
    def __init__(self, node_cnt, rank, world):
        super(SocialNetworkGraph, self).__init__(node_cnt, rank, world)
        assert node_cnt == 32

    def get_W(self):
        # define the graph.
        graph = nx.davis_southern_women_graph()

        # get the mixing matrix.
        W = torch.from_numpy(nx.adjacency_matrix(
            graph).toarray().astype(np.float32))

        degrees = W.sum(axis=1)
        for node in torch.argsort(degrees)[::-1]:
            W[:, node][W[:, node] == 1] = 1.0 / degrees[node]
            W[node, :][W[node, :] == 1] = 1.0 / degrees[node]
            W[node, node] = 1 - torch.sum(W[node, :]) + W[node, node]
        return W


class MargulisExpanderGraph(Graph):
    def __init__(self, node_cnt, rank, world):
        super(MargulisExpanderGraph, self).__init__(node_cnt, rank, world)

    def get_W(self):
        base = int(math.sqrt(self.node_cnt))

        graph = nx.generators.margulis_gabber_galil_graph(base)

        W = torch.from_numpy(nx.adjacency_matrix(
            graph).toarray().astype(np.float32))
        W[W > 1] = 1

        degrees = W.sum(axis=1)
        for node in torch.argsort(-degrees):
            W[:, node][W[:, node] == 1] = 1.0 / degrees[node]
            W[node, :][W[node, :] == 1] = 1.0 / degrees[node]
            W[node, node] = 1 - torch.sum(W[node, :]) + W[node, node]

        return W


class ExpanderGraph(Graph):
    def __init__(self, node_cnt, rank, world):
        super(ExpanderGraph, self).__init__(node_cnt, rank, world)

    def get_W(self):
        # define the graph.
        def modulo_inverse(i, p):
            for j in range(1, p):
                if (j * i) % p == 1:
                    return j

        graph = nx.generators.cycle_graph(self.node_cnt)

        # get the mixing matrix.
        W = torch.from_numpy(nx.adjacency_matrix(
            graph).toarray().astype(np.float32))
        # for i in range(0, mixing_matrix.shape[0]):
        #     mixing_matrix[i][i] = 1
        W[0][0] = 1

        # connect with the inverse modulo p node.
        for i in range(1, W.shape[0]):
            W[i][modulo_inverse(i, self.node_cnt)] = 1

        W = W / 3

        return W
