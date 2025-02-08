import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from networkx.algorithms.shortest_paths.dense import floyd_warshall_numpy

from networkx.generators.random_graphs import *
from networkx.generators.ego import ego_graph
from networkx.generators.geometric import random_geometric_graph



class GeometricGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, samples_per_epoch=100000, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        g = random_geometric_graph(n=n, radius=0.5)
        return g


class BinomialGraphDataset(Dataset):
    def __init__(self, n_min=12, n_max=20, p_min=0.4, p_max=0.6,
                 samples_per_epoch=100000, pyg=False, **kwargs):
        super().__init__()
        self.n_min = n_min
        self.n_max = n_max
        self.p_min = p_min
        self.p_max = p_max
        self.samples_per_epoch = samples_per_epoch
        self.pyg = pyg

    def __len__(self):
        return self.samples_per_epoch

    def get_largest_subgraph(self, g):
        g = g.subgraph(sorted(nx.connected_components(g), key=len, reverse=True)[0])
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        return g

    def __getitem__(self, idx):
        n = np.random.randint(low=self.n_min, high=self.n_max)
        if self.p_min == self.p_max:
            p = self.p_min
        else:
            p = np.random.uniform(low=self.p_min, high=self.p_max)
        g = binomial_graph(n, p)
        if self.pyg:
            g = from_networkx(g)
        return g

class DenseGraphBatch(Data):
    def __init__(self, node_features, edge_features, mask, **kwargs):
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask
        for key, item in kwargs.items():
            setattr(self, key, item)

    @classmethod
    def from_sparse_graph_list(cls, data_list, labels=False):
        if labels:
            max_num_nodes = max([graph.number_of_nodes() for graph, label in data_list])
        else:
            max_num_nodes = max([graph.number_of_nodes() for graph in data_list])
        node_features = [] 
        edge_features = []
        mask = []
        y = []
        props = []
        for data in data_list:
            if labels:
                graph, label = data
                y.append(label)
            else:
                graph = data
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            nf = torch.ones(max_num_nodes, 1)
            node_features.append(nf.unsqueeze(0))
            dm = torch.from_numpy(floyd_warshall_numpy(graph)).long()
            dm = torch.clamp(dm, 0, 5).unsqueeze(-1)
            num_nodes = dm.size(1)
            dm = torch.zeros((num_nodes, num_nodes, 6)).type_as(dm).scatter_(2, dm, 1).float()
            edge_features.append(dm)
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.stack(edge_features, dim=0)
        mask = torch.cat(mask, dim=0)
        props = torch.cat(props, dim=0)
        # return node_features, edge_features, mask, props
        batch = cls(node_features=node_features, edge_features=edge_features, mask=mask, properties=props)
        if labels:
            batch.y = torch.Tensor(y)
        return batch

    def __repr__(self):
        repr_list = ["{}={}".format(key, list(value.shape)) for key, value in self.__dict__.items()]
        return "DenseGraphBatch({})".format(", ".join(repr_list))


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, labels=False, **kwargs):
        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=lambda data_list: DenseGraphBatch.from_sparse_graph_list(data_list, labels), **kwargs)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, graph_family, graph_kwargs=None, samples_per_epoch=100000, batch_size=32,
                 distributed_sampler=True, num_workers=1):
        super().__init__()
        if graph_kwargs is None:
            graph_kwargs = {}
        self.graph_family = graph_family
        self.graph_kwargs = graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def make_dataset(self, samples_per_epoch):
        if self.graph_family == "binomial":
            ds = BinomialGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        else:
            raise NotImplementedError
        return ds

    def train_dataloader(self):
        self.train_dataset = self.make_dataset(samples_per_epoch=self.samples_per_epoch)
        if self.distributed_sampler:
            train_sampler = DistributedSampler(
                dataset=self.train_dataset,
                shuffle=False
            )
        else:
            train_sampler = None
        return DenseGraphDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )

    def val_dataloader(self):
        self.eval_dataset = self.make_dataset(samples_per_epoch=4096)
        if self.distributed_sampler:
            eval_sampler = DistributedSampler(
                dataset=self.eval_dataset,
                shuffle=False
            )
        else:
            eval_sampler = None
        return DenseGraphDataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=eval_sampler,
        )


def binomial_ego_graph(n, p):
    g = ego_graph(binomial_graph(n, p), 0)
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    return g


logging.getLogger("lightning").setLevel(logging.WARNING)


graph_kwargs = {
        "n_min": 12,
        "n_max": 20,
        "m_min": 1,
        "m_max": 5,
        "p_min": 0.4,
        "p_max": 0.6
    }

graph_family = "binomial"


datamodule = GraphDataModule(
        graph_family=graph_family,
        graph_kwargs=graph_kwargs,
        batch_size=32,
        num_workers=0,
        samples_per_epoch=1000,
        distributed_sampler=None
    )

data = datamodule.train_dataloader()

for el in data:
    break

print(el)