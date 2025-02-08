import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import random
import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import networkx.generators as nxg
from networkx.generators.random_graphs import *
import scipy.sparse as scs
import numpy as np
import torch


SIZE = 24
PATCH_SIZE = 4
class CellDataset(Dataset):

    def __init__(self, imgs, targets, channels, img_transform=None):

        super().__init__()
        self.img_transform = img_transform
        self.imgs = imgs
        self.targets = targets
        self.channels = channels

    def __getitem__(self, idx):
        
        img = self.imgs[idx][self.channels, :].to(torch.float32)
        target = self.targets[idx]
        if self.img_transform:
            img = self.img_transform(img.unsqueeze(0))

        return img, target

    def __len__(self):
        return len(self.imgs)
    
class SplitPatches(nn.Module):
    
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        # bs, c, h, w = x.shape
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        a = a.view(bs, -1, c *self.patch_size* self.patch_size)
        # a -> ( B no.of patches c p p )
        return a
     
class GridGraphDataset(Dataset):
    def __init__(self,
                 grid_size: int,
                 imgs,
                 targets, 
                 channels, 
                 img_transform=None,
                 sample_per_epoch: int = 10_000
        ):
        self.grid_size = grid_size
        self.sample_per_epoch = sample_per_epoch
        self.img_transform = img_transform
        self.imgs = imgs[:self.sample_per_epoch]
        self.targets = targets[:self.sample_per_epoch]
        self.channels = channels
        
    def __len__(self):
        return self.sample_per_epoch
    
    def __getitem__(self, idx):
        g = nx.grid_graph((self.grid_size, self.grid_size))
        img = self.imgs[idx][self.channels, :].to(torch.float32)
        target = self.targets[idx]
        if self.img_transform:
            img = self.img_transform(img.unsqueeze(0))
        return (g, img, target)

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
        g = nxg.geometric.random_geometric_graph(n=n, radius=0.5)
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
        g = nxg.random_graphs.binomial_graph(n, p)
        if self.pyg:
            g = from_networkx(g)
        return g

class DenseGraphBatch(Data):
    def __init__(self, node_features, edge_features, mask, **kwargs):
        super().__init__(**kwargs)  # Call the parent class' constructor (Data)
        self.node_features = node_features
        self.edge_features = edge_features
        self.mask = mask

    @classmethod
    def from_sparse_graph_list(cls, data_list, labels=True):
        if labels:
            max_num_nodes = max([graph.number_of_nodes() for graph, _, _ in data_list])
        else:
            max_num_nodes = max([graph.number_of_nodes() for graph in data_list])
        node_features = [] 
        edge_features = []
        mask = []
        y = []
        props = []
        for graph, embedding, label in data_list:
            y.append(label)
            num_nodes = graph.number_of_nodes()
            props.append(torch.Tensor([num_nodes]))
            graph.add_nodes_from([i for i in range(num_nodes, max_num_nodes)])
            nf = embedding
            node_features.append(nf)
            mask.append((torch.arange(max_num_nodes) < num_nodes).unsqueeze(0))
        node_features = torch.cat(node_features, dim=0)
        edge_features = torch.tensor(edge_features)
        mask = torch.cat(mask, dim=0)
        props = torch.cat(props, dim=0)
        batch = DenseGraphBatch(node_features=node_features, edge_features=edge_features, mask=mask, properties=props)
        if labels:
            batch.y = torch.Tensor(y)
        return batch
    


def dense_graph_collate_fn(data_list):
    return DenseGraphBatch.from_sparse_graph_list(data_list)


class DenseGraphDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, labels=True, **kwargs):
        self.labels = labels
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dense_graph_collate_fn,  # Directly pass the standalone function
            **kwargs,
        )


class GraphDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            graph_family,
            train_graph_kwargs=None, 
            val_graph_kwargs=None,
            samples_per_epoch=100000, 
            batch_size=32,
            distributed_sampler=True, 
            num_workers=1
        ):
        super().__init__()
        # if graph_kwargs is None:
        #     graph_kwargs = {}
        self.graph_family = graph_family
        self.train_graph_kwargs = train_graph_kwargs
        self.val_graph_kwargs = val_graph_kwargs
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.distributed_sampler = distributed_sampler
        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None

    def make_dataset(self, samples_per_epoch, is_train = True):
        if self.graph_family == "binomial":
            ds = BinomialGraphDataset(samples_per_epoch=samples_per_epoch, **self.graph_kwargs)
        elif self.graph_family == 'grid':
            if is_train:
                graph_kwargs = self.train_graph_kwargs
            else:
                graph_kwargs = self.val_graph_kwargs
            ds = GridGraphDataset(sample_per_epoch=samples_per_epoch, **graph_kwargs) 
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
            persistent_workers=True
        )

    def val_dataloader(self):
        self.eval_dataset = self.make_dataset(samples_per_epoch=self.samples_per_epoch // 5)
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
            persistent_workers=True
            
        )
        
        
class LaplacianGridEmbedding(nn.Module):
    def __init__(
        self, 
        grid_size: int, 
        out_dim: int, 
    ):
        super().__init__()
        self.grid_size = grid_size
        self.out_dim = out_dim

        # Define the fully connected layer
        self.fc1 = nn.Linear(grid_size**2, out_dim)

    def forward(self, x: torch.Tensor):
       
        x = x.transpose(-2, -1)
        output = self.fc1(x)
        return output
    
    @classmethod
    def get_graph_laplacian(cls, A):
        D = np.diag(A.sum(axis=0))
        L = D - A
        return L