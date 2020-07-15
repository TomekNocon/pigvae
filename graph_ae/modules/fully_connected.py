import torch
from torch.nn import Linear, Sequential, ELU, BatchNorm1d, ReLU


class FNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, non_linearity="relu", batch_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        layers = [Linear(input_dim, hidden_dim)]
        layers += [Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        layers += [Linear(hidden_dim, output_dim)]
        self.layers = torch.nn.ModuleList(layers)
        if batch_norm:
            bn_layers = [BatchNorm1d(input_dim)]
            bn_layers += [BatchNorm1d(hidden_dim) for _ in range(num_layers - 2)]
            self.bn_layers = torch.nn.ModuleList(bn_layers)
        if non_linearity == "relu":
            self.non_linearity = torch.nn.ReLU
        elif non_linearity == "elu":
            self.non_linearity = torch.nn.ELU()

    def forward(self, x):
        for i in range(self.num_layers):
            if i > 0:
                x = self.non_linearity(x)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.layers[i](x)
        return x