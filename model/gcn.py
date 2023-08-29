import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, node_features, out_channels):
        super(GCN, self).__init__()
        self.recurrent = GCNConv(node_features, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        return h
