import torch
from torch_geometric_temporal import GConvGRU


class GRUConv(torch.nn.Module):
    def __init__(self, node_features, out_channels, k=2):
        super(GRUConv, self).__init__()
        self.recurrent = GConvGRU(node_features, out_channels, k)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        return h
