import torch
from torch_geometric_temporal import EvolveGCNO
import torch.nn.functional as F


class EvolveGCN(torch.nn.Module):
    def __init__(self, node_features, out_channels):
        super(EvolveGCN, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, out_channels)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
