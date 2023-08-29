import torch
from torch import nn
from torch_geometric_temporal import DyGrEncoder
import torch.nn.functional as F


class Dygre(torch.nn.Module):
    def __init__(self, conv_out_channels, conv_num_layers, lstm_out_channels, out_channels, lstm_num_layers=1, conv_aggr='add'):
        super(Dygre, self).__init__()
        self.recurrent = DyGrEncoder(conv_out_channels, conv_num_layers, conv_aggr, lstm_out_channels, lstm_num_layers)
        self.linear = nn.Linear(lstm_out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, H, C):
        h, H, C = self.recurrent(x, edge_index, edge_weight, H, C)
        h = F.relu(h)
        h = self.linear(h)
        return h, H, C
