import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_bases, dropout=0.1):
        super(RGCN, self).__init__()
        out_channels = (out_channels - in_channels) // 2
        self.conv1 = RGCNConv(in_channels,  # 256
                              out_channels,  # 128
                              num_relations, num_bases, )
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations, num_bases, )
        self.act_fn = nn.LeakyReLU()
        self.act_fn2 = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data, edge_index, edge_type):
        x = data
        all_embed = [F.normalize(data, p=2, dim=-1)]
        x = self.act_fn(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)
        norm_x = F.normalize(x, p=2, dim=-1)
        all_embed.append(norm_x)
        x = self.act_fn2(self.conv2(x, edge_index, edge_type))
        x = self.dropout(x)
        norm_x = F.normalize(x, p=2, dim=-1)
        all_embed.append(norm_x)
        return torch.cat(all_embed, dim=-1)
