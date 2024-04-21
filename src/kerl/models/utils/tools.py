import torch
from torch_geometric.utils import sort_edge_index


def edge_to_pyg_format(edge):
    edge_sets = torch.as_tensor(edge, dtype=torch.long)
    edge_idx = edge_sets[:, :2].t().contiguous()
    edge_type = edge_sets[:, 2]

    edge_index = torch.stack([edge_idx[1], edge_idx[0]])
    # sort edge index
    edge_idx = sort_edge_index(edge_index, sort_by_row=False)
    return edge_idx, edge_type
