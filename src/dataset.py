# -*- coding: utf-8 -*-
from torch_geometric.utils import dropout_adj, degree, dense_to_sparse
import torch
import torch_geometric as pygeo
from torch_scatter import scatter_sum
from torch_geometric.nn.pool.topk_pool import filter_adj


class useonlyAttribute(object):
    r"""Adds noise to edge indices."""
    def __init__(self, fea_dim=2):
        self.fea_dim = int(fea_dim)

    def __call__(self, data):
        x = data.x[:, :self.fea_dim]
        data.x = x
        
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RandomRemoveEdges(object):
    r"""Adds noise to edge indices."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        edge_index = data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=self.prob)
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class TotalRandomEdges(object):
    r"""Adds noise to edge indices."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        x = data.x
        num_nodes = x.shape[0]
        mask = x.new_full((num_nodes,num_nodes), self.prob, dtype=torch.float, device=x.device)
        mask = torch.bernoulli(mask)
        adj_m = mask.type(torch.long)
        edge_index, _ = dense_to_sparse(adj_m)
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RandomDropNodes(object):
    r"""randomly drop nodes."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        edge_index, x, edge_attr = data.edge_index, data.x, data.edge_attr
        num_nodes = x.shape[0]
        mask = x.new_full((num_nodes,), 1 - self.prob, dtype=torch.float)
        mask = torch.bernoulli(mask)
        mask = mask.type(torch.long)
        mask = mask.nonzero().view(-1)
        x = x[mask]
        data.x = x

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, mask, num_nodes=num_nodes)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.num_nodes=mask.size(0) # after you drop some nodes, you must update the num_nodes

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)




class RandomRemoveNodeFeatures(object):
    r"""Randomly change node features to -100 or -1. Keep the features with 1-p probability """
    def __init__(self, prob=0.5, padding=-1.0):
        self.prob = prob
        self.padding = padding

    def __call__(self, data):
        x = data.x
        num_nodes = x.shape[0]
        mask = x.new_full((num_nodes, ), 1 - self.prob, dtype=torch.float)
        mask = torch.bernoulli(mask).unsqueeze(1)
        data.x = x * mask + (mask - 1.0) * self.padding * -1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

        
class RemoveImportantNodeFeatures(object):
    r"""Change important node features to -1. Keep the features with 1-p probability """
    def __init__(self, prob=0.5, padding=0):
        self.prob = prob
        self.padding = padding

    def __call__(self, data):
        x = data.x
        edge_index = data.edge_index
        degrees = degree(edge_index[0])
        top_nodes_num = int(degrees.shape[0] * self.prob)
        degrees_sort_index = torch.argsort(degrees, descending=True)[:top_nodes_num]
        mask = torch.ones_like(degrees)
        mask[degrees_sort_index]=0.0
        mask = mask.unsqueeze(1)
#threshold = np.percentile(degrees.cpu().numpy(), int(100*(1-self.prob))) # this is for setting the top prop nodes features as -1
#        mask = ((degrees.unsqueeze(1)) >= threshold).type(torch.float32)
        data.x = x * mask + (mask - 1.0) * self.padding * -1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class RemoveEdgesNodeFeatures(object):
    r""" Randomly change node features to -1. Randomly drop edges. Keep the features with 1-p probability """
    def __init__(self, prob=0.5, padding=0):
        self.prob = prob
        self.padding = padding

    def __call__(self, data):
        #randomly remove edges
        edge_index = data.edge_index
        edge_index, _ = dropout_adj(edge_index, p=self.prob)
        data.edge_index = edge_index
        
        x = data.x
        num_nodes = x.shape[0]
        mask = x.new_full((num_nodes, ), 1 - self.prob, dtype=torch.float)
        mask = torch.bernoulli(mask).unsqueeze(1)
        data.x = x * mask + (mask - 1.0) * self.padding * -1
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def preproc(data):
    """ Preprocess Pytorch Geometric data objects to be used with our walk generator """

    if data.num_edges == 0:
        return data

    if not data.is_coalesced():
        data.coalesce()

    if data.num_node_features == 0:
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float32)

    if data.num_edge_features == 0:
        data.edge_attr = torch.zeros((data.num_edges, 1), dtype=torch.float32)

    edge_idx = data.edge_index
    edge_feat = data.edge_attr
    node_feat = data.x

    # remove isolated nodes
    if data.contains_isolated_nodes():
        edge_idx, edge_feat, mask = pygeo.utils.remove_isolated_nodes(edge_idx, edge_feat,     data.num_nodes)
        node_feat = node_feat[mask]

    # Enforce undirected graphs
    if not pygeo.utils.is_undirected(edge_idx):
        x = edge_feat.detach().numpy()
        e = edge_idx.detach().numpy()
        x_map = {(e[0,i], e[1,i]): x[i] for i in range(e.shape[1])}
        edge_idx = pygeo.utils.to_undirected(edge_idx)
        e = edge_idx.detach().numpy()
        x = [x_map[(e[0,i], e[1,i])] if (e[0,i], e[1,i]) in x_map.keys() else x_map[(e[1,i]    , e[0,i])] for i in range(e.shape[1])]
        edge_feat = torch.tensor(x)

    data.edge_index = edge_idx
    data.edge_attr = edge_feat
    data.x = node_feat

    if not torch.is_tensor(data.y):
        data.y = torch.tensor(data.y)
    data.y = data.y.view(1, -1)

    order = node_feat.shape[0]

    # create bitwise encoding of adjacency matrix using 64-bit integers
    data.node_id = torch.arange(0, order)
    bit_id = torch.zeros((order, order // 63 + 1), dtype=torch.int64)
    bit_id[data.node_id, data.node_id // 63] = torch.tensor(1) << data.node_id % 63
    data.adj_bits = scatter_sum(bit_id[edge_idx[0]], edge_idx[1], dim=0)

    # compute node offsets in the adjacency list
    data.degrees = pygeo.utils.degree(edge_idx[0], dtype=torch.int64)
    adj_offset = torch.zeros((order,), dtype=torch.int64)
    adj_offset[1:] = torch.cumsum(data.degrees, dim=0)[:-1]
    data.adj_offset = adj_offset

    return data
