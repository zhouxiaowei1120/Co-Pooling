import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from .layers import CoPooling
from myutils import construct_graph


def graph_visualize(edge_index, batch, nodes_index, node_attr):
    # visualize graphs in a mini-batch
    g = []
    total_nodes = 0
    cur_node_attr = None
    for i in torch.unique(batch): # process each graph
        new_batch = batch[batch==i] # precess the graph i 
        num_nodes = new_batch.size(0) # get the number of nodes in graph i
        new_edge_index = edge_index[:, (edge_index[0]>=total_nodes)*(edge_index[0]<total_nodes+num_nodes)] 
        cur_nodes_index = nodes_index[total_nodes:total_nodes+num_nodes]
        if node_attr is not None:
            cur_node_attr = node_attr[total_nodes:total_nodes+num_nodes]
        total_nodes += num_nodes
        g.append(construct_graph(new_edge_index, new_batch, nodes_index, cur_nodes_index, cur_node_attr))
    return g

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.eps = args.eps
        self.edge_ratio = args.edge_ratio
        self.save_flag = args.save_flag
        assert self.eps == 0 or self.edge_ratio == 0
        num_layers = args.num_layers

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.convs.append(GCNConv(self.nhid, self.nhid))
            if args.modelname == 'Co-Pooling':
                self.pools.append(CoPooling(self.pooling_ratio, args.K, self.edge_ratio, self.nhid, args.alpha, args.Init, args.Gamma))
            else:
                raise('Unknown pooling type. Check the value of args.modelname: {}'.format(args.modelname))

        self.lin1 = torch.nn.Linear((self.nhid) * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()

        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        graphList = []
        ori_edge_attr_list = []
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.ones(batch.shape[0], 10).to(batch.device)
        nodes_index = batch.clone()
        if self.save_flag:
            for idx in torch.unique(batch):
                nodes_index[batch==idx] = torch.arange(0, batch[batch==idx].size(0), dtype=torch.long, device=batch.device) # change the index of each node starting from 0 in a graph for better visualization. The original index of node in graph is starting from 0 and ending as number-of-nodes in one batch. original index [0,1,2,3,...,300,301] changed index [0,1,2,3,0,1,2,3,4,...,0,1,2]

        edge_attr = None
        node_attr = None
        if self.args.dataset == 'MUTAG':
            node_attr = x # get the node labels
        if not self.training and self.save_flag:
            g = graph_visualize(edge_index, batch, nodes_index, node_attr)
            graphList.append(g)
            
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x_cat = []
        for pool, conv in zip(self.pools, self.convs):
            x, edge_index, edge_attr, batch, nodes_index, node_attr, adj_ppr_matrix = pool(x, edge_index, edge_attr, batch, nodes_index, node_attr)
            if edge_index.nelement() == 0:
                edge_index = edge_index.type(torch.long)
            if not self.training and self.save_flag:
                ori_edge_attr_list.append(adj_ppr_matrix) # save ppr_matrix for visualization
                g = graph_visualize(edge_index, batch, nodes_index, node_attr)
                graphList.append(g)
            x_cat.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            x = F.relu(conv(x, edge_index, edge_attr))
            
        x_cat.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))

        x = F.relu(x_cat[0])
        for x_tmp in x_cat[1:]:
            x += F.relu(x_tmp)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.lin3(x)
        if self.args.phase == 'test':
            save_x = x
        x = F.log_softmax(x, dim=-1)
        
        if self.args.phase == 'test':
            return x, graphList, ori_edge_attr_list, save_x
        else:
            return x, graphList, ori_edge_attr_list
