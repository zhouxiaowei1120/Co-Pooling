import logging
import argparse
import ast
from collections import OrderedDict
import torch
import os
from datetime import datetime
import numpy as np
import json
irange = range
import networkx as nx

def mylogger(logpath='./param.log'):
    logger = logging.getLogger('mylogger')
    logger.setLevel('DEBUG')
    logger.propagate = False
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    fhlr = logging.FileHandler(logpath) # 
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    str_sharp = '#####################################################################'
    logger.info(str_sharp +'Record Experiment Information and Conditions\n')
    # logger.info('  Experiment Setting and Running Logs\n\n')

    chlr = logging.StreamHandler() # 
    chlr.setFormatter(formatter)
    chlr.setLevel('DEBUG')  # 
    logger.addHandler(chlr)

def parseArg ():
    parseArgs = argparse.ArgumentParser(description='Arguments for project.')
    parseArgs.add_argument('--alpha', type=float, default=0.1, help='the value of alpha for pageRank')
    parseArgs.add_argument('--batch_size',type=int, default= 512, help='Number of batch size') 
    parseArgs.add_argument('--baseline', action='store_true', default=False, help='the mode of training baseline or our model')
    parseArgs.add_argument('--regression', action='store_true', default=False, help='the mode of training regression model or others')
    parseArgs.add_argument('--baseLr', type=float, default = '0.001', help='The base learning rate for optimizer')
    parseArgs.add_argument('--dataset',type=str,default = 'PROTEINS', help='specify the training dataset')
    parseArgs.add_argument('--datapath',type=str,default = './data', help='specify the path to the dataset')
    parseArgs.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parseArgs.add_argument('--epoch_start',type=int, default= 0, help='the num of training epoch for restore,0 means training from scrach')
    parseArgs.add_argument('--epochs',type=int, default= 300, help='the num of max iteration')
    parseArgs.add_argument('--exp_att',type=str, default='test', help='the name of current experiment')
    parseArgs.add_argument('--eps', type=float, default=0, help='used for clipping edges')
    parseArgs.add_argument('--edge_ratio', type=float, default=0.6, help='ratio to select the edges')
    parseArgs.add_argument('--Gamma', default=None)
    parseArgs.add_argument('--gpu_ids',type=str, default= '', help='the ids of GPUs')
    parseArgs.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parseArgs.add_argument('--info', '-I', type=str, default='Info for running program', help='This info is used to record the running conditions for the current program, which is stored in param.log')
    parseArgs.add_argument('--K',type=int, default= 10, help='the num of pageRank step')
    parseArgs.add_argument('--l2', type=float, default=1.0, help='gate loss(i.e. the final classification loss) weight')
    parseArgs.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parseArgs.add_argument('--logfile',type=str, default= './logs.log', help='the name of log file')
    parseArgs.add_argument('--log_interval',type=int, default= 200, help='the interval of training epoch for logging')
    parseArgs.add_argument('--modelname',type=str, default= 'HGPSL', help='the name of DNN for explaining') 
    parseArgs.add_argument('--num_classes',type=int, default= 2, help='the num of classes in dataset')    
    parseArgs.add_argument('--num_layers',type=int, default= 3, help='the num of conv layers in model')    
    parseArgs.add_argument('--max_num_nodes',type=int, default= 100, help='the max num of nodes in a dataset')    
    parseArgs.add_argument('--nhid', type=int, default=128, help='hidden size')
    parseArgs.add_argument('--target_dim', type=int, default=0, help='regression target for QM9')
    parseArgs.add_argument('--noise_ratio', type=float, default=0.0, help='the probability of removing edges/node features')
    parseArgs.add_argument('--noise_type',type=str, default= 'randomnodes_-1', help='Use for path conflict') 
    parseArgs.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parseArgs.add_argument('--patience',type=int, default=50, help='the patience of optimizer')
    parseArgs.add_argument('--path_conflict',type=str, default= '', help='Use for path conflict') 
    parseArgs.add_argument('--phase',type=str, default= 'test', help='train or test')
    parseArgs.add_argument('--res_dir',type=str, default='./experiments', help='the path for saving results')
    parseArgs.add_argument('--restore_file',type=str, default='', help='the path/file for restore models')
    parseArgs.add_argument('--save_flag', type=ast.literal_eval, default=False, help='save figures or not')
    parseArgs.add_argument('--seed', type=int, default=200, help='the seed for random selection')
    parseArgs.add_argument('--test_batch',type=int, default= 512, help='Number of test batch size') 
    parseArgs.add_argument('--v', type=ast.literal_eval, default = False, help='display the debug info or not')
    parseArgs.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')

    return parseArgs.parse_args()

def loadArgs(args, r_iterms):
    filename = os.path.join(args.restore_file, 'parameters.txt')
    argsDict = args.__dict__
    with open(filename, 'r') as f:
        arg = json.load(f)
    for itm in r_iterms:
        argsDict[itm] = arg[itm]
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def time_stamp():
  TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  return TIMESTAMP

def get_para(model): # calculate the number of parameters
    para_num = 0
    for para in model.parameters():
        para_num += para.numel()
    return para_num

def create_name_experiment(parameters, attribute_experiment):
    name_experiment = '{}/{}'.format(parameters['dataset'], attribute_experiment)

    print('Name experiment: {}'.format(name_experiment))

    return name_experiment

def create_folder(folder, force=True):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    else:
        if force:
            folder = folder+str(np.random.randint(100))
            os.makedirs(folder, exist_ok=True)
    return folder


def loadweights(model, filename_model, gpu_ids=''):
    '''
    @Description: Load weights for pytorch model in different hardware environments
    @param {type} : {model: pytorch model, model that waits for loading weights
                     filename_model: str, name of pretrained weights
                     gpu_ids: list, available gpu list}
    @return: 
    '''
    if filename_model != '' and os.path.exists(filename_model):
        if len(gpu_ids) == 0:
            # load weights to cpu
            state_dict = torch.load(filename_model, map_location=lambda storage, loc: storage)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','') # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif len(gpu_ids) == 1:
            state_dict = torch.load(filename_model)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','') # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = torch.load(filename_model)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' not in k:
                    name = ''.join(['module.',k]) # add `module.`
                    new_state_dict[name] = v
            if new_state_dict:
                state_dict = new_state_dict
    else:
         state_dict = ''
    return model, state_dict


def construct_graph(edge_index, batch, nodes_index, cur_nodes_index, node_attr):
    edge_index = edge_index.cpu().detach().numpy()
    nodes_index = nodes_index.cpu().detach().numpy()
    cur_nodes_index = cur_nodes_index.cpu().detach().numpy()
    # if edge_attr is not None:
    #     edge_attr = np.round(edge_attr.cpu().detach().numpy(), 3)
    batch = batch.cpu().detach().numpy()
    g = nx.Graph()

    for idx in range(edge_index.shape[1]):
        edge_index[0][idx] = nodes_index[edge_index[0][idx]]
        edge_index[1][idx] = nodes_index[edge_index[1][idx]]
    if node_attr is not None:
        node_attr = node_attr.cpu().detach().numpy()
        nodes_added={}
        if node_attr.shape[1] == 7:
            # tmp_str = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
            tmp_str = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            node_labels = [np.where(r==1)[0][0] for r in node_attr]
            node_symbols = [tmp_str[nl] for nl in node_labels]
            for idx in range(cur_nodes_index.shape[0]):
                nodes_added[cur_nodes_index[idx]]=node_symbols[idx]
        nodes_added_sort = dict(sorted(nodes_added.items(), key=lambda kv: kv[0]))
        g.add_nodes_from(nodes_added_sort.keys())
        for idx, value in nodes_added_sort.items():
            g.nodes[idx]['color'] = value
    else:
        cur_nodes_index.sort()
        g.add_nodes_from(cur_nodes_index)
    g.add_edges_from(edge_index.transpose())
    # if edge_attr is not None:
    #     for i in range(edge_index.shape[1]):
    #         g.edges[edge_index[0][i],edge_index[1][i]]['weight'] = edge_attr[i]
    return g
