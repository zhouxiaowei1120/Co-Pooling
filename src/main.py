# coding=utf-8
import os
import sys
sys.path.append(os.path.abspath('./src'))
import logging
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg') # make plot figure work fine without 

from trainer import trainer
from myutils import *
import json

if __name__ == '__main__':
    t1=time.time()
    args = parseArg()
    if args.phase != 'train':
        restore_items = ['baseline', 'dataset', 'modelname', 'num_classes', 'time_stamp', 'path_conflict', 'exp_att', 'eps', 'nhid', 'pooling_ratio', 'dropout_ratio', 'seed', 'edge_ratio', 'alpha']
        try:
            args = loadArgs(args, restore_items)
        except Exception as e:
            print(e)
    
    args.folds = 10 # set as 10 fold cross validation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # set seed for cpu
    torch.cuda.manual_seed(seed)       # set seed for GPU
    torch.cuda.manual_seed_all(seed)   # set seed for GPU

    device = torch.device("cpu" if args.gpu_ids == '' else "cuda")
    args.GpuNum = 0
    args.gpu_ids = list(map(int,args.gpu_ids.replace(',','')))
    if device.type == 'cuda':
        # When run code on single gpu, restore weights trained on multi gpus without bugs
        args.GpuNum = len(args.gpu_ids) 
        GpuNum = args.GpuNum
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.GpuNum > 1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl')

    if args.phase == 'train':
        args.time_stamp = time_stamp()
    args.path_experiment = os.path.join(args.res_dir, args.exp_att, args.dataset)
    logpath = os.path.join(args.path_experiment, 'logs', args.path_conflict+'-'+args.time_stamp)
    tmppath = os.path.join(args.path_experiment, 'tmp', logpath.split('/')[-1])
    logpath = create_folder(logpath, False)
    if args.phase == 'train':
        tmppath = create_folder(tmppath)
    
    mylogger(os.path.join(logpath, args.logfile))
    logger = logging.getLogger('mylogger')
    logger.setLevel('INFO')
       
    if args.phase == 'train':
        argsDict = args.__dict__
        with open(os.path.join(logpath, 'parameters.txt'), 'w') as f:
            json.dump(argsDict, f, indent=2)
    
    if args.local_rank==0:
        logger.info("{}. {}".format(args.info, args.time_stamp))
        str_sharp = '#####################################################################'
        logger.info(str_sharp+'\n')
        logger.info('Use device:{}'.format(device.type))
        logger.info('There are {} GPUs, use GPU {}'.format(args.GpuNum, args.gpu_ids)) 
        logger.info(args)
   
    args.device = device
    
    trainer(args, logpath, tmppath)

    if args.phase == 'train' and args.local_rank==0:
        logger.info(args)
    if args.local_rank==0:
        logger.info('Total cost time: {:.2f}'.format(time.time()-t1))
        logger.info('Finished!')