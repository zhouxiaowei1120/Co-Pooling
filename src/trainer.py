import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import traceback
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
from torch.utils.tensorboard import SummaryWriter

from myutils import AverageMeter, loadweights, accuracy, create_folder, get_para, time_stamp
from models.interSubGNN import Model as InterSubPool
from dataset import RandomRemoveEdges, RandomRemoveNodeFeatures, RemoveImportantNodeFeatures,RemoveEdgesNodeFeatures, RandomDropNodes, TotalRandomEdges, useonlyAttribute
from torch_geometric.transforms import ToDense, Compose

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

class data_prefetcher(): 
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.use_cuda = False
        if device.type == 'cuda':
           self.use_cuda = True
           self.stream = torch.cuda.Stream()
        self.preload()
        self.end = True

    def preload(self):
        try:
           self.data = next(self.loader)
        except StopIteration:
            self.end = None
            return
        if self.use_cuda:
          with torch.cuda.stream(self.stream):
              i = len(self.data)
              while i > 0:
                i -= 1
                if torch.is_tensor(self.data[i]):
                    self.data[i] = self.data[i].cuda(non_blocking=True)
                      
    def next(self):
        if self.use_cuda:
           torch.cuda.current_stream().wait_stream(self.stream)
        end = self.end
        data = self.data
        self.preload()
        return data, end


class MyFilter(object):
    def __init__(self, max_num_nodes):
        self.max_nodes = max_num_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes

def load_data(args, logger):
    # num_workers = args.num_workers
    pin_memory = True
    datapath = args.datapath

    if args.dataset in ['PROTEINS', 'DD', 'ENZYMES', 'NCI1', 'NCI109', 'MUTAG', 'REDDIT-MULTI-12K', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'MSRC_21', 'BZR', 'FRANKENSTEIN', 'AIDS']:
        pre_trans = None
        pre_filter = None
        if args.noise_type=='onlyAttribute':
            if args.dataset == 'BZR':
                pre_trans = useonlyAttribute(3)
            if args.dataset == 'AIDS':
                pre_trans = useonlyAttribute(4)
        if args.noise_ratio > 0.0 and args.noise_ratio <= 1.0:
            if args.noise_type == 'randomedges':
                pre_trans = RandomRemoveEdges(args.noise_ratio)
            elif args.noise_type == 'totalrandomedges':
                pre_trans = TotalRandomEdges(args.noise_ratio)
            elif 'randomnodes' in args.noise_type:
                padding = float(args.noise_type.split('_')[-1])
                pre_trans = RandomRemoveNodeFeatures(args.noise_ratio, padding)
            elif 'ImpNodes' in args.noise_type:
                padding = float(args.noise_type.split('_')[-1])
                pre_trans = RemoveImportantNodeFeatures(args.noise_ratio, padding)
            elif 'randomEdgeNodes' in args.noise_type:
                padding = float(args.noise_type.split('_')[-1])
                pre_trans = RemoveEdgesNodeFeatures(args.noise_ratio, padding)
            elif 'randomDropNodes' == args.noise_type:
                pre_trans = RandomDropNodes(args.noise_ratio)
            else:
                raise Exception('Unkown noise type')
            if args.dataset == 'BZR':
                pre_trans = Compose([useonlyAttribute(3), pre_trans])
            if args.dataset == 'AIDS':
                pre_trans = Compose([useonlyAttribute(4), pre_trans])
            
        if args.modelname == 'DiffPool':
            if pre_trans is not None:
                pre_trans = Compose([pre_trans, ToDense(args.max_num_nodes)])
            else:
                pre_trans = ToDense(args.max_num_nodes)
            pre_filter=MyFilter(args.max_num_nodes)

        dataset = datasets.TUDataset(datapath, name=args.dataset, pre_transform=pre_trans, use_node_attr=True, pre_filter=pre_filter)
        args.num_classes = dataset.num_classes
        if dataset.num_features == 0:
            args.num_features = 10
        else:
            args.num_features = dataset.num_features
    else:
        logger.error('Unknown type of dataset!')
        raise Exception('Unknown type of dataset!')

    kf = KFold(args.folds, shuffle=True, random_state=args.seed)

    test_indices, train_indices = [], []
    for _, idx in kf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(args.folds)]

    for i in range(args.folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return dataset, train_indices, test_indices, val_indices

def train(model, optimizer_c, args, train_loader, epoch, logger, writer):
    device = args.device
    log_interval = args.log_interval
    fold = args.cur_fold
    train_loss = AverageMeter('train_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    model.train()

    for batchID, data in enumerate(train_loader):
        data = data.to(args.device)
        optimizer_c.zero_grad()
        
        out, _, _ = model(data)
        acc1 = accuracy(out, data.y)

        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer_c.step()

        top1.update(acc1[0].item(), data.y.size(0))
        train_loss.update(loss.item(), data.y.size(0))
        
        torch.cuda.empty_cache()
        if batchID % log_interval == 0:
            if args.local_rank==0 and writer:
                writer.add_scalar('Loss/TotalTrain/'+str(fold), train_loss.avg, (epoch-1)*len(train_loader)+batchID)
                writer.add_scalar('Acc/Train'+str(fold), top1.avg, (epoch-1)*len(train_loader)+batchID)   
    
    if args.local_rank==0:
        logger.info('Total training Loss: {:.3f} | Training Class Acc: {:.3f}%'.format(train_loss.avg, top1.avg)) 
    return top1.avg, train_loss.avg


def val(model, args, val_loader, logger, writer, epoch, logpath):
    model.eval()
    
    device = args.device
    save_flag=args.save_flag
    fold = args.cur_fold

    with torch.no_grad():
        total_val_loss = AverageMeter('total_val_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
    
        for batchID, data in enumerate(val_loader):
            data = data.to(args.device)
            
            #forward 
            # t1 = time()
            out, graphList, ori_edge_attr_list = model(data)
            acc1 = accuracy(out, data.y)

            loss = F.nll_loss(out, data.y)

            top1.update(acc1[0].item(), data.y.size(0))
            total_val_loss.update(loss.item(), data.y.size(0))

            # logging images and loss
            if save_flag and batchID == 0:
                graphname = ['ori', 'pool1', 'pool2']
                if epoch == 1 or epoch % 10 == 0:      
                    for idx, tmpG in enumerate(graphList):
                        #print(tmpG.nodes)
                        total_num = 0
                        gname=graphname[idx]
                        tmpg = tmpG[0]
                        edge_attr = ori_edge_attr_list[idx][total_num:total_num+tmpg.number_of_edges()]
                        total_num += tmpg.number_of_edges()
                        tmpfig = save_graphFig(tmpg, edge_attr, '', gname)
                        # tmpfig.set_title(gname)
                        # tmpfig.legend()
                        if writer:
                            writer.add_figure('pooled graph/layer/'+str(fold)+str(idx), tmpfig, epoch)
                        del tmpfig
               
        if args.local_rank==0 and writer:
            #logging loss and accuracy
            writer.add_scalar('Loss/Totalval/'+str(fold), total_val_loss.avg, epoch)
            writer.add_scalar('Acc/val/'+str(fold), top1.avg, epoch)
       
        if args.local_rank==0:
            logger.info('Epoch: {} | Total val Loss: {:.3f} | val Class Acc: {:.3f}%'.format(epoch, total_val_loss.avg, top1.avg))
    return top1.avg, total_val_loss.avg

def save_graphFig(tmpg, edge_attr, filename, gname):
    tmpfig = plt.figure()
    pos = nx.shell_layout(tmpg)
    for j, g_edge in enumerate(tmpg.edges):
        tmpg.edges[g_edge[0],g_edge[1]]['weight'] = edge_attr[j] 
    edge_labels = nx.get_edge_attributes(tmpg,'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(tmpg, pos, edge_labels=edge_labels)
    node_labels = nx.get_node_attributes(tmpg,'color')                        
    # print(node_labels)
    if node_labels:
        node_colors = []
        for _, value in node_labels.items():
            node_colors.append(value)
    else:
        node_colors = '#1f78b4'
    # nx.draw_networkx_labels(tmpg, pos, )
    nx.draw_networkx(tmpg, pos, node_color=node_colors, label=gname)
    if filename:
        tmpfig.savefig(filename)
        del tmpfig
    else:
        return tmpfig

def test(model, args, test_loader, logger, logpath):
    model.eval()
    
    device = args.device
    save_flag=args.save_flag
    save_batch = [10, 1]
    if save_flag:
        test_res_dir = os.path.join(logpath, 'test_res')
        create_folder(test_res_dir, False)

    with torch.no_grad():
        total_test_loss = AverageMeter('total_test_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        if args.phase == 'test':
            save_fea = np.empty((0, args.num_classes))
            save_label = np.empty((0))
    
        for batchID, data in enumerate(test_loader):
            data = data.to(args.device)
            
            #forward 
            # t1 = time()
            if args.phase == 'test':
                out, graphList, ori_edge_attr_list, save_x = model(data)
                save_fea = np.append(save_fea, save_x.cpu().detach().numpy(), axis=0)
                _, pred = out.max(1)
                save_label = np.append(save_label, pred.cpu().detach().numpy(), axis=0)
            else:
                out, graphList, ori_edge_attr_list = model(data)
            acc1 = accuracy(out, data.y)

            loss = F.nll_loss(out, data.y)

            top1.update(acc1[0].item(), data.y.size(0))
            total_test_loss.update(loss.item(), data.y.size(0))
            
            if save_flag:
                # print('process batch '+ str(batchID))
                if batchID in save_batch:
                    # logging images
                    img_name_dir = 'fold_'+str(args.cur_fold)+'/'+'batch_'+str(batchID) 
                    create_folder(os.path.join(test_res_dir, img_name_dir), False)
                    graphname = ['ori', 'pool1', 'pool2']
                    glabel = data.y.cpu().detach().numpy()
                    for idx, tmpG in enumerate(graphList):
                        gname = graphname[idx]
                        total_num = 0
                        for idx1, tmpg in enumerate(tmpG):
                            tmpg = tmpG[idx1]
                            #print(tmpG.nodes)
                            total_num += tmpg.number_of_edges()
                            edge_attr = ori_edge_attr_list[idx][total_num:total_num+tmpg.number_of_edges()]
                            graph_name = os.path.join(test_res_dir, img_name_dir, str(idx1)+'_'+gname+'_label'+str(glabel[idx1])+'.png')
                            save_graphFig(tmpg, edge_attr, graph_name, gname)
                            plt.close('all')
                             
        if args.local_rank==0:
            logger.info('Total test Loss: {:.3f} | Test Class Acc: {:.3f}%'.format(total_test_loss.avg, top1.avg))
        if args.phase == 'test':
            markers = ['x', '+', 'o', '^', '*', '#', '@', '!']
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
            fea_tsne = TSNE(n_components=2).fit_transform(save_fea)
            f2 = plt.figure()
            for label_tmp in range(args.num_classes):
                idx_1 = np.where(save_label==label_tmp)
                plt.scatter(fea_tsne[idx_1,1], fea_tsne[idx_1,0], marker = markers[label_tmp], color = colors[label_tmp], label=str(label_tmp), s = 30)
            plt.legend(loc = 'upper right')
            plt.savefig(logpath+'/fold_'+str(args.cur_fold)+'_tsne.png') 
            plt.close('all')
    return top1.avg, total_test_loss.avg

def trainer(args, logpath, tmppath):
    Failed = False
    logger = logging.getLogger('mylogger')
    
    # load data
    dataset_loaded, train_indices, test_indices, val_indices = load_data(args, logger)
    val_acc_list, test_acc_list = [], []

    # k-fold cross validation code
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(train_indices, test_indices, val_indices)):
        logger.info('Process the fold: '+str(fold))
        args.cur_fold = fold
        # build model
        weightsname = args.restore_file
        if args.modelname in ['Co-Pooling']:
            model = InterSubPool(args)
        else:
            raise Exception('Error: Unknown type of modelname {}'.format(args.modelname))

        model.reset_parameters()
        if args.phase == 'test':
            if not os.path.exists(os.path.join(weightsname, 'best_'+str(fold)+'.pth')):
                continue
        model, state_dic = loadweights(model, os.path.join(weightsname, 'best_'+str(fold)+'.pth'), args.gpu_ids)

        if state_dic:
            model.load_state_dict(state_dic, strict=True)
            if args.local_rank==0:
                logger.info('Use pre-trained model {}'.format(weightsname))
        model.to(args.device)

        train_dataset	= dataset_loaded[train_idx]
        test_dataset	= dataset_loaded[test_idx]
        val_dataset	= dataset_loaded[val_idx]

        train_loader	= DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_loader	= DataLoader(val_dataset, args.batch_size, shuffle=False)
        test_loader	= DataLoader(test_dataset, args.batch_size, shuffle=False)
    
        if args.phase == 'train' and args.local_rank==0:
            logger.info(model.modules)
            model_name = ['model']
            tmpIdx = 0
            for tmp_model in [model]:
                para_num = get_para(tmp_model)
                if args.local_rank==0:
                    logger.info('The number of parameters of {} is {}'.format(model_name[tmpIdx], para_num))
                tmpIdx += 1
            if args.save_flag:
                writer = SummaryWriter(tmppath)
            else:
                writer = ''

        optimizer_c = torch.optim.Adam([
                   {'params': model.parameters()},], 
                   args.baseLr, weight_decay=args.weight_decay)
   
        best_acc = 0.0
        best_epoch = 0
        min_loss = 1e10
        patience_cnt = 0
    
        if args.phase == 'train':
            if args.local_rank==0:
                logger.info('Start training the model.')
            try:
                for epoch in range(args.epoch_start + 1, args.epochs + 1):
                    # train and log accuracy and loss
                    train_acc, train_loss  = train(model, optimizer_c, args, train_loader, epoch, logger, writer)
                    
                    # val and log accuracy and loss
                    val_acc, val_loss = val(model, args, val_loader, logger, writer, epoch, logpath)
                    if args.gpu_ids != '':
                        torch.cuda.empty_cache()

                    if val_loss < min_loss:
                        min_loss = val_loss
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                    if patience_cnt == args.patience:
                        break
                    
                    if val_acc >= best_acc:
                        if args.local_rank==0:
                            logger.info('The epoch {} gets better accuracy:{} >= {}'.format(epoch, val_acc, best_acc))
                        best_acc = val_acc
                        best_epoch = epoch
                        if args.local_rank==0:
                            torch.save(model.state_dict(), os.path.join(logpath,'best_'+str(fold)+'.pth'))
            
                logger.info('fold: {}. The val accuracy of best model of epoch {} on dataset {} is {}\n'.format(fold, best_epoch, args.dataset, best_acc))
                
                # test the accuracy on the test dataset
                if args.gpu_ids != '':
                    torch.cuda.empty_cache()
                model, state_dic = loadweights(model, os.path.join(logpath, 'best_'+str(fold)+'.pth'), args.gpu_ids)
                if state_dic:
                    model.load_state_dict(state_dic, strict=True)
                else:
                    raise('Load weights of the best model failed!')
                
                logger.info('Start test the best model of epoch {}'.format(best_epoch))                
                test_acc, _ = test(model, args, test_loader, logger, logpath)
            
                # logger.info('fold: {}. The test accuracy of best model of epoch {} on dataset {} is {}\n'.format(fold, best_epoch, args.dataset, test_acc))
                val_acc_list.append(best_acc)
                test_acc_list.append(test_acc)
                logger.info('Current val acc: '+str(val_acc_list)+'   '+str(np.mean(val_acc_list))+'   '+str(np.std(val_acc_list)))
                logger.info('Current test acc: '+str(test_acc_list)+'   '+str(len(test_acc_list))+'    '+str(np.mean(test_acc_list))+'   '+str(np.std(test_acc_list)))
            except Exception as e:
                if args.local_rank==0:
                    logger.info(e)
                traceback.print_exc()
                Failed = True
                errors = e
            else:
                logger.info('fold {}. Training and test finished.\n'.format(fold))
        elif args.phase == 'test':
            if args.local_rank==0:
                logger.info('Start test the model. {}'.format(args.restore_file))
            acc_class, _ = test(model, args, test_loader, logger, logpath)
            test_acc_list.append(acc_class)
            val_acc_list.append(0)
            if args.local_rank==0:
                logger.info('The accuracy of model {} on dataset {} is {}'.format(args.restore_file, args.dataset, acc_class)) 
                logger.info('Finish test!')
        else:
            raise Exception('Error! The phase should be in train/test/val.')
        if Failed:
            break

    val_acc_mean = np.round(np.mean(val_acc_list), 6)
    test_acc_mean = np.round(np.mean(test_acc_list), 6)
    val_acc_std = np.round(np.std(val_acc_list), 6)
    test_acc_std = np.round(np.std(test_acc_list), 6)
    logger.info('Each val acc:  ')
    logger.info(str(val_acc_list))
    logger.info('Each test acc:  ')
    logger.info(str(test_acc_list)+'   '+str(len(test_acc_list)))
    logger.info('The mean val and test accuracy of model on dataset {} is {}({}), {}({})\n'.format(args.dataset, val_acc_mean, val_acc_std, test_acc_mean, test_acc_std))
    res_summary = open(os.path.join(args.res_dir, args.exp_att, args.dataset, args.dataset+'.txt'), 'a')
    res_summary.writelines('------------------------------{}-------------------------------\n'.format(args.time_stamp))
    res_summary.writelines(str(args)+'\n')
    res_summary.writelines('Each val acc:  ')
    res_summary.writelines(str(val_acc_list))
    res_summary.writelines('\nEach test acc:  ')
    res_summary.writelines(str(test_acc_list))
    res_summary.writelines('\nThe mean val and test accuracy of model on dataset {} is {}({}), {}({})'.format(args.dataset, val_acc_mean, val_acc_std, test_acc_mean, test_acc_std))
    res_summary.writelines('  '+str(len(test_acc_list)))
    res_summary.writelines(time_stamp()+'\n')
    res_summary.writelines('\n\n')
    res_summary.close()
    res_table = open(os.path.join(logpath, 'accuracy.txt'), 'w')
    res_table.writelines(str(args.seed)+'\t'+str(args.baseLr)+'\t'+str(args.weight_decay)+'\t'+str(args.pooling_ratio)+'\t'+str(args.nhid)+'\t'+str(args.edge_ratio)+'\t')
    res_table.writelines(str(val_acc_mean)+'\t'+str(val_acc_std)+'\t'+str(test_acc_mean)+'\t'+str(test_acc_std)+'\t'+str(args.dropout_ratio)+'\t'+str(len(test_acc_list))+'\t'+str(args.alpha)+'\n')
    res_table.close()
    if Failed:
        logger.info('Failed!')
        sys.exit(errors)
