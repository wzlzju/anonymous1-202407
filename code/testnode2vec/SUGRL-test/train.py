from time import perf_counter as t
from evaluate import get_roc_score
import random
import time
import torch
from torch_geometric.utils import is_undirected, to_undirected
import os.path as osp
import torch.nn.functional as F
import torch.nn as nn
from colorama import Fore
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import coo_matrix, csr_matrix
from data_unit.utils import blind_other_gpus, row_normalize, sparse_mx_to_torch_sparse_tensor,normalize_graph
from models import LogReg, SUGRL_Fast
from torch_geometric.utils import degree
import os
import argparse
from sklearn.cluster import KMeans
from ruamel.yaml import YAML
from termcolor import cprint
from evaluate import mask_test_edges
from evaluate import clustering_metrics

import networkx as nx
import json, pickle

def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])

def get_args(model_name, dataset_class, dataset_name, data_name, weight_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Simple Unsupervised Graph Representation Learning')
    # Basics
    parser.add_argument("--num-gpus-total", default=0, type=int)
    parser.add_argument("--num-gpus-to-use", default=0, type=int)
    parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--custom_key", default=custom_key)
    parser.add_argument("--save_model", default=False)
    parser.add_argument("--seed", default=0)
    # Dataset
    parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--data-name", default=data_name)
    parser.add_argument("--weight-name", default=weight_name)
    # Pretrain
    parser.add_argument("--pretrain", default=False, type=bool)
    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate2', dest='lr2')
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link
    parser.add_argument('--w_loss1', type=float, default=1, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--w_loss3', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")
    # Update params from .yamls
    args = parser.parse_args()
    return args

def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))

def get_dataset(args, dataset_kwargs):
    if args.dataset_name in ["cloudmap"]:
        data = None
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/cloudmap/') + args.data_name
        g = nx.read_edgelist(path, data=False)
        A = torch.FloatTensor(nx.to_numpy_array(g))
        I = torch.eye(A.shape[1]).to(A.device)
        A_I = A + I
        A_I_nomal = normalize_graph(A_I)
        A_I_nomal = A_I_nomal.to_sparse()
        adj_1 = nx.to_scipy_sparse_array(g)

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset/cloudmap/') + args.weight_name
        g = nx.read_edgelist(path, data=True)
        x = nx.to_numpy_array(g)
        x = -np.sort(-x, axis=-1)[:,:]
        # x = np.zeros(x.shape)
        x = torch.FloatTensor(x)

        label = None
        nb_feature = x.shape[1]
        nb_classes = None
        nb_nodes = x.shape[0]

    return data, [A_I_nomal,adj_1], [x], [label, nb_feature, nb_classes, nb_nodes]


def run_SUGRL(args, gpu_id=None, **kwargs):
    # ===================================================#
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # ===================================================#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    cprint("## Loading Dataset ##", "yellow")
    dataset_kwargs = {}
    data, adj_list, x_list, nb_list = get_dataset(args, dataset_kwargs)
    nb_feature = nb_list[1]
    nb_nodes = nb_list[3]
    feature_X = x_list[0].to(running_device)
    A_I_nomal = adj_list[0].to(running_device)
    adj_1 = adj_list[1]

    cprint("## Done ##", "yellow")
    # ===================================================#
    model = SUGRL_Fast(nb_feature, cfg=args.cfg,
                       dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(running_device)
    # ===================================================#
    A_degree = degree(A_I_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    edge_index = A_I_nomal._indices()[1]
    # ===================================================#
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    lbl_z = torch.tensor([0.]).to(running_device)
    deg_list_2 = []
    deg_list_2.append(0)
    for i in range(nb_nodes):
        deg_list_2.append(deg_list_2[-1] + A_degree[i])
    idx_p_list = []
    for j in range(1, 101):
        random_list = [deg_list_2[i] + j % A_degree[i] for i in range(nb_nodes)]
        idx_p = edge_index[random_list]
        idx_p_list.append(idx_p)
    start = t()
    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        model.train()
        optimiser.zero_grad()
        idx_list = []
        for i in range(num_neg):
            idx_0 = np.random.permutation(nb_nodes)
            idx_list.append(idx_0)

        h_a, h_p = model(feature_X, A_I_nomal)

        h_p_1 = (h_a[idx_p_list[epoch % 100]] + h_a[idx_p_list[(epoch + 2) % 100]] + h_a[
            idx_p_list[(epoch + 4) % 100]] + h_a[idx_p_list[(epoch + 6) % 100]] + h_a[
                     idx_p_list[(epoch + 8) % 100]]) / 5
        s_p = F.pairwise_distance(h_a, h_p)
        s_p_1 = F.pairwise_distance(h_a, h_p_1)
        s_n_list = []
        for h_n in idx_list:
            s_n = F.pairwise_distance(h_a, h_a[h_n])
            s_n_list.append(s_n)
        margin_label = -1 * torch.ones_like(s_p)

        loss_mar = 0
        loss_mar_1 = 0
        mask_margin_N = 0
        for s_n in s_n_list:
            loss_mar += (margin_loss(s_p, s_n, margin_label)).mean()
            loss_mar_1 += (margin_loss(s_p_1, s_n, margin_label)).mean()
            mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
        mask_margin_N = mask_margin_N / num_neg

        loss = loss_mar * args.w_loss1 + loss_mar_1 * args.w_loss2 + mask_margin_N * args.w_loss3
        loss.backward()
        optimiser.step()
        string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||".format(loss_mar.item(), loss_mar_1.item(),
                                                                              mask_margin_N.item())
        if args.pretrain:
            if os.path.exists(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth'):
                    load_params = torch.load(args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth', map_location='cuda:0')
                    model_params = model.state_dict()
                    same_parsms = {k: v for k, v in load_params.items() if k in model_params.keys()}
                    model_params.update(same_parsms)
                    model.load_state_dict(model_params)
        if args.save_model:
            torch.save(model.state_dict(), args.checkpoint_dir + '/' + args.dataset_name + '_weights.pth')
        
    model.eval()
    h_a, h_p = model.embed(feature_X, A_I_nomal)
    embs = h_p
    embs = embs / embs.norm(dim=1)[:, None]
    print(embs.shape)
    print(embs)
    embs = embs.cpu().detach().numpy().tolist()
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'embs/') + '.'.join(args.data_name.split('.')[:-1]) + "_%s_embs.json"%args.custom_key
    with open(path, 'w') as f:
        json.dump(embs, f)






if __name__ == '__main__':

    # num_total_runs = 1
    main_args = get_args(
        model_name="SUGRL",  # GCN SUGRL
        dataset_class="cloudmap",
        # Planetoid,MyAmazon
        dataset_name="cloudmap",  # Cora, CiteSeer, PubMed, Photo, Computers
        data_name="ring.edgelist",
        weight_name="ring-0.2.edgelist",
        custom_key="classification",  # classification, link, clu
    )
    ### Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)
    # | Dataset class          | Dataset name | Custom key    |
    # | Planetoid              | Cora         | classification|
    # | Planetoid              | CiteSeer     | classification|
    # | Planetoid              | PubMed       | classification|
    # | MyAmazon               | Photo        | classification|
    # | MyAmazon               | Computers    | classification|
    # | PygNodePropPredDataset | ogbn-arxiv   | classification|
    # | PygNodePropPredDataset | ogbn-mag     | classification|
    # | PygNodePropPredDataset | ogbn-products| classification|
    pprint_args(main_args)

    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     black_list=main_args.black_list)
        if not alloc_gpu:
            alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                               if g not in main_args.black_list], 1))]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    t0 = time.perf_counter()
    run_SUGRL(main_args, gpu_id=alloc_gpu[0])

    cprint("Done")
