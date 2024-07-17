import seaborn as sns
import json, pickle

from time import time
import logging
import os
import os.path as osp
import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
from torch.optim.lr_scheduler import StepLR


from utils import stat_graph, split_class_graphs, align_graphs
from utils import two_graphons_mixup, universal_svd
from graphon_estimator import universal_svd
import matplotlib.pyplot as plt

import networkx as nx

import flask


c_class = 0
c_resdir = 3
c_resvar = "d2v"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TUDataset("../data/", name="REDDIT-BINARY")
dataset = list(dataset)

class_graphs = split_class_graphs(dataset)
graphs0 = class_graphs[c_class][1]
# print(class_graphs)
g = nx.from_numpy_array(graphs0[907])
g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
pos = nx.nx_agraph.graphviz_layout(g)
pos = nx.spring_layout(g, iterations=50)
nx.draw(g, pos=pos, node_size=6,alpha=0.1)
plt.show()

d = nx.json_graph.node_link_data(g)  
json.dump(d, open("force/force.json", "w"))

d1,d2,d2o,d1p,d2p,d2op,d1v,d2v,d2ov = [],[],[],[],[],[],[],[],[]
maxnodes = max([len(g) for g in graphs0])
for i,g in enumerate(graphs0):
    g = nx.from_numpy_array(g)
    # g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    ds = nx.degree(g)
    ds = list(ds)
    ds.sort(reverse=True, key=lambda x:x[1])
    d1.append((i,ds[0][1]))
    d2.append((i,ds[1][1]))
    d1p.append((i,ds[0][1]/g.number_of_nodes()))
    d2p.append((i,ds[1][1]/g.number_of_nodes()))
    d1v.append((i,ds[0][1]/g.number_of_nodes()+ds[0][1]/maxnodes))
    d2v.append((i,ds[1][1]/g.number_of_nodes()+ds[1][1]/maxnodes))
    ns1 = list(nx.neighbors(g, ds[0][0]))
    ns2 = list(nx.neighbors(g, ds[1][0]))
    ns2o = [n for n in ns2 if n not in ns1]
    d2o.append((i,len(ns2o)))
    d2op.append((i,len(ns2o)/g.number_of_nodes()))
    d2ov.append((i,len(ns2o)/g.number_of_nodes()+len(ns2o)/maxnodes))

d1.sort(reverse=True, key=lambda x:x[1])
d2.sort(reverse=True, key=lambda x:x[1])
d2o.sort(reverse=True, key=lambda x:x[1])
d1p.sort(reverse=True, key=lambda x:x[1])
d2p.sort(reverse=True, key=lambda x:x[1])
d2op.sort(reverse=True, key=lambda x:x[1])
d1v.sort(reverse=True, key=lambda x:x[1])
d2v.sort(reverse=True, key=lambda x:x[1])
d2ov.sort(reverse=True, key=lambda x:x[1])

print(d1p[:3])
print(d2p[:3])
print(d2op[:3])
print(d1v[:3])
print(d2v[:3])
print(d2ov[:3])

json.dump({
    "1stdegreerank": d1,
    "2nddegreerank": d2,
    "only2ndnoderank": d2o,
    "1stdegreerank_per": d1p,
    "2nddegreerank_per": d2p,
    "only2ndnoderank_per": d2op,
    "1stdegreerank_score": d1v,
    "2nddegreerank_score": d2v,
    "only2ndnoderank_score": d2ov
}, open("../data/REDDIT-BINARY-nodedegreerank-class%d.json"%c_class, "w"))

for i,_ in locals()[c_resvar][:20]:
    g = nx.from_numpy_array(graphs0[i])
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    nx.write_edgelist(g,"../data/REDDIT-BINARY-edgelist/%d/%d.edgelist"%(c_resdir,i))

# graphs0 = list(graphs0)[:100]

# align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
#                     graphs0, padding=True, N=256)
# graphon0 = universal_svd(align_graphs_list, threshold=0.2)

# ax=sns.heatmap(graphon0)
# plt.show()

# graph00 = list(graphs0)[0]
# ax=sns.heatmap(graph00)
# plt.show()

# import networkx as nx
# import matplotlib.pyplot as plt
# def draw(i):
#     g=nx.read_edgelist("%d.edgelist"%i)
#     pos = nx.spring_layout(g)
#     nx.draw(g, pos=pos, node_size=6,alpha=0.1)
#     plt.show()