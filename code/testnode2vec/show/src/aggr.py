import os, sys, time
import random
import json, pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

from Tfunctions import *
sys.path.append(os.path.abspath("./"))
from testgraphon.graphonalignment.src.utility import *
import grl


def aggr_clus(g, embs):
    nodelist = list(g.nodes())
    embs_list = embs

    embs_np = np.array(embs_list)
    data = tsne(embs_np), umap(embs_np), pca(embs_np)
    embs = data[0]
    # cat = kmeans(embs, 2)
    cat = dbscan(embs, eps=1.2)
    nodedata = {nodelist[i]:cat[i] for i in range(len(nodelist))}
    nx.set_node_attributes(g, nodedata, "catgory")
    draw(g, node_color=cat)

    aggr_g = nx.snap_aggregation(g, node_attributes=("catgory",))

    aggr_g.remove_edges_from(nx.selfloop_edges(aggr_g))

    return aggr_g

def aggr_attr(g, data):
    nodelist = list(g.nodes())

    cat = nx.get_node_attributes(g, data)
    cat = [int(cat[node]) for node in nodelist]
    print(len(set(cat)))
    nodedata = {nodelist[i]:cat[i] for i in range(len(nodelist))}
    nx.set_node_attributes(g, nodedata, "catgory")
    draw(g, node_color=cat)

    aggr_g = nx.snap_aggregation(g, node_attributes=("catgory",))

    return aggr_g


if __name__ == "__main__":
    g = nx.read_edgelist("./testnode2vec/show/data/line-0.2.edgelist", nodetype=int)
    embs = readembeddings("./testnode2vec/show/data/line_classification_embs.json", ret=list)
    with open("./testnode2vec/show/data/line-nodedata.pickle", "rb") as f:
        nodes = pickle.load(f)
    g.add_nodes_from(nodes)

    g = nx.read_edgelist("./testnode2vec/show/data/ring-0.2.edgelist", nodetype=int)
    embs = readembeddings("./testnode2vec/show/data/ring_classification_embs.json", ret=list)
    with open("./testnode2vec/show/data/ring-nodedata.pickle", "rb") as f:
        nodes = pickle.load(f)
    g.add_nodes_from(nodes)

    g = nx.read_edgelist("./testnode2vec/show/data/ring.edgelist", nodetype=int)
    embs = grl.node2vec(g)

    aggr_g = aggr_clus(g, embs)
    # aggr_g = aggr_attr(g, 'layer')
    nx.set_edge_attributes(aggr_g, 1.0, 'weight')
    groups = nx.get_node_attributes(aggr_g, 'group')
    draw(aggr_g, node_size=[len(groups[node])*20 for node in list(aggr_g.nodes())])

    
