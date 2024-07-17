import os, sys
import time, datetime
import json, pickle
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from generator import *
from utility import *
from interlayer import *
# from generating_layer_1 import generate_layer_1
# from generating_layer_2 import generate_layer_2
# from generating_layer_3 import generate_layer_3

import logging
# logging.basicConfig(filename='tmp/debug.log', level=logging.DEBUG)

def graphon_alignment(graphons, links, l=None, diag=1.0):
    graphons = [g.copy() for g in graphons]
    target_g = graphons[0]
    linking_gs = graphons[1:]
    edges = target_g.edges()
    old_weight = graph_completion(target_g, ret="adj", diag=diag)
    linking_weights = [graph_completion(g, ret="adj", diag=diag) for g in linking_gs]
    if not l:
        l = [1/(len(linking_gs)+1) for _ in range(len(linking_gs))]
    if not is_iterable(l):
        l = [l]
    l0 = 1-sum(l)
    if len(linking_gs) != len(links) or len(linking_gs) != len(l):
        raise Exception("Error in <graphon_align>: unaligned inputs (%d, %d, %d)." % (len(linking_gs), len(links), len(l)))
    new_weight = {}
    for e in edges:
        vi, vj = e
        w = old_weight[vi][vj]*l0
        for link, linking_weight, linking_l in zip(links, linking_weights, l):
            wg = []
            for linking_vi, linking_vj in [(x, y) for x in link[vi] for y in link[vj]]:
                wg.append(linking_weight[linking_vi][linking_vj])
            if len(wg)>0:
                wg = sum(wg)/len(wg)
            else:
                wg = old_weight[vi][vj]
            w += wg*linking_l
        new_weight[e] = {"weight": w}
    return new_weight

def graphon_alignment_shortest_path(graphons, links, l=None, diag=1.0):
    graphons = [g.copy() for g in graphons]
    target_g = graphons[0]
    linking_gs = graphons[1:]
    edges = target_g.edges()
    old_weight = graph_completion(target_g, ret="adj", diag=diag)
    linking_weights = [graph_completion(g, ret="adj", diag=diag) for g in linking_gs]
    shortest_path_matrix = shortest_path(target_g, ret="adj")
    if not l:
        l = [1/(len(linking_gs)+1) for _ in range(len(linking_gs))]
    if not is_iterable(l):
        l = [l]
    l0 = 1-sum(l)
    if len(linking_gs) != len(links) or len(linking_gs) != len(l):
        raise Exception("Error in <graphon_align>: unaligned inputs (%d, %d, %d)." % (len(linking_gs), len(links), len(l)))
    new_weight = {}
    for e in edges:
        vi, vj = e
        w = old_weight[vi][vj]*l0
        for link, linking_weight, linking_l in zip(links, linking_weights, l):
            wg = []
            for linking_vi, linking_vj in [(x, y) for x in link[vi] for y in link[vj]]:
                wg.append(linking_weight[linking_vi][linking_vj])
            wg = sum(wg)/len(wg)
            w += wg*linking_l/shortest_path_matrix[vi][vj]
        new_weight[e] = {"weight": w}
    return new_weight

def align2graphs(g1, g2, links, l=0.5):
    old_g1 = g1
    old_g2 = g2
    g1 = graph_completion(g1)
    g2 = graph_completion(g2)
    links_list, r_links_list = links

    links = {v:[] for v in g1}
    for vi, vj in links_list:
        links[vi].append(vj)
    new_weight1 = graphon_alignment([g1, g2], [links], l=[l], diag=1.0)

    links = {v:[] for v in g2}
    for vi, vj in r_links_list:
        links[vi].append(vj)
    new_weight2 = graphon_alignment([g2, g1], [links], l=[l], diag=1.0)

    # print(new_weight1)
    # print(new_weight2)
    # print("lambda = %.2f"%l)

    new_g1 = g1.copy()
    nx.set_edge_attributes(new_g1, new_weight1)
    new_g1 = graph_cleaning(new_g1)
    new_g2 = g2.copy()
    nx.set_edge_attributes(new_g2, new_weight2)
    new_g2 = graph_cleaning(new_g2)

    return new_g1, new_g2

def testcase3(ax, seed=88, lv=None, gno=None, layout=None):
    random.seed(seed)
    np.random.seed(seed)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")

    if not layout:
        layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    graph_series = {}
    for i,l in enumerate(ls):
        if l != lv:
            continue
        kwargs = {
            "channel": "V",
            "max_V": 0.675,
            "min_V": 0.95,
            "color_S": 0,
            # "cmap": matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Accent").colors[:6]),
        }
        link_dict = {v:[] for v in g1}
        for vi, vj in links[0]:
            link_dict[vi].append(vj)
        node_color2 = [g2.nodes[node]['cluster'] for node in g2.nodes()]
        node_color1 = [None if link_dict[node] == [] else node_color2[link_dict[node][0]] for node in g1.nodes()]
        none_node1 = [node for node in g1.nodes() if node_color1[node] is None]
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        if gno == 1:
            draw2(new_g1, layout, draw_edges_of_g=g1, ax=ax, node_color=node_color1, draw_nodes=none_node1, **kwargs)
        elif gno == 2:
            draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)

def testcase4(ax, seed=8, lv=None, gno=None, pos=None, layout=None):
    random.seed(seed)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="interleaving")
    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

    if not layout:
        layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    graph_series = {}
    for i,l in enumerate(ls):
        if l != lv:
            continue
        kwargs = {
            "channel": "V",
            "max_V": 0.675,
            "min_V": 0.95,
            "color_S": 0,
            # "cmap": matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Accent").colors[:6]),
        }
        link_dict = {v:[] for v in g1}
        for vi, vj in links[0]:
            link_dict[vi].append(vj)
        node_color2 = [g2.nodes[node]['cluster'] for node in g2.nodes()]
        node_color1 = [None if link_dict[node] == [] else node_color2[link_dict[node][0]] for node in g1.nodes()]
        none_node1 = [node for node in g1.nodes() if node_color1[node] is None]
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        if gno == 1:
            draw2(new_g1, layout, draw_edges_of_g=g1, ax=ax, node_color=node_color1, draw_nodes=none_node1, **kwargs)
        elif gno == 2:
            if pos:
                draw2(new_g2, pos=pos, draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)
            else:
                draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)

def testcase9(ax, seed=8, lv=None, gno=None, pos=None, layout=None):
    random.seed(seed)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")
    for i in g1.nodes():
        if i >= 100:
            j = random.choice(list(range(180,200)))
            links[0].append((i,j))
            links[1].append((j,i))
    if not layout:
        layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    graph_series = {}
    for i,l in enumerate(ls):
        if l != lv:
            continue
        kwargs = {
            "channel": "V",
            "max_V": 0.675,
            "min_V": 0.95,
            "color_S": 0,
            # "cmap": matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Accent").colors[:6]),
        }
        link_dict = {v:[] for v in g1}
        for vi, vj in links[0]:
            link_dict[vi].append(vj)
        node_color2 = [g2.nodes[node]['cluster'] for node in g2.nodes()]
        node_color1 = [None if link_dict[node] == [] else node_color2[link_dict[node][0]] for node in g1.nodes()]
        none_node1 = [node for node in g1.nodes() if node_color1[node] is None]
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        if gno == 1:
            draw2(new_g1, layout, draw_edges_of_g=g1, ax=ax, node_color=node_color1, draw_nodes=none_node1, **kwargs)
        elif gno == 2:
            if pos:
                draw2(new_g2, pos=pos, draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)
            else:
                draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)

def exp_layout():
    figsize = (15, 6)
    layouts = ["random", "cycle", "multilayer", "kamada_kawai*random", "kamada_kawai*cycle", "kamada_kawai*multilayer"]
    seeds = [888, 88, 8]
    fig, axes = plt.subplots(3, len(layouts), figsize=figsize)
    for i, layout in enumerate(layouts):
        testcase3(ax=axes[0][i], seed=seeds[i%len(seeds)], lv=0.5, gno=1, layout=layout)
    for i, layout in enumerate(layouts):
        testcase4(ax=axes[1][i], seed=seeds[i%len(seeds)], lv=0.5, gno=1, layout=layout)
    for i, layout in enumerate(layouts):
        testcase9(ax=axes[2][i], seed=seeds[i%len(seeds)], lv=0.5, gno=1, layout=layout)
    plt.show()



if __name__ == "__main__":
    exp_layout()