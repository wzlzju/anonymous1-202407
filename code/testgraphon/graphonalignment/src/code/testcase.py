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

def testcase_temp():
    random.seed(88)
    np.random.seed(88)
    
    # g1 = line(100, layer_ave=6)
    g1 = dual_line(50, 0.5, 1)
    # g1 = single_line(50)
    # g1 = random_cluster(200, 10)
    # g2 = ring(360, 60, k=6, p=0.5)
    # g2 = random_cluster(300, 10)
    # g2 = dual_line(50, 0.5, 1)
    # g2 = random_cluster(200, 10)
    g2 = line_cluster(200,6, 1)

    # g1 = generate_layer_1()
    # g2 = generate_layer_2()
    # g1 = generate_layer_2()
    # g2 = generate_layer_3()

    # links = linkingLineToDualLine(g1, g2)
    # links = linkingDualLineToRandomCluster(g1,g2, mode="linear")
    # links = linkingDualLineToRandomCluster(g1,g2, mode="cross")
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")
    # links = linkingDualLineToLineCluster(g1, g2, mode="single")
    # links = linkingSingleLineToLineCluster(g1, g2)
    # links = linkingRandomClusterToLineCluster(g1, g2)
    # links = linkingLayer1ToLayer2(g1, g2)
    # links = linkingLayer2ToLayer3(g1, g2)

    layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        pos1, edges1 = draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], label="lambda = %.2f"%l)
        draw(new_g1, layout, ax=axes[1][i])
        pos2, edges2 = draw(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i])
        draw(new_g2, layout="random", ax=axes[3][i])
        graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    # nx.write_edgelist(new_g1, "tmp/line-0.2.edgelist")
    # nx.write_edgelist(new_g2, "tmp/ring-0.2.edgelist")
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    # plt.savefig('./images/cases/Layer2ToLayer3.png')
    plt.show()

def testcase0():
    pass

def testcase1():
    pass

def testcase2():
    pass

def testcase3(ax, seed=88, lv=None, gno=None):
    random.seed(seed)
    np.random.seed(seed)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")

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

def testcase4(ax, seed=8, lv=None, gno=None, pos=None):
    random.seed(seed)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="interleaving")
    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

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

def testcase5(ax, seed=88, lv=None, gno=None):
    random.seed(88)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = random_cluster(200, 10)

    # links = linkingDualLineToRandomCluster(g1,g2, mode="linear")
    links = linkingDualLineToRandomCluster(g1,g2, mode="cross")

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

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

def testcase6(ax, seed=88, lv=None, gno=None):
    random.seed(88)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = random_cluster(200, 10)

    links = linkingDualLineToRandomCluster(g1,g2, mode="linear")
    # links = linkingDualLineToRandomCluster(g1,g2, mode="cross")

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

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

def testcase7():
    pass

def testcase8(ax, seed=88, lv=None, gno=None):
    random.seed(seed)
    np.random.seed(seed)
    
    g1 = random_cluster(200, 10)
    g2 = line_cluster(200,10, 1)

    links = linkingRandomClusterToLineCluster(g1, g2)

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

    layout = "default"
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
            # draw2(new_g1, layout, draw_edges_of_g=g1, node_color=node_color1, draw_nodes=none_node1, **kwargs)
            draw2(new_g2, layout="default", draw_edges_of_g=g2, ax=ax, node_color=node_color2, **kwargs)

def testcase9(ax, seed=8, lv=None, gno=None, pos=None):
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

def testcase10():
    pass

def testcase11(ax, seed=88, lv=None, gno=None):
    random.seed(seed)
    np.random.seed(seed)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = random_cluster(200, 10)

    # links = linkingDualLineToRandomCluster(g1,g2, mode="cross2")
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")
    # links = (links[1], links[0])

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

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

def testcase(case, **kwargs):
    {
        0: testcase0,
        1: testcase1,
        2: testcase2,
        3: testcase3,
        4: testcase4,
        5: testcase5,
        6: testcase6,
        7: testcase7,
        8: testcase8,
        9: testcase9,
        10: testcase10,
        11: testcase11,
    }[case](**kwargs)
