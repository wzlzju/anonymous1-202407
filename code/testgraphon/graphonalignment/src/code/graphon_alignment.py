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
from testcase import testcase
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
    random.seed(8)
    np.random.seed(8)
    g1 = nx.circular_ladder_graph(10)
    nx.set_edge_attributes(g1, 1, "weight")
    g2 = nx.complete_graph(100)
    nx.set_edge_attributes(g2, 1, "weight")

    links = addInterlayerLink_oneall2manyall(g1, g2)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, ax=axes[0][i], label="lambda = %.2f"%l)
        draw(new_g2, layout, ax=axes[1][i])
    plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
    plt.savefig('case0.png')
    plt.show()
    
def are_edges_equal(list1, list2):    
    # 将列表中的边转换为一组无序的元组，以消除方向性    
    set1 = {frozenset(edge) for edge in list1}    
    set2 = {frozenset(edge) for edge in list2}    
  
    # 找出不相同的边  
    diff1 = set1 - set2  
    diff2 = set2 - set1  
  
    # 打印不相同的边  
    # if diff1 or diff2:  
    #     print("不相同的边：")  
    #     if diff1:  
    #         print("在第一个列表中但不在第二个列表中的边：", [list(edge) for edge in diff1])  
    #     if diff2:  
    #         print("在第二个列表中但不在第一个列表中的边：", [list(edge) for edge in diff2])  
    # else:  
    #     print("所有边都相等。")  
  
    # 直接比较两个集合是否相等    
    return set1 == set2  

def testcase1():
    random.seed(88)
    np.random.seed(88)
    g1 = line(300, layer_ave=18)
    g2 = ring(360, 60, k=6, p=0.5)

    links = linkingLineRing(g1, g2)
    # print("links: {}".format(links))
    # print("type: {}".format(type(links)))
    g2 = graphgrowth(g2, p=[0.5,0.1,0.1], maxchildnum=[3,2,1], maxdepth=3)
    g2 = randomLinking(g2, 20)
    # g2 = graphgrowth(g2, p=[0.5,0.1,0.6], maxchildnum=[3,2,5], maxdepth=3)
    # g2 = randomLinking(g2, 10)

    with open("tmp/line-nodedata.pickle", "wb") as f:
        pickle.dump(g1.nodes(data=True), f)
    with open("tmp/ring-nodedata.pickle", "wb") as f:
        pickle.dump(g2.nodes(data=True), f)

    layout = "multilayer"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(2, len(ls),figsize=(20, 16))
    graph_series = {}
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        pos1, edges1 = draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], label="lambda = %.2f"%l)
        pos2, edges2 = draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[1][i])
        graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))
        
    # print("len: {}".format(g1.edges()))
    # print("len: {}".format(new_g1.edges()))
    

    with open("../data/sample-data/raw_graph_info.pickle", "wb") as f:
        pickle.dump(graph_series, f)
    with open("../data/sample-data/links.pickle", "wb") as f:
        pickle.dump(links, f)

    new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    nx.write_edgelist(new_g1, "tmp/line-0.2.edgelist")
    nx.write_edgelist(new_g2, "tmp/ring-0.2.edgelist")
    plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
    plt.savefig('case1.png', dpi = 600)
    plt.show()

def testcase2():
    random.seed(88)
    np.random.seed(88)
    g1 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/1/874.edgelist", nodetype=int)
    g1 = graph_sorting(g1)
    g2 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/2/697.edgelist", nodetype=int)
    g2 = graph_sorting(g2)

    d1 = sorted(list(nx.degree(g1)), reverse=True, key=lambda x:x[1])
    d2 = sorted(list(nx.degree(g2)), reverse=True, key=lambda x:x[1])

    links = addInterlayerLink_tmp1(g1, g2)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls))
    f1_node_color = lambda d: "#ff0000" if d[0] in [d1[0][0], d1[1][0]] else "#1f78b4"
    f2_node_color = lambda d: "#ff0000" if d[0] in [d2[0][0], d2[1][0]] else "#1f78b4"
        
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, ax=axes[0][i], label="lambda = %.2f"%l, f_node_color=f1_node_color)
        draw(new_g2, layout, ax=axes[1][i], f_node_color=f2_node_color)
        draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[2][i], f_node_color=f1_node_color)
        draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[3][i], f_node_color=f2_node_color)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase3():
    random.seed(88)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = line_cluster(200,10, 1)
    links = linkingDualLineToLineCluster(g1, g2, mode="dual")

    layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-linecluster-dual.png')
    plt.show()

def testcase4():
    random.seed(8)
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
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        if l==0.5:
            with open("tmp/pos_case4_g2_0.5.pickle", "wb") as f:
                pickle.dump(pos2, f)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-linecluster-interleaving.png')
    plt.show()

def testcase5():
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
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-randomcluster-cross.png')
    plt.show()

def testcase6():
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
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-randomcluster-line.png')
    plt.show()

def testcase7():
    random.seed(88)
    np.random.seed(88)
    
    g1 = line(1000, layer_ave=6)
    g2 = dual_line(500, 0.5, 1)

    links = linkingLineToDualLine(g1, g2)

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

    layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        start = time.time()
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        end = time.time()
        print(end-start)
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="multilayer", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="multilayer", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/line-dualline.png')
    plt.show()

def testcase8():
    random.seed(88)
    np.random.seed(88)
    
    g1 = random_cluster(200, 10)
    g2 = line_cluster(200,10, 1)

    links = linkingRandomClusterToLineCluster(g1, g2)

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

    layout = "default"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="default", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="default", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/randomcluster-linecluster.png')
    plt.show()

def testcase9():
    random.seed(8)
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
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        if l==0.5:
            with open("tmp/pos_case9_g2_0.5.pickle", "wb") as f:
                pickle.dump(pos2, f)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-linecluster-aggregated.png')
    plt.show()

def testcase10():
    random.seed(88)
    np.random.seed(88)
    
    g1 = dual_line(50, 0.5, 1)
    g2 = random_cluster(200, 10)

    links = linkingDualLineToRandomCluster(g1,g2, mode="cross2")
    # links = (links[1], links[0])

    # for i, (vi,vj) in enumerate(links[0]):
    #     if vi==48 or vi==49:
    #         links[0][i] = (vi,80)
    #         links[1][i] = (80,vi)

    layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-randomcluster-cross2.png')
    plt.show()

def testcase11():
    random.seed(88)
    np.random.seed(88)
    
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
    fig, axes = plt.subplots(4, len(ls), figsize=(20, 24))
    graph_series = {}
    for i,l in enumerate(ls):
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
        pos1, edges1 = draw2(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], node_color=node_color1, draw_nodes=none_node1, **kwargs)
        draw2(new_g1, layout, ax=axes[1][i], node_color=node_color1, **kwargs)
        pos2, edges2 = draw2(new_g2, layout="random", draw_edges_of_g=g2, ax=axes[2][i], node_color=node_color2, **kwargs)
        draw2(new_g2, layout="random", ax=axes[3][i], node_color=node_color2, **kwargs)
        # graph_series[i] = [{'pos': pos1, 'edges': edges1}, {'pos': pos2, 'edges': edges2}]
        # print("is equal: {}".format(are_edges_equal(g1.edges(), new_g1.edges())))

    # new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    plt.subplots_adjust(left=0,bottom=0.1,right=1,top=0.9,hspace=0.2,wspace=0)
    plt.savefig('./images/dualline-randomcluster-dual.png')
    plt.show()

def exp_figs():

    # 3-1 dualline to linecluster --dual

    figsize = (10, 6)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(3, ax=axes, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-1-1.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(3, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-1-2.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(8, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-1-3.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(3, ax=axes, lv=0.8, gno=2)
    plt.savefig('./images/exp3/exp 3-1-4.png')

    # 3-2 dualline to linecluster --interleaving

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(4, ax=axes, seed=88, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-2-1.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(4, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-2-2.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(8, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-2-3.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    with open("tmp/pos_case4_g2_0.5.pickle", "rb") as f:
        pos = pickle.load(f)
    testcase(4, ax=axes, lv=0.5, gno=2, pos=pos)
    plt.savefig('./images/exp3/exp 3-2-4.png')

    # 3-3 dualline to linecluster --aggragated

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(9, ax=axes, seed=88, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-3-1.png')

    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(9, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-3-2.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(8, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-3-3.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    with open("tmp/pos_case9_g2_0.5.pickle", "rb") as f:
        pos = pickle.load(f)
    testcase(9, ax=axes, lv=0.5, gno=2, pos=pos)
    plt.savefig('./images/exp3/exp 3-3-4.png')

    # 3-4 dualline to randomcluster --dual
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, seed=88, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-4-1.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-4-2.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-4-3.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, lv=0.5, gno=2)
    plt.savefig('./images/exp3/exp 3-4-4.png')

    # 3-5 dualline to randomcluster --line
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(6, ax=axes, seed=88, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-5-1.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(6, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-5-2.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-5-3.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(6, ax=axes, lv=0.5, gno=2)
    plt.savefig('./images/exp3/exp 3-5-4.png')

    # 3-6 dualline to randomcluster --cross
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(5, ax=axes, seed=88, lv=0., gno=1)
    plt.savefig('./images/exp3/exp 3-6-1.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(5, ax=axes, lv=0.5, gno=1)
    plt.savefig('./images/exp3/exp 3-6-2.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(11, ax=axes, lv=0., gno=2)
    plt.savefig('./images/exp3/exp 3-6-3.png')
    
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    testcase(5, ax=axes, lv=0.5, gno=2)
    plt.savefig('./images/exp3/exp 3-6-4.png')


if __name__ == "__main__":
    start = time.time()
    # testcase_temp()
    # testcase0()
    # testcase1()
    # testcase2()
    testcase7()   # line to dualline
    # testcase8()   # randomcluster to linecluster

    # testcase10()   # dualline to randomcluster --cross2

    # testcase3()   # dualline to linecluster --dual
    # testcase4()   # dualline to linecluster --interleaving
    # testcase9()   # dualline to linecluster --aggragated

    # testcase11()   # dualline to randomcluster --dual
    # testcase6()   # dualline to randomcluster --line
    # testcase5()   # dualline to randomcluster --cross

    # exp_figs()
    end = time.time()
    print(end-start)
