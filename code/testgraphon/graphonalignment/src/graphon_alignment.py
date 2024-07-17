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
from cmp import *

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

def aligngraphs(gs, links, ls):
    old_gs = copy.deepcopy(gs)
    gs = [graph_completion(g) for g in gs]
    new_gs = []
    
    for i in range(len(gs)):
        cg = gs[i]
        if i==0:
            operated_gs = [gs[0], gs[1]]
            clinks = {v:[] for v in cg}
            for vi, vj in links[0][0]:
                clinks[vi].append(vj)
            operated_links = [clinks]
            operated_ls = [ls[0]]
        elif i==len(gs)-1:
            operated_gs = [gs[-1], gs[-2]]
            clinks = {v:[] for v in cg}
            for vi, vj in links[-1][1]:
                clinks[vi].append(vj)
            operated_links = [clinks]
            operated_ls = [ls[-1]]
        else:
            operated_gs = [gs[i], gs[i-1], gs[i+1]]
            clinks1 = {v:[] for v in cg}
            for vi, vj in links[i-1][1]:
                clinks1[vi].append(vj)
            clinks2 = {v:[] for v in cg}
            for vi, vj in links[i][0]:
                clinks2[vi].append(vj)
            operated_links = [clinks1, clinks2]
            operated_ls = [ls[i]/2, ls[i]/2]
        new_weight = graphon_alignment(operated_gs, operated_links, operated_ls, diag=1.0)
        new_g = cg.copy()
        nx.set_edge_attributes(new_g, new_weight)
        new_g = graph_cleaning(new_g)
        new_gs.append(new_g)
    
    return new_gs

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
    # ls = [0.0, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    for i,l in enumerate(ls):
        kwargs = {
            "channel": "V",
            "max_V": 0.4,
            "min_V": 0.95,
            "color_S": 0,
        }
        kwargs2 = {
            "node_size": 30,
            "edge_width": 1,
            "alpha": 0.6,
        }
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, min_S=0., ax=axes[0][i], label="λ = %.2f"%l, **kwargs)
        draw(new_g2, layout, min_S=0., ax=axes[1][i], **kwargs)
    plt.subplots_adjust(left=0,bottom=0.268,right=1,top=0.8,hspace=0,wspace=0)
    plt.show()

def testcase1():
    random.seed(88)
    np.random.seed(88)
    g1 = line(100, layer_ave=6)
    g2 = ring(120, 60, k=6, p=0.5)

    links = linkingLineRing(g1, g2)
    g2 = graphgrowth(g2, p=[0.5,0.1,0.1], maxchildnum=[3,2,1], maxdepth=3)
    g2 = randomLinking(g2, 20)
    # g2 = graphgrowth(g2, p=[0.5,0.1,0.6], maxchildnum=[3,2,5], maxdepth=3)
    # g2 = randomLinking(g2, 10)
    with open("tmp/line-nodedata.pickle", "wb") as f:
        pickle.dump(g1.nodes(data=True), f)
    with open("tmp/ring-nodedata.pickle", "wb") as f:
        pickle.dump(g2.nodes(data=True), f)
    with open("tmp/line-ring-links.pickle", "wb") as f:
        pickle.dump(links, f)

    layout = "multilayer"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], label="λ = %.2f"%l)
        draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[1][i])
    new_g1, new_g2 = align2graphs(g1, g2, links, l=0.2)
    nx.write_edgelist(new_g1, "tmp/line-0.2.edgelist")
    nx.write_edgelist(new_g2, "tmp/ring-0.2.edgelist")
    plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
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
        draw(new_g1, layout, ax=axes[0][i], label="λ = %.2f"%l, f_node_color=f1_node_color)
        draw(new_g2, layout, ax=axes[1][i], f_node_color=f2_node_color)
        draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[2][i], f_node_color=f1_node_color)
        draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[3][i], f_node_color=f2_node_color)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase3():
    random.seed(88)
    np.random.seed(88)
    g1 = line(100, layer_ave=6)
    g2 = ring(120, 60, k=6, p=0.5)
    g3 = hierBlocks([[10,20,30],[25,25,25],[30,30,20]], [0.9, 0.05, 0.001])

    links12 = linkingLineRing(g1, g2)
    links23 = linkingRingBlocks(g2, g3)
    g2 = graphgrowth(g2, p=[0.5,0.1,0.1], maxchildnum=[3,2,1], maxdepth=3, weight=0.5)
    g2 = randomLinking(g2, 20, weight=0.5)
    g3 = graphgrowth(g3, p=[0.2,0.5,0.6], maxchildnum=[1,2,10], maxdepth=3, weight=0.5)
    g3 = randomLinking(g3, 10, weight=0.5)
    # g2 = graphgrowth(g2, p=[0.5,0.1,0.6], maxchildnum=[3,2,5], maxdepth=3)
    # g2 = randomLinking(g2, 10)
    with open("tmp/line-nodedata.pickle", "wb") as f:
        pickle.dump(g1.nodes(data=True), f)
    with open("tmp/ring-nodedata.pickle", "wb") as f:
        pickle.dump(g2.nodes(data=True), f)

    # layout = "multilayer"
    # # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    # ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # fig, axes = plt.subplots(2, len(ls))
    # for i,l in enumerate(ls):
    #     new_g1, new_g2 = aligngraphs([g1, g2], [links12], [l])
    #     draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], label="lambda = %.2f"%l)
    #     draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[1][i])
    # # new_g1, new_g2 = align2graphs(g1, g2, links12, l=0.2)
    # # nx.write_edgelist(new_g1, "tmp/line-0.2.edgelist")
    # # nx.write_edgelist(new_g2, "tmp/ring-0.2.edgelist")
    # plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
    # plt.show()

    # layout = "multilayer"
    # ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # fig, axes = plt.subplots(2, len(ls))
    # for i,l in enumerate(ls):
    #     new_g2, new_g3 = aligngraphs([g2, g3], [links23], [l])
    #     draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[0][i], label="lambda = %.2f"%l)
    #     draw(new_g3, "default", draw_edges_of_g=g3, ax=axes[1][i])
    # # new_g2, new_g3 = align2graphs(g2, g3, links23, l=0.2)
    # # nx.write_edgelist(new_g2, "tmp/line-0.2.edgelist")
    # # nx.write_edgelist(new_g3, "tmp/ring-0.2.edgelist")
    # plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
    # plt.show()

    layout = "multilayer"
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(3, len(ls))
    for i,l in enumerate(ls):
        new_g1, new_g2, new_g3 = aligngraphs([g1, g2, g3], [links12, links23], [l, l])
        draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[0][i], label="λ = %.2f"%l)
        draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[1][i])
        draw(new_g3, "default", draw_edges_of_g=g3, ax=axes[2][i])
    # new_g2, new_g3 = align2graphs(g2, g3, links23, l=0.2)
    # nx.write_edgelist(new_g2, "tmp/line-0.2.edgelist")
    # nx.write_edgelist(new_g3, "tmp/ring-0.2.edgelist")
    plt.subplots_adjust(left=0,bottom=0.4,right=1,top=0.8,hspace=0,wspace=0)
    plt.show()

def testcase4():
    random.seed(8)
    np.random.seed(8)
    g1 = hierBlocks([30 for i in range(4)], [1.0, 0.01])
    nx.set_edge_attributes(g1, 1, "weight")
    g2 = hierBlocks([60,60], [0.3, 0.005])
    nx.set_edge_attributes(g2, 1, "weight")

    links = linking2Blocks(g1, g2, [([str(i) for i in range(2)], [str(i) for i in range(1)]), ([str(i) for i in range(2,4)], [str(i) for i in range(1,2)])])

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # ls = [0.0, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    # fig, axes = plt.subplots(4, len(ls))
    for i,l in enumerate(ls):
        kwargs = {
            "channel": "V",
            "max_V": 0.4,
            "min_V": 0.95,
            "color_S": 0,
            "cmap": matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Set1").colors[:2])
        }
        kwargs2 = {
            "node_size": 30,
            "edge_width": 1,
            "alpha": 0.6,
        }
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, ax=axes[0][i], label="λ = %.2f"%l, node_color=[int(g1.nodes[node]['type'])//2 for node in g1.nodes], **kwargs)
        draw(new_g2, layout, ax=axes[1][i], node_color=[int(g2.nodes[node]['type']) for node in g2.nodes], **kwargs)
        # draw(new_g1, layout, draw_edges_of_g=g1, min_S=0., ax=axes[2][i])
        # draw(new_g2, layout, draw_edges_of_g=g2, min_S=0., ax=axes[3][i])
    plt.subplots_adjust(left=0,bottom=0.268,right=1,top=0.8,hspace=0,wspace=0)
    # plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase5():
    random.seed(8)
    np.random.seed(8)
    g1 = hierBlocks([30, 30], [0.9, 0.1])
    nx.set_edge_attributes(g1, 1, "weight")
    g2 = hierBlocks([30, 30], [0.9, 0.01])
    nx.set_edge_attributes(g2, 1, "weight")

    # links = linking2Blocks(g1, g2, [([str(i) for i in range(2)], [str(i) for i in range(1)]), ([str(i) for i in range(2,3)], [str(i) for i in range(1,3)])])
    # nx.set_node_attributes(g1, 0, 'rack')
    # nx.set_node_attributes(g2, 0, 'rack')

    g1, g2, links, linksr = addRackandFault(g1, g2, rackNum=3)
    links = (links, linksr)
    # print(links)
    # print(linksr)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # ls = [0.0, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0]
    fig, axes = plt.subplots(4, len(ls))
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, min_S=0., ax=axes[0][i], label="λ = %.2f"%l, node_size=30, edge_width=1, alpha=0.8, node_color=[g1.nodes[node]['rack'] for node in g1.nodes])
        draw(new_g2, layout, min_S=0., ax=axes[1][i], node_size=30, edge_width=1, alpha=0.8, node_color=[g2.nodes[node]['rack'] for node in g2.nodes])
        draw(new_g1, layout, draw_edges_of_g=g1, min_S=0., ax=axes[2][i], node_size=30, edge_width=1, alpha=0.8, node_color=[g1.nodes[node]['rack'] for node in g1.nodes])
        draw(new_g2, layout, draw_edges_of_g=g2, min_S=0., ax=axes[3][i], node_size=30, edge_width=1, alpha=0.8, node_color=[g2.nodes[node]['rack'] for node in g2.nodes])
    plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase6():
    random.seed(8)
    np.random.seed(8)
    g1 = hierHeteroBlocks([30, 30, 40], [
        [0.2, 0.2, 0.2],
        {(0,1): 0.01, (0,2): 0.01, (1,2): 0.01}
    ])
    nx.set_edge_attributes(g1, 1, "weight")
    g2 = hierHeteroBlocks([30, 60], [
        [0.8, 0.05],
        {(0,1): 0.01}
    ])
    nx.set_edge_attributes(g2, 1, "weight")

    # links = linking2Blocks(g1, g2, [([str(i) for i in range(2)], [str(i) for i in range(1)]), ([str(i) for i in range(2,3)], [str(i) for i in range(1,2)])])
    g1, g2, links, linksr = addRackandFault(g1, g2, rackNum=50)
    links = (links, linksr)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    # ls = [0.0, 0.02, 0.2, 0.5, 0.8, 0.98, 1.0]
    fig, axes = plt.subplots(4, len(ls))
    for i,l in enumerate(ls):
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        draw(new_g1, layout, min_S=0., ax=axes[0][i], label="λ = %.2f"%l, node_size=30, edge_width=1, alpha=0.8, node_color=[hsv2rgb(0.0,g1.nodes[node]['fault'],1.0) for node in g1.nodes()])
        draw(new_g2, layout, min_S=0., ax=axes[1][i], node_size=30, edge_width=1, alpha=0.8, node_color=[hsv2rgb(0.0,g2.nodes[node]['fault'],1.0) for node in g2.nodes()])
        draw(new_g1, layout, draw_edges_of_g=g1, min_S=0., ax=axes[2][i], node_size=30, edge_width=1, alpha=0.8, node_color=[hsv2rgb(0.0,g1.nodes[node]['fault'],1.0) for node in g1.nodes()])
        draw(new_g2, layout, draw_edges_of_g=g2, min_S=0., ax=axes[3][i], node_size=30, edge_width=1, alpha=0.8, node_color=[hsv2rgb(0.0,g2.nodes[node]['fault'],1.0) for node in g2.nodes()])
    plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase7():
    random.seed(88)
    np.random.seed(88)
    g1 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/1/874.edgelist", nodetype=int)
    g1 = graph_sorting(g1)
    g2 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/2/697.edgelist", nodetype=int)
    g2 = graph_sorting(g2)

    d1 = sorted(list(nx.degree(g1)), reverse=True, key=lambda x:x[1])
    d2 = sorted(list(nx.degree(g2)), reverse=True, key=lambda x:x[1])

    links = addInterlayerLinkbasedonDegreeRanking(g1, g2, 100)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    f1_node_color = lambda d: "#ff0000" if d[0] in [d1[i][0] for i in range(2)] else "#1f78b4"
    f2_node_color = lambda d: "#ff0000" if d[0] in [d2[i][0] for i in range(2)] else "#1f78b4"
        
    for i,l in enumerate(ls):
        kwargs = {
            "channel": "V",
            "max_V": 0.4,
            "min_V": 0.95,
            "color_S": 0,
        }
        new_g1, new_g2 = align2graphs(g1, g2, links, l=l)
        new_g1 = max_connected_component(new_g1)
        new_g2 = max_connected_component(new_g2)
        draw(new_g1, layout, ax=axes[0][i], label="λ = %.2f"%l, f_node_color=f1_node_color, **kwargs)
        draw(new_g2, layout, ax=axes[1][i], f_node_color=f2_node_color, **kwargs)
        # draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[2][i], f_node_color=f1_node_color, **kwargs)
        # draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[3][i], f_node_color=f2_node_color, **kwargs)
    plt.subplots_adjust(left=0,bottom=0.268,right=1,top=0.8,hspace=0,wspace=0)
    # plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

def testcase7c():
    random.seed(88)
    np.random.seed(88)
    g1 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/1/874.edgelist", nodetype=int)
    g1 = graph_sorting(g1)
    g2 = nx.read_edgelist("../data/REDDIT-BINARY-edgelist/2/697.edgelist", nodetype=int)
    g2 = graph_sorting(g2)

    d1 = sorted(list(nx.degree(g1)), reverse=True, key=lambda x:x[1])
    d2 = sorted(list(nx.degree(g2)), reverse=True, key=lambda x:x[1])

    links = addInterlayerLinkbasedonDegreeRanking(g1, g2, 100)
    # links = addInterlayerLink_tmp1(g1, g2)
    m1, m2 = getCorrespondences(g1, g2, *links)

    layout = "default"
    # ls = [0.0, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
    ls = [0.0, 0.2, 0.5, 0.8, 1.0]
    fig, axes = plt.subplots(2, len(ls))
    f1_node_color = lambda d: "#ff0000" if d[0] in [d1[i][0] for i in range(2)] else "#1f78b4"
    f2_node_color = lambda d: "#ff0000" if d[0] in [d2[i][0] for i in range(2)] else "#1f78b4"
        
    for i,l in enumerate(ls):
        kwargs = {
            "channel": "V",
            "max_V": 0.4,
            "min_V": 0.95,
            "color_S": 0,
        }
        new_g1, new_g2 = hard_alignment(g1, g2, m1, m2, l=1-l)
        # new_g1 = graph_cleaning(new_g1)
        # new_g2 = graph_cleaning(new_g2)
        # new_g1 = max_connected_component(new_g1)
        # new_g2 = max_connected_component(new_g2)
        draw(new_g1, layout, ax=axes[0][i], label="λ = %.2f"%l, f_node_color=f1_node_color, **kwargs)
        draw(new_g2, layout, ax=axes[1][i], f_node_color=f2_node_color, **kwargs)
        # draw(new_g1, layout, draw_edges_of_g=g1, ax=axes[2][i], f_node_color=f1_node_color, **kwargs)
        # draw(new_g2, layout, draw_edges_of_g=g2, ax=axes[3][i], f_node_color=f2_node_color, **kwargs)
    plt.subplots_adjust(left=0,bottom=0.268,right=1,top=0.8,hspace=0,wspace=0)
    # plt.subplots_adjust(left=0,bottom=0,right=1,top=0.95,hspace=0,wspace=0)
    plt.show()

if __name__ == "__main__":
    testcase0()   # exp 2-1
    # testcase1()
    # testcase2()
    # testcase3()
    # testcase4()   # exp 2-2
    # testcase5()
    # testcase6()
    # testcase7()   # exp 2-3 social network
    # testcase7c()   # exp 2-4 social network