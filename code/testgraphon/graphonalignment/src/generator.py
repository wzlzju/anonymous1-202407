import os, sys
import json, pickle
import random
import copy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from interlayer import *
from utility import *


def line(m, layer_ave=3, p=1.0):
    # line stream graph
    beginning = 1
    ending = 1
    intervals = m-beginning-ending
    layer_num = intervals//layer_ave + 1
    splits_num = layer_num-1
    splits = [0] + random.sample(list(range(1,intervals)), splits_num) + [intervals]
    splits.sort()
    subset_sizes = [beginning] + list(np.diff(np.array(splits))) + [ending]
    g = nx.Graph()
    nodeID = 0
    for layer,subset_size in enumerate(subset_sizes):
        for offset in range(subset_size):
            g.add_node(nodeID+offset, layer=layer)
        nodeID += subset_size
    for vi, vj in [(x,y) for x in range(0,m-1) for y in range(x+1,m)]:
        if g.nodes[vj]['layer']-g.nodes[vi]['layer'] == 1 and random.random()<=p:
            g.add_edge(vi, vj, weight=1.0)

    return g

def ring(m, n=None, k=6, p=1.0):
    if not n:
        n = m

    g = nx.Graph()
    g.add_node(0, branch=0, layer=0)
    nodeID = 1
    for i in range(1,m-1):
        g.add_node(nodeID, branch=1, layer=i)
        nodeID += 1
    for i in range(1,n+1):
        g.add_node(nodeID, branch=2, layer=i)
        nodeID += 1
    g.add_node(nodeID, branch=3, layer=m+n-1)

    for i in range(m+n):
        if g.nodes[i+1]['branch'] <= 1:
            g.add_edge(i, i+1, weight=1.0)
        else:
            g.add_edge(i, m+n-1, weight=1.0)
            break
        for j in range (2, k//2+1):
            if random.random()<=p:
                if g.nodes[i+j]['branch'] <= 1:
                    g.add_edge(i, i+j, weight=1.0)
                else:
                    g.add_edge(i, m+n-1, weight=1.0)
                    break
    for i in range(m+n-1,0,-1):
        if g.nodes[i-1]['branch'] >= 2:
            g.add_edge(i-1, i, weight=1.0)
        else:
            g.add_edge(0, i, weight=1.0)
            break
        for j in range (2, k//2+1):
            if random.random()<=p:
                if g.nodes[i-j]['branch'] >= 2:
                    g.add_edge(i-j, i, weight=1.0)
                else:
                    g.add_edge(0, i, weight=1.0)
                    break
    return g

def linkingLineRing(g1, g2, mode="oneall2manyall"):

    node_list1 = list(g1.nodes())[1:-1]
    node_list21 = list()
    node_list22 = list()
    for node in g2.nodes():
        if g2.nodes[node]['branch'] == 1:
            node_list21.append(node)
        elif g2.nodes[node]['branch'] == 2:
            node_list22.append(node)

    link1, linkr1 = linkingLists_oneall2manyall(node_list1, node_list21)
    link2, linkr2 = linkingLists_oneall2manyall(node_list1, node_list22)
    link = [(0, 0)] + link1 + link2 + [(len(g1)-1, len(g2)-1)]
    linkr = [(0, 0)] + linkr1 + linkr2 + [(len(g2)-1, len(g1)-1)]

    return link, linkr

def linkingRingBlocks(g1, g2, mode="oneall2manyall"):
    node_list11 = list()
    node_list12 = list()
    for node in g1.nodes():
        if g1.nodes[node]['branch'] == 1:
            node_list11.append(node)
        elif g1.nodes[node]['branch'] == 2:
            node_list12.append(node)
    node_list21 = list()
    node_list22 = list()
    for node in g2.nodes():
        type = g2.nodes[node]['type']
        branch = type.split('-')[0]
        if branch in ['0', '1']:
            node_list21.append(node)
        elif branch in ['2']:
            node_list22.append(node)

    link1, linkr1 = linkingLists_oneall2manyall(node_list11, node_list21)
    link2, linkr2 = linkingLists_oneall2manyall(node_list12, node_list22)
    link = [(0, 0)] + link1 + link2 + [(len(g1)-1, len(g2)-1)]
    linkr = [(0, 0)] + linkr1 + linkr2 + [(len(g2)-1, len(g1)-1)]

    return link, linkr

def linking2Blocks(g1, g2, linking_relationships, mode="oneall2manyall"):
    # linking_relationships: [(['0','1','2'], ['0']), (['3','4','5'], ['1'])]
    node_lists1 = []
    node_lists2 = []
    for linking_relationship in linking_relationships:
        c_node_list1 = []
        c_node_list2 = []
        for node in g1.nodes():
            if g1.nodes[node]['type'] in linking_relationship[0]:
                c_node_list1.append(node)
        for node in g2.nodes():
            if g2.nodes[node]['type'] in linking_relationship[1]:
                c_node_list2.append(node)
        node_lists1.append(c_node_list1)
        node_lists2.append(c_node_list2)
    link = []
    linkr = []
    for node_list1, node_list2 in zip(node_lists1, node_lists2):
        links = linkingLists_oneall2manyall(node_list1, node_list2)
        link += links[0]
        linkr += links[1]
    
    return link, linkr

def graphgrowth(g, p, maxchildnum=3, maxdepth=6, weight=1.0):
    if not is_iterable(p):
        p = [p for _ in range(maxdepth)]
    if not is_iterable(maxchildnum):
        maxchildnum = [maxchildnum for _ in range(maxdepth)]
    for node in g.nodes():
        g.nodes[node]['depth'] = 0
    p_nodes = list(g.nodes())
    c_nodes = []
    nodeID = p_nodes[-1]+1
    for d in range(maxdepth):
        for p_node in p_nodes:
            for child_i in range(maxchildnum[d]):
                if random.random() <= p[d]:
                    c_node = nodeID
                    g.add_node(c_node)
                    for key in g.nodes[p_node].keys():
                        g.nodes[c_node][key] = g.nodes[p_node][key]
                    g.nodes[c_node]['depth'] = d+1
                    c_nodes.append(c_node)

                    g.add_edge(p_node, c_node, weight=weight)

                    nodeID += 1
        p_nodes = c_nodes
        c_nodes = []
    return g

def randomLinking(g, n=None, weight=1):
    if not n:
        n = g.number_of_nodes()
    non_edges = list(nx.non_edges(g))
    if n > len(non_edges):
        n = len(non_edges)
    if n == 0:
        return g
    new_edges = random.sample(non_edges, n)
    g.add_edges_from(new_edges, weight=weight)
    return g

def hierBlocksnolabelRec(sizes, ps):
    # [[20,30,40],[10,15,30]]
    c_block_sizes, c_p_mat = [], np.array([[]], dtype=float)
    for size in sizes:
        if type(size) is int:
            block_sizes, p_mat = [size], np.array([[ps[0]]], dtype=float)
        else:
            block_sizes, p_mat = hierBlocksnolabelRec(size, ps[:-1])
        tmp_block_sizes = c_block_sizes + block_sizes
        tmp_p_mat = np.array([[ps[-1] for _ in range(len(tmp_block_sizes))] for _ in range(len(tmp_block_sizes))], dtype=float)
        tmp_p_mat[:len(c_block_sizes), :len(c_block_sizes)] = c_p_mat
        tmp_p_mat[len(c_block_sizes):len(tmp_block_sizes), len(c_block_sizes):len(tmp_block_sizes)] = p_mat
        c_block_sizes = tmp_block_sizes
        c_p_mat = tmp_p_mat

    return c_block_sizes, c_p_mat

def hierBlocksnolabel(sizes, ps):
    block_sizes, p_mat = hierBlocksnolabelRec(sizes, ps)
    g = nx.stochastic_block_model(block_sizes, p_mat)
    return g

def hierBlocksRec(sizes, ps, depth):
    g = nx.empty_graph()
    for i, size in enumerate(sizes):
        if type(size) is int:
            cg = nx.stochastic_block_model([size], [[ps[0]]])
            nx.set_node_attributes(cg, str(i), "type")
        else:
            cg = hierBlocksRec(size, ps, depth+1)
            for node in cg.nodes(data='type'):
                cg.nodes[node[0]]['type'] = str(i) + "-" + node[1]
        c_nodenum = len(list(g.nodes()))
        for j, node in enumerate(cg.nodes(data='type')):
            g.add_node(c_nodenum+node[0], type=node[1])
        nodelist = list(g.nodes())
        for j in range(0, len(nodelist)-1):
            for k in range(j+1, len(nodelist)):
                if j >= c_nodenum:
                    if cg.has_edge(k-c_nodenum, j-c_nodenum):
                        g.add_edge(k, j, weight=1)
                elif k >= c_nodenum:
                    if random.random() <= ps[-depth]:
                        g.add_edge(k, j, weight=1)
                else:
                    pass
    
    return g

def hierBlocks(sizes, ps):
    g = hierBlocksRec(sizes, ps, 1)
    return g

def getBlockId(sizes, node):
    total_size = 0
    for i, size in enumerate(sizes):
        total_size += size
        if node < total_size:
            return i

def hierHeteroBlocksRec(sizes, ps, depth):
    g = nx.empty_graph()
    for i, size in enumerate(sizes):
        if type(size) is int:
            cg = nx.stochastic_block_model([size], [[ps[0][i]]])
            nx.set_node_attributes(cg, str(i), "type")
        else:
            cg = hierBlocksRec(size, ps, depth+1)
            for node in cg.nodes(data='type'):
                cg.nodes[node[0]]['type'] = str(i) + "-" + node[1]
        c_nodenum = len(list(g.nodes()))
        for j, node in enumerate(cg.nodes(data='type')):
            g.add_node(c_nodenum+node[0], type=node[1])
        nodelist = list(g.nodes())
        for j in range(0, len(nodelist)-1):
            for k in range(j+1, len(nodelist)):
                if j >= c_nodenum:
                    if cg.has_edge(k-c_nodenum, j-c_nodenum):
                        g.add_edge(k, j, weight=1)
                elif k >= c_nodenum:
                    if random.random() <= ps[-depth][(getBlockId(sizes,j),getBlockId(sizes,k))]:
                        g.add_edge(k, j, weight=1)
                else:
                    pass
    
    return g

def hierHeteroBlocks(sizes, ps):
    g = hierHeteroBlocksRec(sizes, ps, 1)
    return g


if __name__ == "__main__":
    g = hierBlocks([[10,20,30,40],[25,25,25,25,25,25],[30,30,20,20,30]], [0.9, 0.05, 0.001])
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, node_size=10)
    # plt.axis('equal')  
    plt.show()

    g = nx.stochastic_block_model([20,30,40],[[0.9,0.1,0.1],[0.1,0.9,0.1],[0.1,0.1,0.9]])
    pos = nx.spring_layout(g)
    nx.draw(g, pos=pos, node_size=10)
    # plt.axis('equal')  
    plt.show()

    g = line(100)
    pos = nx.multipartite_layout(g, subset_key="layer")
    pos = nx.spring_layout(g, pos=pos)
    nx.draw(g, pos=pos, node_size=10)
    # plt.axis('equal')  
    plt.show()

    g = ring(120, 60, 6, 0.5)
    pos = nx.multipartite_layout(g, subset_key="layer")
    pos = nx.spring_layout(g, pos=pos)
    nx.draw(g, pos=pos, node_size=10)
    # plt.axis('equal')  
    plt.show()