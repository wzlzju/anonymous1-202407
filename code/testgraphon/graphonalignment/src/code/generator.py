import os, sys
import json, pickle
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from interlayer import *
from utility import *
from utils import get_cluster_mapping

def single_line(n, max_branch = 5):
    g = nx.Graph()
    
    branch_list = [random.randint(0,max_branch) for i in range(0, n)]
    
    branch_node_id = 0
    
    for i in range(n):
        g.add_node(i, cluster=i, layer = i %n)
        
    for i in range(0, n-1):
        g.add_edge(i, i+1)
        
    for i in range(0, n):
        for j in range(branch_list[i]):
            node_id = n + branch_node_id
            branch_node_id += 1
            g.add_node(node_id, cluster=i, layer=i%n)
            g.add_edge(i, node_id)
            
    # 为所有边添加权重属性  
    for (u, v) in g.edges():  
        g[u][v]['weight'] = 1.0  
        
    return g

def dual_line(n, inner_p = 0.5, max_branch = 5):
    inner_links = int(n * inner_p) if int(n * inner_p) > 0 else 1
    interval = n//inner_links
    g = nx.Graph()
    
    branch_list = [random.randint(0,max_branch) for i in range(0, 2*n)]
    
    branch_node_id = 0
    
    for i in range(2*n):
        g.add_node(i, cluster=i, layer = i %n)
        
    for i in range(0, n-1):
        g.add_edge(i, i+1)
        # if i+2 < n:
        #     g.add_edge(i, i+2)
    for i in range(n,2*n-1):
        g.add_edge(i, i+1)
        # if i+2 < 2*n:
        #     g.add_edge(i, i+2)
    for i in range(0, n, interval):
        g.add_edge(i, i+n)
        
    for i in range(0, 2*n):
        for j in range(branch_list[i]):
            node_id = 2*n + branch_node_id
            branch_node_id += 1
            g.add_node(node_id, cluster=i, layer=i%n)
            g.add_edge(i, node_id)
            
    # 为所有边添加权重属性  
    for (u, v) in g.edges():  
        g[u][v]['weight'] = 1.0  
        
    return g

  
def line_cluster(node_num, cluster_num, inter_cluster_edges = 4):  
    sizes = [node_num // cluster_num] * cluster_num  # 创建大小相等的社区  
  
    # 概率参数  
    p_intra = 0.5  # 社区内部节点相连的概率  
    p_inter = 0.0  # 社区之间节点相连的概率设为0，即不允许簇团之间直接连接  
  
    # 使用 random_partition_graph 生成具有社区结构的图  
    g = nx.random_partition_graph(sizes, p_intra, p_inter)  
  
    # 为每个节点添加所属簇团信息  
    for cluster_id, size in enumerate(sizes):  
        nodes_of_cluster = range(cluster_id * size, (cluster_id + 1) * size)  
        nx.set_node_attributes(g, {node: {'cluster': cluster_id, 'layer': cluster_id} for node in nodes_of_cluster})  
  
    # 将各簇团首尾相连，形成线型  
    for i in range(cluster_num - 1):  
        for _ in range(inter_cluster_edges):  
            for _ in range(random.choice([3,4,5])):
                start_node = (i * sizes[0] + int(sizes[0] * random.uniform(0, 1)))  # 上一个簇团的最后一个节点  
                end_node = ((i + 1) * sizes[0] + int(sizes[0] * random.uniform(0, 1)))  # 下一个簇团的第一个节点  
                g.add_edge(start_node, end_node)  
  
    # 为所有边添加权重属性  
    for (u, v) in g.edges():  
        g[u][v]['weight'] = 1.0  
  
    return g  
  
def random_cluster(node_num, cluster_num):    
    sizes = [node_num // cluster_num] * cluster_num  # 创建10个大小相等的社区      
      
    # 概率参数      
    p_intra = 0.5  # 社区内部节点相连的概率      
    p_inter = 0.005  # 社区之间节点相连的概率      
        
    # 使用 random_partition_graph 生成具有社区结构的图      
    g = nx.random_partition_graph(sizes, p_intra, p_inter)      
  
    # 为每个节点添加簇团信息  
    for i in range(cluster_num):  
        cluster_id = i  # 簇团编号从 0 开始  
        for node in range(i * (node_num // cluster_num), (i + 1) * (node_num // cluster_num)):  
            g.nodes[node]['cluster'] = cluster_id  
            g.nodes[node]['class'] = cluster_id 
      
    # 为所有边添加权重属性    
    for (u, v) in g.edges():    
        g[u][v]['weight'] = 1.0    
    
    return g   


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

def linkingLayer1ToLayer2(graphon1, graphon2):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)

    len1 = len(nodes_list1)
    len2 = len(nodes_list2)//2

    print("len: {}    {}".format(len1, len2))

    links = []
    linksr = []

    interval = len2//len1

    for i in range(len1):
        for j in range(i*interval, (i+1)*interval):
            links.extend([(i, j), (i, j+len2)])
            linksr.extend([(j, i), (j+len2, i)])

    return links, linksr

def linkingLayer2ToLayer3(graphon1, graphon2):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)

    len1 = len(nodes_list1)
    len2 = len(nodes_list2)

    print("len: {}    {}".format(len1, len2))

    links = []
    linksr = []

    interval = len2/len1

    for i in range(len1):
        for j in range(int(i*interval), int((i+1)*interval)):
            links.extend([(i, j)])
            linksr.extend([(j, i)])

    return links, linksr



# TODO：当前只能处理长度一样的
# 双线型只考虑主干节点
def linkingLineToDualLine(graphon1, graphon2):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)
    
    nodes_id_list1 = set()
    nodes_id_list2 = set()
    
    for key, _ in nodes_list1:
        nodes_id_list1.add(key)
    for key, _ in nodes_list2:
        nodes_id_list2.add(nodes_list2[key]['cluster'])
    
    links = []
    linksr = []
    
    for key, _ in nodes_list2: 
        linksr.append((key, nodes_list2[key]['cluster']))
        links.append((nodes_list2[key]['cluster'], key))

    return links, linksr

# 双线型只考虑主干节点
def linkingDualLineToRandomCluster(graphon1, graphon2, mode = "linear"):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)
    
    nodes_id_list1 = set()
    nodes_id_list2 = set()
    
    for key, _ in nodes_list1:
        nodes_id_list1.add(nodes_list1[key]['cluster'])
    for key, _ in nodes_list2:
        nodes_id_list2.add(key)
        
    rev = False
    links = []
    linksr = []
    
    if mode == "linear":
        if len(nodes_id_list1) > len(nodes_id_list2):
            rev = True
        interval = len(nodes_id_list2)//len(nodes_id_list1) if not rev else len(nodes_id_list1)//len(nodes_id_list2)

        if rev:
            nodes_list1 = nodes_id_list2
            nodes_list2 = nodes_id_list1
        else:
            nodes_list1 = nodes_id_list1
            nodes_list2 = nodes_list2
        for id in nodes_list1:
            for i in range(id, id+interval):
                links.append((id, id + i))
                linksr.append((id + i, id))
            
    elif mode == "cross":
        cluster_mapping = get_cluster_mapping(graphon2)
        start_cluster_id = 0
        round = 0
        cluster_num = len(cluster_mapping)
        for node_id in nodes_id_list1:
            links.append((node_id, cluster_mapping[start_cluster_id][round]))
            linksr.append((cluster_mapping[start_cluster_id][round], node_id))
            start_cluster_id += 1
            if start_cluster_id%cluster_num == 0:
                start_cluster_id = 0
                round+=1
            if round >= len(cluster_mapping[0]):
                break
    
    elif mode == "cross2":
        cluster_mapping = get_cluster_mapping(graphon2)
        interval12 = len(nodes_id_list2)//len(nodes_id_list1)
        cluster_num = len(cluster_mapping)
        interval = len(nodes_id_list1)//cluster_num
        # print(len(nodes_id_list1),cluster_num,cluster_mapping)
        for i in nodes_id_list1:
            for j in range(interval12):
                links.append((i, cluster_mapping[i%interval][j]))
                linksr.append((cluster_mapping[i%interval][j], i))

    if(rev):
        temp = links
        links = linksr
        linksr = links
        
    return links, linksr

# 双线型只考虑主干节点
def linkingDualLineToLineCluster(graphon1, graphon2, mode="dual"):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)
    
    nodes_id_list1 = set()
    nodes_id_list2 = set()
    
    for key, _ in nodes_list1:
        nodes_id_list1.add(nodes_list1[key]['cluster'])
    for key, _ in nodes_list2:
        nodes_id_list2.add(nodes_list2[key]['cluster'])
        
    len1 = len(nodes_id_list1)
    if mode == "dual":
        len1 //= 2
    len2 = len(nodes_id_list2)
    
    links = []
    linksr = []
    
    cluster_mapping_1 = get_cluster_mapping(graphon1)
    cluster_mapping_2 = get_cluster_mapping(graphon2)
    
    # print("mapping1: {}".format(cluster_mapping_1))
    # print("mapping2: {}".format(cluster_mapping_2))
    
    interval = len1 // len2
    start_node = 0
    end_node = start_node + interval
    for i in range(len2):
        nodes_in_cluster2 = cluster_mapping_2[i]
        # print(nodes_in_cluster2)
        if mode == "single" or mode == "dual":
            for j in range(start_node, end_node):
                idx2 = random.randint(0,len(nodes_in_cluster2)-1)
                links.append((j, nodes_in_cluster2[idx2]))
                linksr.append((nodes_in_cluster2[idx2], j))
        if mode == "dual":
            for j in range(start_node + len1, end_node + len1):
                idx2 = random.randint(0,len(nodes_in_cluster2)-1)
                links.append((j, nodes_in_cluster2[idx2]))
                linksr.append((nodes_in_cluster2[idx2], j))
        if mode == "interleaving":
            if i%2 == 0:
                for j in range(start_node, end_node):
                    idx2 = random.randint(0,len(nodes_in_cluster2)-1)
                    links.append((j, nodes_in_cluster2[idx2]))
                    linksr.append((nodes_in_cluster2[idx2], j))
            else:
                start_node -= interval
                if(i==len2-1):
                    end_node = len1//2
                else:
                    end_node -= interval
                for j in range(start_node, end_node):
                    idx2 = random.randint(0,len(nodes_in_cluster2)-1)
                    links.append((j+len1//2, nodes_in_cluster2[idx2]))
                    linksr.append((nodes_in_cluster2[idx2], j+len1//2))
        start_node += interval
        if(i == len2 - 2):
            end_node = len1
        else:
            end_node += interval

    return links, linksr


def linkingSingleLineToLineCluster(graphon1, graphon2):
    nodes_list1 = graphon1.nodes(data = True)
    nodes_list2 = graphon2.nodes(data = True)
    
    nodes_id_list1 = set()
    nodes_id_list2 = set()
    
    for key, _ in nodes_list1:
        nodes_id_list1.add(nodes_list1[key]['cluster'])
    for key, _ in nodes_list2:
        nodes_id_list2.add(nodes_list2[key]['cluster'])
        
    len1 = len(nodes_id_list1)
    len2 = len(nodes_id_list2)
    
    links = []
    linksr = []
    
    cluster_mapping_1 = get_cluster_mapping(graphon1)
    cluster_mapping_2 = get_cluster_mapping(graphon2)

    interval = len1 // len2
    start_node = 0
    end_node = start_node + interval
    for i in range(len2):
        nodes_in_cluster2 = cluster_mapping_2[i]
        # print(nodes_in_cluster2)
        for j in range(start_node, end_node):
            idx2 = random.randint(0,len(nodes_in_cluster2)-1)
            links.append((j, nodes_in_cluster2[idx2]))
            linksr.append((nodes_in_cluster2[idx2], j))
        start_node += interval
        if(i == len2 - 2):
            end_node = len1
        else:
            end_node += interval
            
    return links, linksr


def linkingRandomClusterToLineCluster(graphon1, graphon2, inner_cluster_link_p = 1):
    cluster_mapping_1 = get_cluster_mapping(graphon1)
    cluster_mapping_2 = get_cluster_mapping(graphon2)
    # print("mapping: {}".format(cluster_mapping_1))
    # print("mapping: {}".format(cluster_mapping_2))
    
    len1 = len(cluster_mapping_1.items())
    len2 = len(cluster_mapping_2.items())
    
    # print("len: {}---{}".format(len1, len2))
    
    rev = False
    if(len1 > len2):
        rev = True
        temp_mapping = cluster_mapping_1
        cluster_mapping_1 = cluster_mapping_2
        cluster_mapping_2 = temp_mapping
        temp = len1
        len1 = len2
        len2 = temp
    interval = len2 // len1
    
    links = []
    linksr = []
    
    for i in range(len1):
        nodes_in_cluster1 = cluster_mapping_1[i]
        for j in range(int(i*interval), int((i+1)*interval)):
            nodes_in_cluster2 = cluster_mapping_2[j]
            max_links = int(inner_cluster_link_p * min(len(nodes_in_cluster1), len(nodes_in_cluster2)))
            for _ in range(max_links):
                idx1 = random.randint(0, len(nodes_in_cluster1)-1)
                idx2 = random.randint(0, len(nodes_in_cluster2)-1)
                links.append((nodes_in_cluster1[idx1], nodes_in_cluster2[idx2]))
                linksr.append((nodes_in_cluster2[idx2], nodes_in_cluster1[idx1]))
    if rev:
        temp = links
        links = linksr
        linksr = temp
    
    # for i in range(len(graphon1.nodes())):
    #     links.append((i, i))
    #     linksr.append((i, i))
    
    # print("link length: {}".format(len(links)))
    # print("links: {}".format(links))

    return links, linksr
    
    

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

def graphgrowth(g, p, maxchildnum=3, maxdepth=6):
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

                    g.add_edge(p_node, c_node, weight=1.0)

                    nodeID += 1
        p_nodes = c_nodes
        c_nodes = []
    return g

def randomLinking(g, n=None):
    if not n:
        n = g.number_of_nodes()
    non_edges = list(nx.non_edges(g))
    if n > len(non_edges):
        n = len(non_edges)
    if n == 0:
        return g
    new_edges = random.sample(non_edges, n)
    g.add_edges_from(new_edges, weight=1)
    return g





if __name__ == "__main__":
    g = line(100)
    pos = nx.multipartite_layout(g, subset_key="layer")
    pos = nx.spring_layout(g, pos=pos)
    nx.draw(g, pos=pos, node_size=10)
    plt.savefig("./images/fig1.png")
    # plt.axis('equal')  
    plt.show()

    g = ring(120, 60, 6, 0.5)
    pos = nx.multipartite_layout(g, subset_key="layer")
    pos = nx.spring_layout(g, pos=pos)
    nx.draw(g, pos=pos, node_size=10)
    plt.savefig("./images/fig2.png")
    # plt.axis('equal')  
    plt.show()