import os, sys
import json, pickle
import random, math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb

from PIL import ImageColor

INF=1e6

def sigmoid(x, k=1):
    return 1/(1+math.exp(-x/k))

def sigmoid01(x, k=3):
    if x>=1:
        return 1
    if x<=0:
        return 0
    return 1/(1+((1-x)/x)**k)

def is_iterable(obj):  
    try:  
        iter(obj)  
        return True  
    except TypeError:  
        return False  

def dict2np(d):
    length = len(d)
    ret = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            ret[i][j] = d[i].get(j)
    return ret

def dictintersec(d1, d2):
    for k in d2.keys():
        d1[k] = d2[k]
    return d1

def reciprocal(n):
    return 1/n if n!=0 else INF

def max_connected_component(g):
    return g.subgraph(max(nx.connected_components(g), key=len)).copy()

def argmax_degree(g, degree_ranks=0):
    degrees = sorted(list(nx.degree(g)), key=lambda x:x[1], reverse=True)
    if is_iterable(degree_ranks):
        return [degrees[rank][0] for rank in degree_ranks]
    else:
        return degrees[degree_ranks][0]

def color_specified_nodes(colors, node_indexes, color=None):
    if callable(color):
        color = color(colors)
    
    if is_iterable(node_indexes):
        for node_index in node_indexes:
            colors[node_index] = color
    else:
        colors[node_indexes] = color
    
    return colors

def loadjson(url):
    d = json.load(open(url, "r"))
    return nx.node_link_graph(d)

def dumpjson(g, url):
    d = nx.json_graph.node_link_data(g)  
    json.dump(d, open(url, "w"))

def to_jsapp(g):
    dumpjson(g, url="force/force.json")

def graph_completion(g, ret=None, diag=None):
    g = g.copy()
    if ret == "adj":
        a = nx.to_numpy_array(g, weight="weight")
        if diag:
            a[np.diag_indices_from(a)] = diag
        return a
    else:
        edges = set(g.edges())
        for vi, vj in [(x, y) for x in g.nodes() for y in g.nodes()]:
            if vj > vi and (vi, vj) not in edges and (vj, vi) not in edges:
                g.add_edge(vi, vj, weight=0)
            if vj == vi and diag:
                g.add_edge(vi, vj, weight=diag)
        return g

def graph_cleaning(g):
    g.remove_edges_from(nx.selfloop_edges(g))
    remove_list = []
    for edge in g.edges():
        if g.edges[edge]['weight'] < 1e-6:
            remove_list.append(edge)
    g.remove_edges_from(remove_list)
    return g

def graph_sorting(g):
    h = nx.Graph()
    h.add_nodes_from(sorted(g.nodes(data=True)))
    h.add_edges_from(g.edges(data=True))
    return h

def graph_relabelling(g, labels):
    h = nx.Graph()
    nodes = list(g.nodes(data=True))
    h.add_nodes_from([nodes[label] for label in labels])
    h.add_edges_from(g.edges(data=True))
    return h

def group_points_by_nearest_centers(points, centers):
    distances = np.sqrt(((points[:, np.newaxis] - centers)**2).sum(axis=2))  
    closest_center_indices = np.argmin(distances, axis=1)  
    grouped_points = {i: np.where(closest_center_indices == i)[0] for i in range(len(centers))}  
    return grouped_points, closest_center_indices

def graph_summary_from_topology(g, ids, **kwargs):
    nodelist = list(g.nodes())
    dist_matrix = np.array([list(nx.single_source_shortest_path_length(g, i).values()) for i in ids]).T
    belong_to_indexes = np.argmin(dist_matrix, axis=1)
    groups = [{
                'id': nid,
                'pointsInDisk': [{'id': iid} for i,iid in enumerate(nodelist) if iid!=nid and belong_to_indexes[i]==ids.index(nid)]
            } for nid in ids]
    return graph_summary_from_groups(g, groups, 1e-6)

def graph_summary_from_position(g, ids, layout=None, pos=None, rate=1e-6, **kwargs):
    if not layout:
        layout = "default" 
        if pos:
            layout = "given_pos"
    if layout == "cycle":
        pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    elif layout == "multilayer":
        pos = nx.spring_layout(g, pos=nx.multipartite_layout(g, subset_key="layer"))
    elif layout == "random":
        pos = nx.spring_layout(g)
    elif layout == "given_init":
        pos = nx.spring_layout(g, pos=pos)
    elif layout == "given_pos":
        pos = pos
    elif layout == "default":
        pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    else:
        print("Unsupported layout:", layout)
        pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    
    points = np.array([pos[nid] for nid in pos])
    centers = np.array([pos[nid] for nid in ids])
    groups, _nearest_centers_indexes = group_points_by_nearest_centers(points, centers)
    # return [ids[i] for i in _nearest_centers_indexes]
    nodelist = list(g.nodes())
    groups = [{
                'id': nid,
                'pointsInDisk': [{'id': nodelist[i]} for i in groups[ids.index(nid)]]
            } for nid in ids]
    return graph_summary_from_groups(g, groups,rate)


def graph_summary_from_groups(g, groups, rate=0.5, **kwargs):
    new_g = nx.Graph()
    nodes = [p['id'] for p in groups]
    new_g.add_nodes_from(nodes)
    old_g_nodelist = list(g.nodes())
    points_indexes_sets = [[old_g_nodelist.index(pp['id']) for pp in p['pointsInDisk']] for p in groups]
    for i in range(len(nodes)-1):
        disk_i = set(points_indexes_sets[i])
        disk_i.add(old_g_nodelist.index(nodes[i]))
        for j in range(i+1, len(nodes)):
            disk_j = set(points_indexes_sets[j])
            disk_j.add(old_g_nodelist.index(nodes[j]))
            # if nodes[i] in [100,118] and nodes[j] in [100,118]:
                # new_nodes_indexes = list(disk_i|disk_j)
                # new_nodes = [old_g_nodelist[index] for index in new_nodes_indexes]
                # subg = g.subgraph(new_nodes)
                # pos = {id:pos[id] for id in new_nodes}
                # nx.draw(subg, pos=pos, with_labels=True)
                # plt.axis('equal')  
                # plt.show()
            if g.has_edge(nodes[i], nodes[j]) or \
                len(disk_i&disk_j) > 0 or \
                nx.to_numpy_array(g)[list(disk_i),:][:,list(disk_j)].sum()/(len(disk_j)*len(disk_j)) > rate:
                new_g.add_edge(nodes[i], nodes[j], weight=1.0)
    return new_g

def shortest_path(g, ret=None, diag=None):
    g = g.copy()
    if ret == "adj":
        d = dict(nx.shortest_path_length(g, weight=lambda vi,vj,attr: reciprocal(attr['weight'])))
        a = dict2np(d)
        if diag:
            a[np.diag_indices_from(a)] = diag
        return a
    else:
        edges = set(g.edges())
        for vi, vj in [(x, y) for x in g.nodes() for y in g.nodes()]:
            if vj > vi and (vi, vj) not in edges:
                g.add_edge(vi, vj, weight=nx.shortest_path_length(g, vi, vj, weight=lambda vi,vj,attr: reciprocal(attr['weight'])))
            if vj == vi and diag:
                g.add_edge(vi, vj, weight=diag)
        return g

def hsv2rgb(h, s, v):
    h_i = int(h * 6)  
    f = h * 6 - h_i  
    p = v * (1 - s)  
    q = v * (1 - f * s)  
    t = v * (1 - (1 - f) * s)  
    if h_i == 0:  
        r, g, b = v, t, p  
    elif h_i == 1:  
        r, g, b = q, v, p  
    elif h_i == 2:  
        r, g, b = p, v, t  
    elif h_i == 3:  
        r, g, b = p, q, v  
    elif h_i == 4:  
        r, g, b = t, p, v  
    else:  
        r, g, b = v, p, q  
    return r, g, b  

def link_reverse(links):
    r_links = [(vj,vi) for vi,vj in links]
    return r_links

def readembeddings(file, ret=dict):
    if file.endswith('.emb'):
        res = {}
        with open(file, 'r') as f:
            header = None
            for line in f:
                line = line.strip().split()
                if not header and len(line) <= 2:
                    header = line.copy()
                    N, M = map(int, header)
                else:
                    id = int(line[0])
                    emb = list(map(float, line[1:]))
                    res[id] = emb
        ret = ret()
        if type(ret) is list:
            res = [res[id] for id in sorted(list(res.keys()))]
    elif file.endswith('.json'):
        with open(file, 'r') as f:
            res = json.load(f)
        if type(ret) is dict and type(res) is list:
            res = {i:res[i] for i in range(len(res))}
        elif type(ret) is list and type(res) is dict:
            res = [res[id] for id in sorted(list(res.keys()))]
    elif file.endswith('.pickle'):
        with open(file, 'r') as f:
            res = pickle.load(f)
        if type(ret) is dict and type(res) is list:
            res = {i:res[i] for i in range(len(res))}
        elif type(ret) is list and type(res) is dict:
            res = [res[id] for id in sorted(list(res.keys()))]
    return res

def scatter(data, color=None, ax=None, label=None, **kwargs):
    if ax is not None:
        ax.scatter(data[:,0], data[:,1], c=color, **kwargs)
        if label:
            ax.set_title(label)
    else:
        plt.scatter(data[:,0], data[:,1], c=color, **kwargs)
        if label:
            plt.set_title(label)
        plt.show()

def draw(g, layout=None, pos=None, draw_edges_of_g=None, ax=None, label=None, 
        f_node_color=None, f_edge_color=None, f_node_size=None, f_edge_width=None,
        node_color=None, edge_color=None, node_size=None, edge_width=None, alpha=None,
        min_S=0, max_S=1, min_V=0, max_V=1, color_H=0.736, color_S=1.0, color_V=1.0, channel='S', edge_color_attr="weight",
        cmap=None,
        **kwargs):
    if not layout:
        layout = "default" 
        if pos:
            layout = "given_pos"
    if layout == "cycle":
        pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    elif layout == "multilayer":
        pos = nx.spring_layout(g, pos=nx.multipartite_layout(g, subset_key="layer"))
    elif layout == "random":
        pos = nx.spring_layout(g)
    elif layout == "given_init":
        pos = nx.spring_layout(g, pos=pos)
    elif layout == "given_pos":
        pos = pos
    elif layout == "default":
        pos = nx.spring_layout(g, pos=nx.circular_layout(g), iterations=50)
    else:
        print("Unsupported layout:", layout)
        pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    
    if not draw_edges_of_g:
        draw_edges_of_g = g
    if label and ax:
        ax.set_title(label)
    # min_S = 0.2
    # min_S = 0
    kwargs["node_size"] = 10
    kwargs["alpha"] = 0.5 if alpha is None else alpha
    if channel == 'S':
        kwargs["edge_color"] = [hsv2rgb(color_H, d[2][edge_color_attr]*(max_S-min_S)+min_S, color_V) for d in draw_edges_of_g.edges.data()]
    elif channel == 'V':
        kwargs["edge_color"] = [hsv_to_rgb(color_H, color_S, d[2][edge_color_attr]*(max_V-min_V)+min_V) for d in draw_edges_of_g.edges.data()]
    if f_node_color:
        kwargs["node_color"] = [f_node_color(d) for d in draw_edges_of_g.nodes.data()]
    if f_edge_color:
        kwargs["edge_color"] = [f_edge_color(d) for d in draw_edges_of_g.edges.data()]
    if f_node_size:
        kwargs["node_size"] = [f_node_size(d) for d in draw_edges_of_g.edges.data()]
    if f_edge_width:
        kwargs["width"] = [f_edge_width(d) for d in draw_edges_of_g.edges.data()]
    if node_color is not None:
        kwargs["node_color"] = node_color #if is_iterable() else [node_color for _ in range(g.number_of_nodes())]
    if edge_color is not None:
        kwargs["edge_color"] = edge_color #if is_iterable() else [edge_color for _ in range(g.number_of_edges())]
    if node_size is not None:
        kwargs["node_size"] = node_size #if is_iterable() else [node_size for _ in range(g.number_of_nodes())]
    if edge_width is not None:
        kwargs["width"] = edge_width #if is_iterable() else [edge_width for _ in range(g.number_of_edges())]
    nx.draw(draw_edges_of_g, pos=pos, ax=ax, cmap=cmap, **kwargs)
    if not ax:
        plt.axis('equal')  
        plt.show()
    
    return pos

if __name__ == "__main__":
    g1 = nx.random_internet_as_graph(1000)
    nx.set_edge_attributes(g1, 1, 'weight')
    draw(g1, layout="default", 
        f_node_color=lambda d: {'T':(0.125,0.29,0.692),'M':(0.278,0.735,1),'C':(0.624,1,0.862),'CP':(0.8,1,1)}[d[1]['type']], 
        f_edge_color=lambda d: hsv2rgb({"peer":0.736,"transit":0}[d[2]['type']], d[2]['weight'], 1.0))