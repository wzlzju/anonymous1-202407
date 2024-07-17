import os, sys, time
import random
import json, pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import copy



# sys.path.append(os.path.abspath("./testgraphon/graphonalignment/src/"))
# from utility import *
sys.path.append(os.path.abspath("./"))
from testgraphon.graphonalignment.src.utility import *
from exp.syn import *
from Tfunctions import *
from args import *

# sys.path.append(os.path.abspath("./testsampling/pybluenoise/"))
# from blue_noise import blueNoise
from testsampling.pybluenoise.blue_noise import blueNoise

def test_tsne(embs0, cat):
    n_perplexity = [5, 15, 25, 35, 45]
    n_early_exaggeration = [1, 10, 100, 1000, 10000]
    fig, axes = plt.subplots(len(n_perplexity), len(n_early_exaggeration))
    for i,perplexity in enumerate(n_perplexity):
        for j,early_exaggeration in enumerate(n_early_exaggeration):
            data = tsne(embs0, perplexity=perplexity, early_exaggeration=early_exaggeration)
            ax = axes[i][j]
            scatter(data, cat, ax=ax)
            if j==0:
                ax.set_ylabel("perplexity: %d"%perplexity)
            if i==len(n_perplexity)-1:
                ax.set_xlabel("early_exaggeration: %d"%early_exaggeration)
            ax.axis("equal")
    plt.show()



def show(showplt=True, ret=False, sampling_r=None):
    rets = {}
    config = "ring-0.2"
    counter_config = "line-0.2"
    args = parse_args(configs, config)
    counter_args = parse_args(configs, counter_config)
    random.seed(args["random_state"])
    np.random.seed(args["random_state"])

    g = nx.read_edgelist(args["data_path"]+args["graph_path"], nodetype=int)
    if args.get("origin_path", None):
        old_g = nx.read_edgelist(args["data_path"]+args["origin_path"], nodetype=int)
    nodelist = list(g.nodes())
    embs_list = readembeddings(args["data_path"]+args["embs_path"], ret=list)
    if args.get("nodedata_path", None):
        with open(args["data_path"]+args["nodedata_path"], "rb") as f:
            nodes = pickle.load(f)
        g.add_nodes_from(nodes)

    embs_np = np.array(embs_list)
    data = (tsne(embs_np), umap(embs_np, random_state=args["random_state"]), pca(embs_np))
    embs = data[1]
    cat = {"dbscan": dbscan, "kmeans": kmeans}[args["clu_method"]](embs, **args["clu_args"])
    if args["highlight_nodes_method"] == "max_degree":
        centers = argmax_degree(g, args["highlight_nodes_paras"])
        center_indexes = [nodelist.index(center) for center in centers]
        cat = color_specified_nodes(cat, center_indexes, color=lambda x:-1)

    # pos = nx.spring_layout(g, pos=nx.circular_layout(g))
    fig, axes = plt.subplots(2,4)
    scatter(data[0], cat, ax=axes[0][0], alpha=args["alpha"], label="t-SNE")
    scatter(data[1], cat, ax=axes[0][1], alpha=args["alpha"], label="UMAP")
    scatter(data[2], cat, ax=axes[0][2], alpha=args["alpha"], label="PCA")
    pos = draw(g, draw_edges_of_g=old_g, layout=args["layout_method"], label="", node_color=cat, node_size=args["node_size"], edge_color=args["edge_color"], ax=axes[0][3], alpha=args["alpha"])
    axes[0][0].axis('equal')  
    axes[0][1].axis('equal')  
    axes[0][2].axis('equal')  
    axes[0][3].axis('equal')  

    # sampling
    points = [{
        "id": nodelist[i],
        "lat": float(emb[0]),
        "lng": float(emb[1])
    } for i,emb in enumerate(data[0])]
    points = blueNoise(points, r=args["sampling_r"] if not sampling_r else sampling_r)
    ids = [p['id'] for p in points]
    subg = g.subgraph(ids)
    # print(cal("1", g1=g, g2=subg))
    rets['1'] = cal("1", g1=g, g2=subg)
    # print(cal("2", g1=g, g2=subg))
    rets['2'] = cal("2", g1=g, g2=subg)
    cg = nx.read_edgelist(counter_args["data_path"]+counter_args["graph_path"], nodetype=int)
    with open(counter_args["data_path"]+counter_args["links_path"], "rb") as f:
        links = pickle.load(f)
        links = (links[1], links[0])
    og = nx.read_edgelist(args["data_path"]+args["origin_path"], nodetype=int)
    # print(cal("3", g1=og, g2=g, cg=cg, links=links))
    rets['3'] = cal("3", g1=og, g2=g, cg=cg, links=links)
    cnodelist = list(cg.nodes())
    if True:
        cembs_list = readembeddings(counter_args["data_path"]+counter_args["embs_path"], ret=list)
        if counter_args.get("nodedata_path", None):
            with open(counter_args["data_path"]+counter_args["nodedata_path"], "rb") as f:
                nodes = pickle.load(f)
            cg.add_nodes_from(nodes)

        cembs_np = np.array(cembs_list)
        cdata = (tsne(cembs_np), umap(cembs_np, random_state=counter_args["random_state"]), pca(cembs_np))
        cembs = cdata[1]
        cpoints = [{
            "id": cnodelist[i],
            "lat": float(cemb[0]),
            "lng": float(cemb[1])
        } for i,cemb in enumerate(cdata[0])]
        cpoints = blueNoise(cpoints, r=counter_args["sampling_r"])
        cids = [p['id'] for p in cpoints]
        csubg = cg.subgraph(cids)
    # print(cal("4", g1=subg, g2=csubg, links=links))
    rets['4'] = cal("4", g1=subg, g2=csubg, links=links)
    if args["new_layout"]:
        subg = max_connected_component(subg)
        sampling_rate =  subg.number_of_nodes()/len(embs)
        scatter(np.array([e for i,e in enumerate(data[0]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][0], alpha=args["alpha"], label="t-SNE")
        scatter(np.array([e for i,e in enumerate(data[1]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][1], alpha=args["alpha"], label="UMAP")
        scatter(np.array([e for i,e in enumerate(data[2]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][2], alpha=args["alpha"], label="PCA")
        node_color = [cat[nodelist.index(nid)] for nid in subg.nodes()] #[e for i,e in enumerate(cat) if nodelist[i] in ids]
        draw(g, node_color=node_color, node_size=args["node_size"], edge_color=args["edge_color"], ax=axes[1][3], alpha=args["alpha"], label="sampling rate %f"%sampling_rate)
    else:
        sampling_rate = len(ids)/len(embs)
        scatter(np.array([e for i,e in enumerate(data[0]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][0], alpha=args["alpha"], label="t-SNE")
        scatter(np.array([e for i,e in enumerate(data[1]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][1], alpha=args["alpha"], label="UMAP")
        scatter(np.array([e for i,e in enumerate(data[2]) if nodelist[i] in ids]), [e for i,e in enumerate(cat) if nodelist[i] in ids], ax=axes[1][2], alpha=args["alpha"], label="PCA")
        node_color = [cat[nodelist.index(nid)] for nid in ids] #[e for i,e in enumerate(cat) if nodelist[i] in ids]
        if args["layout_arrangement"]:
            g = {
                "from_position": graph_summary_from_position,
                "from_topology": graph_summary_from_topology,
                "from_groups": graph_summary_from_groups,
            }.get(args["layout_arrangement"], graph_summary_from_position)(old_g, ids, pos=pos, rate=0.2)
        pos = {nid: pos[nid] for nid in ids}
        # edgelist = [e for e in g.edges() if e[0] in ids and e[1] in ids]
        # draw(g, pos=pos, nodelist=ids, edgelist=edgelist, node_color=node_color, node_size=args["node_size"], edge_color=args["edge_color"], ax=axes[1][3], alpha=args["alpha"])
        draw(g, pos=pos, node_color=node_color, node_size=args["node_size"], edge_color=args["edge_color"], ax=axes[1][3], alpha=args["alpha"], with_labels=False, label="sampling rate %f"%sampling_rate)
    axes[1][0].axis('equal')  
    axes[1][1].axis('equal')  
    axes[1][2].axis('equal')  
    axes[1][3].axis('equal')  

    plt.axis('equal')
    if showplt:
        plt.show()
    if ret:
        return rets


def spline(x, y):
    xnew = np.linspace(min(x), max(x), 300) 
    spl = make_interp_spline(x, y, k=3)
    ynew = spl(xnew)
    return xnew, ynew

if __name__ == "__main__":
    seed = 8
    random.seed(seed)
    np.random.seed(seed)
    show()
    res = show(showplt=False, ret=True, sampling_r=0.0015)
    res['1'] = show(showplt=False, ret=True, sampling_r=0.0001)['1']
    with open("res0.pickle", "wb") as f:
        pickle.dump(res, f)
    # plots(plottype="box")
