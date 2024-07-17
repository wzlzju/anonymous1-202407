import numpy as np
import networkx as nx
import random, math
import matplotlib
import matplotlib.pyplot as plt
import os, sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import jensenshannon, euclidean


def syn(seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed)

def cal(mode, **kwargs):
    if mode == "1":
        s1 = 0
        s2 = 0
        g1 = kwargs['g1']
        g2 = kwargs['g2']
        s1 = sum([1 if g1.edges[e]['weight']>=0.99 else 0 for e in g1.edges()])
        s2 = sum([1 if g2.edges[e]['weight']>=0.99 else 0 for e in g2.edges()])
        return s2/s1
    elif mode == "2":
        g1 = kwargs['g1']
        g2 = kwargs['g2']
        pos1 = np.array([i for i in nx.spring_layout(g1).values()])
        pos2 = np.array([i for i in nx.spring_layout(g2).values()])
        kmean1 = KMeans(best_clustering(pos1)[0], n_init='auto')
        kmean1.fit(pos1)
        kmean2 = KMeans(best_clustering(pos2)[0], n_init='auto')
        kmean2.fit(pos1)
        hist1 = build_histogram(pos1, kmean1)
        hist2 = build_histogram(pos2, kmean2)
        hist1, hist2 = splitHist(hist1, hist2)
        return jensenshannon(hist1, hist2)
    elif mode == "3":
        g1 = kwargs['g1']
        g2 = kwargs['g2']
        pos1 = nx.spring_layout(g1)
        pos2 = nx.spring_layout(g2)
        cg = kwargs['cg']
        links = kwargs['links']
        links, linksr = links[0], links[1]
        cpos = np.array([i for i in nx.spring_layout(cg).values()])
        ccat = best_clustering(cpos)[1]
        dists1 = []
        dists2 = []
        for cat in set(ccat):
            cpoints = []
            for link in links:
                if ccat[link[1]] == cat:
                    cpoints.append(link[0])
            dists1.append(ave_distance(pos1, cpoints))
            # cpoints = list(set(cpoints)&set(list(g2.nodes())))
            dists2.append(ave_distance(pos2, cpoints))
        # print(dists1)
        # print(dists2)
        return sum(dists2)/sum(dists1)
    if mode == "4":
        g1 = kwargs['g1']
        g2 = kwargs['g2']
        links = kwargs['links']
        links, linksr = links[0], links[1]
        associated_nodes = []
        for node in g1.nodes():
            for link in links:
                if link[0] == node:
                    associated_nodes.append(link[1])
        shot_nodes = []
        for node in g2.nodes():
            if node in associated_nodes:
                shot_nodes.append(node)
        # print(associated_nodes)
        # print(shot_nodes)
        return len(shot_nodes)/len(associated_nodes)

def ave_distance(pos, pids):
    ds = []
    for i in range(len(pids)-1):
        for j in range(i+1, len(pids)):
            ds.append(euclidean(pos[pids[i]], pos[pids[j]]))
    return mean(ds)

def mean(l):
    return sum(l)/len(l)

def best_clustering(d):
    scores = []
    cats = []
    for i in range(2,6):
        cluster_op = KMeans(n_clusters=i, n_init='auto')
        cat = cluster_op.fit_predict(d)
        cats.append(cat)
        scores.append(silhouette_score(d, cat))
    kmax = np.argmax(scores)+2
    return kmax, cats[kmax-2], scores[kmax-2]

def build_histogram(d, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    cluster_result =  cluster_alg.predict(d)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram

def splitHist(h1, h2):
    l1, l2 = len(h1), len(h2)
    if l1==l2:
        return h1, h2
    s1 = lcm(l1, l2)//l1
    s2 = lcm(l1, l2)//l2
    rh1 = []
    for i in h1:
        for j in range(s1):
            rh1.append(i/s1)
    rh2 = []
    for i in h2:
        for j in range(s2):
            rh2.append(i/s2)
    return rh1, rh2

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)


if __name__ == "__main__":
    g1 = nx.circular_ladder_graph(50)
    g2 = nx.circulant_graph(100, [0])
    print(cal("2", g1=g1, g2=g2))
    