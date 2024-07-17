import os, sys
import time, datetime
import json, pickle
import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



def linkingLists_oneall2manyall(l1, l2):
    if len(l1) < len(l2):
        g0 = l1
        g1 = l2
        rev = False
    else:
        g1 = l1
        g0 = l2
        rev = True

    rate = len(g1)/len(g0)
    links = []
    linksr = []
    j = 0
    for i,vi in enumerate(g0):
        if i == len(g0)-1:
            nj = len(g1)
        else:
            nj = int((i+1)*rate)
        for cj in range(j, nj):
            links.append((vi, g1[cj]) if not rev else (g1[cj], vi))
            linksr.append((vi, g1[cj]) if rev else (g1[cj], vi))
        j = nj

    return links, linksr

def addInterlayerLink_oneall2manyall(graphon1, graphon2):
    if graphon1.number_of_nodes() < graphon2.number_of_nodes():
        g0 = graphon1
        g1 = graphon2
        rev = False
    else:
        g1 = graphon1
        g0 = graphon2
        rev = True
    d0 = list(nx.degree(g0))
    d1 = list(nx.degree(g1))
    d0.sort(reverse=True, key=lambda x:x[1])
    d1.sort(reverse=True, key=lambda x:x[1])
    rate = g1.number_of_nodes()/g0.number_of_nodes()
    links = []
    linksr = []
    j = 0
    for i,(vi,_) in enumerate(d0):
        if i == len(d0)-1:
            nj = len(d1)
        elif i in [0,1]:
            nj = j+1 
        else:
            nj = int((i+1)*rate)
        for cj in range(j, nj):
            links.append((vi, d1[cj][0]) if not rev else (d1[cj][0], vi))
            linksr.append((vi, d1[cj][0]) if rev else (d1[cj][0], vi))
        j = nj

    return links, linksr

def addInterlayerLink_oneall2one(graphon1, graphon2):
    if graphon1.number_of_nodes() < graphon2.number_of_nodes():
        g0 = graphon1
        g1 = graphon2
        rev = False
    else:
        g1 = graphon1
        g0 = graphon2
        rev = True
    d0 = list(nx.degree(g0))
    d1 = list(nx.degree(g1))
    d0.sort(reverse=True, key=lambda x:x[1])
    d1.sort(reverse=True, key=lambda x:x[1])
    rate = g1.number_of_nodes()/g0.number_of_nodes()
    links = []
    linksr = []
    j = 0
    for i,(vi,_) in enumerate(d0):
        if i == len(d0)-1:
            nj = len(d1)
        elif i in [0,1]:
            nj = j+1 
        else:
            nj = int((i+1)*rate)
        cj = random.choice(list(range(j, nj)))
        links.append((vi, d1[cj][0]) if not rev else (d1[cj][0], vi))
        linksr.append((vi, d1[cj][0]) if rev else (d1[cj][0], vi))
        j = nj

    return links, linksr

def addInterlayerLink_tmp1(graphon1, graphon2):
    if graphon1.number_of_nodes() < graphon2.number_of_nodes():
        g0 = graphon1
        g1 = graphon2
        rev = False
    else:
        g1 = graphon1
        g0 = graphon2
        rev = True
    d0 = list(nx.degree(g0))
    d1 = list(nx.degree(g1))
    d0.sort(reverse=True, key=lambda x:x[1])
    d1.sort(reverse=True, key=lambda x:x[1])
    rate = g1.number_of_nodes()/g0.number_of_nodes()
    links = []
    linksr = []
    for i,(vi,_) in enumerate(d0):
        links.append((vi, d1[i][0]) if not rev else (d1[i][0], vi))
        links.append((vi, d1[-i][0]) if not rev else (d1[-i][0], vi))
        linksr.append((vi, d1[i][0]) if rev else (d1[i][0], vi))
        linksr.append((vi, d1[-i][0]) if rev else (d1[-i][0], vi))

    return links, linksr

def addInterlayerLinkbasedonDegreeRanking(g1, g2, firstN=10):
    d1 = list(nx.degree(g1))
    d2 = list(nx.degree(g2))
    d1.sort(reverse=True, key=lambda x:x[1])
    d2.sort(reverse=True, key=lambda x:x[1])
    links = []
    linksr = []
    for i in range(firstN):
        links.append((d1[i][0], d2[i][0]))
        linksr.append((d2[i][0], d1[i][0]))
    return links, linksr

def addRackandFault(g1, g2, rackNum=None):
    if not rackNum:
        rackNum = (g1.number_of_nodes()+g2.number_of_nodes())//6
    racks = [i for i in range(rackNum)]
    rack2fault = {rack:random.random()*0.5 for rack in racks}
    rack2fault[racks[0]] += 0.5

    rack2node1 = {rack:[] for rack in racks}
    node12rack = {}
    for node in g1.nodes():
        crack = random.choice(racks)
        g1.nodes[node]['rack'] = crack
        g1.nodes[node]['fault'] = rack2fault[crack]
        rack2node1[crack].append(node)
        node12rack[node] = crack
    rack2node2 = {rack:[] for rack in racks}
    node22rack = {}
    for node in g2.nodes():
        crack = random.choice(racks)
        g2.nodes[node]['rack'] = crack
        g2.nodes[node]['fault'] = rack2fault[crack]
        rack2node2[crack].append(node)
        node22rack[node] = crack
    
    links, linksr = [], []
    for node1 in g1.nodes():
        for node2 in rack2node2[node12rack[node1]]:
            links.append((node1, node2))
    for node2 in g2.nodes():
        for node1 in rack2node1[node22rack[node2]]:
            linksr.append((node2, node1))
    
    return g1, g2, links, linksr