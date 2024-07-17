import os
import networkx as nx
from functools import reduce  
from collections import Counter  
import matplotlib.pyplot as plt
import json, pickle

from utility import *

higgspath = "../../networksdata/higgs/component1/dynamicgraphs/"
enronpath = "../../networksdata/Enron Dynamic Network/"
radoslawpath = "../../networksdata/radoslaw_email/slices/"
multilayertransportgraphpath = "../../networksdata/EUAir_Multiplex_Transport/Dataset/EUAirTransportation_multiplex.edges"
resultpath = "./estimations/"
dynamicnetworkspath = radoslawpath

def loadedgelist(edgelistpath):
    with open(edgelistpath, 'r') as f:
        s = f.read()
    while s.endswith('\n'):
        s = s[:-1]
    return s.split('\n')

def weightaverage(edgelists, all=None):
    count = Counter(reduce(lambda x,y:x+y, edgelists))
    if not all:
        all = len(edgelists)
    if all=="max":
        all = max(count.values())
    if callable(all):
        all = all(edgelists)
    graphon = [(k.split(' ')[0], k.split(' ')[1], count[k]/all) for k in count.keys()]
    return graphon



if __name__ == "__main__":
    # Enron
    framefiles = os.listdir(dynamicnetworkspath)
    framefiles.sort()
    # framefiles = [dynamicnetworkspath+file for file in framefiles if file.startswith("RE")]
    framefiles = [dynamicnetworkspath+file for file in framefiles]
    frames = [loadedgelist(framefile) for framefile in framefiles]
    frames = frames[:]
    graphon = weightaverage(frames)
    with open(resultpath+"radoslaw.edgelist", 'w') as f:
        f.write('\n'.join(["%s %s %f"%edge for edge in graphon])+'\n')
    
    

    # with open(multilayertransportgraphpath, 'r') as f:
    #     frames = [[' '.join(line.split(' ')[1:]) for line in f]]
    # graphon = weightaverage(frames, "max")
    # with open(resultpath+"air_transport.edgelist", 'w') as f:
    #     f.write('\n'.join(["%s %s %f"%edge for edge in graphon])+'\n')

    # for file in os.listdir(dynamicnetworkspath):
    #     l = loadedgelist(dynamicnetworkspath+file)
    #     for i,e in enumerate(l):
    #         if e=='':
    #             print(file, i)