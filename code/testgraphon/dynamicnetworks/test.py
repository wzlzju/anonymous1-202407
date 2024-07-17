import os
import networkx as nx
from functools import reduce  
from collections import Counter  
import matplotlib.pyplot as plt
import json, pickle

from utility import *
from graphonestimation import *

higgspath = "../../networksdata/higgs/component1/dynamicgraphs/"
enronpath = "../../networksdata/Enron Dynamic Network/"
radoslawpath = "../../networksdata/radoslaw_email/slices/"
multilayertransportgraphpath = "../../networksdata/EUAir_Multiplex_Transport/Dataset/EUAirTransportation_multiplex.edges"
resultpath = "./estimations/"
dynamicnetworkspath = radoslawpath



if __name__ == "__main__":
    graphfile = "./estimations/radoslaw.edgelist"
    graph = nx.read_edgelist(graphfile, nodetype=int, data=[("weight", float)]) 
    print(graph.number_of_nodes())
    graph.remove_edges_from(nx.selfloop_edges(graph))
        
    pos = layout(graphfile, method=nx.spring_layout, loadpre=False, weight='weight')

    # get edge color
    framefiles = os.listdir(dynamicnetworkspath)
    framefiles.sort()
    # framefiles = [dynamicnetworkspath+file for file in framefiles if file.startswith("RE")]
    framefiles = [dynamicnetworkspath+file for file in framefiles]
    frames = [set(loadedgelist(framefile)) for framefile in framefiles]
    edge_color = [(0,0,0) for _ in range(graph.number_of_edges())]
    edge_width = [1 for _ in range(graph.number_of_edges())]
    edge_alpha = [1 for _ in range(graph.number_of_edges())]
    colornum = len(frames)
    colorinterObj = colorinter((1,0,0),(0,0,1))
    for i, (vi, vj, w) in enumerate(graph.edges.data("weight")):
        inslices = []
        for t in range(colornum):
            if "%d %d 1"%(vi,vj) in frames[t] or "%d %d 1"%(vj,vi) in frames[t]:
                inslices.append(t/colornum)
                # edge_color[i] = (*colorinterObj.color(t/colornum).to_tuple(), w)
                # break
        edge_color[i] = colorinterObj.colorMean(inslices).to_tuple()
        edge_width[i] = len(inslices)
        # edge_alpha[i] = len(inslices)/colornum if len(inslices)>=8 else 0.1
        edge_alpha[i] = max((len(inslices)/colornum)**8, 0.1)
    nx.draw(graph, pos=pos, node_size=1, alpha=edge_alpha, edge_color=edge_color)
    plt.show()