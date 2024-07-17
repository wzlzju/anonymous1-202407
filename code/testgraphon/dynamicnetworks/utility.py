import os
import networkx as nx
import matplotlib.pyplot as plt
import hashlib
import json, pickle
from functools import reduce  
from collections import Counter  


def drawGraph(graph, method=None):
    if os.path.isfile(graph):
        if graph.endswith('.edgelist'):
            graph = nx.read_edgelist(graph, nodetype=int, data=[("weight", float)])
    else:
        graph = graph

    if not method:
        nx.draw_spring(graph)
        plt.show()

def layout(graphfile, method=None, loadpre=False, **kwargs):
    if not method:
        method = nx.spring_layout
    with open(graphfile, 'r') as f:
        filecontent = f.read()
    hash_value = sha256_hash(filecontent+method.__name__).upper()
    if loadpre and os.path.exists("./tmp/"+hash_value):
        with open("./tmp/"+hash_value, 'rb') as f:
            pos = pickle.load(f)
    else:
        if graphfile.endswith('.edgelist'):
            graph = nx.read_edgelist(graphfile, nodetype=int, data=[("weight", float)])
        else:
            print("unsupported graph file type")
        pos = method(graph, **kwargs)
        with open("./tmp/"+hash_value, 'wb') as f:
            pickle.dump(pos, f)
    
    return pos
    

def sha256_hash(s):
    hashObj = hashlib.sha256(s.encode())
    return hashObj.hexdigest()

class Color:  
    def __init__(self, red, green, blue):  
        self.rgb = (red, green, blue)  
  
    def __add__(self, other):  
        new_red = self.rgb[0] + other.rgb[0]  
        new_green = self.rgb[1] + other.rgb[1]  
        new_blue = self.rgb[2] + other.rgb[2]  
        return Color(new_red, new_green, new_blue)  
  
    def __radd__(self, other):  
        new_red = self.rgb[0] + other  
        new_green = self.rgb[1] + other  
        new_blue = self.rgb[2] + other  
        return Color(new_red, new_green, new_blue)  
  
    def __sub__(self, other):  
        new_red = self.rgb[0] - other.rgb[0]  
        new_green = self.rgb[1] - other.rgb[1]  
        new_blue = self.rgb[2] - other.rgb[2]  
        return Color(new_red, new_green, new_blue)  
  
    def __rsub__(self, other):  
        new_red = other - self.rgb[0]  
        new_green = other - self.rgb[1]  
        new_blue = other - self.rgb[2]  
        return Color(new_red, new_green, new_blue)  
  
    def __mul__(self, other):  
        new_red = self.rgb[0] * other  
        new_green = self.rgb[1] * other  
        new_blue = self.rgb[2] * other  
        return Color(new_red, new_green, new_blue)  
  
    def __rmul__(self, other):  
        new_red = self.rgb[0] * other  
        new_green = self.rgb[1] * other  
        new_blue = self.rgb[2] * other  
        return Color(new_red, new_green, new_blue)  
  
    def __truediv__(self, other):  
        new_red = self.rgb[0] / other
        new_green = self.rgb[1] / other
        new_blue = self.rgb[2] / other
        return Color(new_red, new_green, new_blue)  
  
    def __rtruediv__(self, other):  
        new_red = other / self.rgb[0]
        new_green = other / self.rgb[1]
        new_blue = other / self.rgb[2]
        return Color(new_red, new_green, new_blue)  
  
    def __repr__(self):  
        return f'Color{self.rgb}'  
    
    def reg(self):
        new_red = max(0, min(self.rgb[0], 1))  
        new_green = max(0, min(self.rgb[1], 1))  
        new_blue = max(0, min(self.rgb[2], 1))   
        return Color(new_red, new_green, new_blue) 

    def to_tuple(self):  
        return self.rgb 

class colorinter(object):
    def __init__(self, c0, c1, v0=0, v1=1) -> None:
        self.c0 = Color(*c0)
        self.c1 = Color(*c1)
        self.v0 = v0
        self.v1 = v1
    
    def color(self, v, v0=None, v1=None):
        if not v0:
            v0 = self.v0
        if not v1:
            v1 = self.v1

        cv = (v-v0)/(v1-v0)
        color = (self.c0+cv*(self.c1-self.c0)).reg()

        return color

    def colorMean(self, vs, v0=None, v1=None):
        if len(vs)>=2:
            vs
        colors = [self.color(v, v0, v1) for v in vs]
        color = reduce(lambda c1,c2:c1+c2, colors)
        color = color/len(colors)
        color = color.reg()

        return color


if __name__ == "__main__":
    # drawGraph("./estimations/test3.edgelist")
    pass