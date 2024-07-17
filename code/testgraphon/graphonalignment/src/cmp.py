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

import logging


def getCorrespondences(g1, g2, links, linksr):
    m1 = np.zeros((g1.number_of_nodes(), g2.number_of_nodes()))
    m2 = np.zeros((g2.number_of_nodes(), g1.number_of_nodes()))
    for i,j in links:
        m1[i][j] += 1
    for j,i in linksr:
        m2[j][i] += 1  
    for i in range(len(m1)):
        if m1[i].sum() != 0.:
            m1[i] = m1[i]/m1[i].sum()
    for i in range(len(m2)):
        if m2[i].sum() != 0.:
            m2[i] = m2[i]/m2[i].sum()
    # m1 = m1/m1.sum(axis=1)[:,np.newaxis]
    # m2 = m2/m2.sum(axis=1)[:,np.newaxis]

    return m1, m2

def hard_alignment(g1, g2, m1, m2, l):
    a1 = nx.to_numpy_array(g1)
    a2 = nx.to_numpy_array(g2)
    a1_ = np.matmul(np.matmul(m1.T,a1),m1)
    a2_ = np.matmul(np.matmul(m2.T,a2),m2)
    a1_new = l*a1 + (1-l)*a2_
    a2_new = l*a2 + (1-l)*a1_
    new_g1 = nx.from_numpy_array(a1_new)
    new_g2 = nx.from_numpy_array(a2_new)

    return new_g1, new_g2
