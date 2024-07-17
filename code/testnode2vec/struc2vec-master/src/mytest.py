
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import math

N = 100

x1=[1,1,0.6,0.6,0.4,0.2,0.2,0.2,0.1,0.08,0.05,0.05,0.05,0.04,0.02,0.01,0.01,0.01]
x2=[1,1,1,1,0.9,0.9,0.7,0.7,0.68,0.67,0.67,0.2,0.1,0.1]
X=[x1, x2]
y1=[0.2,0.1,0.02,0.01,0.003,0.001,0.0011,0.0002,0.0003,0.0001]
y2=[0.1,0.099,0.098,0.09,0.0,0.09,0.08,0.078,0.077,0.077,0.06,0.06,0.06,0.06,0.01,0.02,0.001]
Y=[y1,y2]


def align(x, y):

    def d(a,b):
        return max(a,b)/(min(a,b)+1e-6)-1

    x = sorted([sorted(xi, reverse=True)+[0.0]*(N-len(xi)) for xi in x], reverse=True)
    y = sorted([sorted(yi, reverse=True)+[0.0]*(N-len(yi)) for yi in y], reverse=True)

    print(fastdtw(x,y, dist=lambda xi,yi: fastdtw(xi,yi, dist=d)[0]))



if __name__ == "__main__":

    align(X,Y)