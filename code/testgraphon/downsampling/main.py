import networkx as nx
import numpy as np
from numpy.lib.stride_tricks import as_strided


def g2m(g):
    return nx.to_numpy_array(g)

def pool2d(A, kernel_size, stride, padding=0, pool_mode='avg'):
    A = np.pad(A, padding, mode='constant')
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

def downsampling(m, kernel_size):
    m = pool2d(m, kernel_size, kernel_size)
    return m

def sampling(m, n):
    h, w = m.shape
    