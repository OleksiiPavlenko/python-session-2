import numpy as np

def sharpen_3x3():
    return np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
