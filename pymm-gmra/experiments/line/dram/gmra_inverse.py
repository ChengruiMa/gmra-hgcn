from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import time
import pickle as pk
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import math
from collections import Counter
# from tensorflow.keras.datasets import mnist

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree
from pysrc.utils.utils import *
from pysrc.utils.inverse import *

# Simple Line Shape
def line():
    # Load points from 'line.txt'
    return np.loadtxt('../line.txt', delimiter=',')

def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    args = parser.parse_args()

    print("loading data")
    start_time = time.time()
    X = line() # Using 1,000 pts here imported from the line.txt file
    X = pt.from_numpy(X.astype(np.float32))
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    if not os.path.exists(args.covertree_path):
        raise ValueError("ERROR: covertree json file does not exist at [%s]"
                         % args.covertree_path)


    print("loading covertree from [%s]" % args.covertree_path)
    start_time = time.time()
    cover_tree: CoverTree = CoverTree(args.covertree_path)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    print("constructing dyadic tree")
    start_time = time.time()
    dyadic_tree = DyadicTree(cover_tree)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    print("constructing wavelet tree")
    start_time = time.time()
    print("X shape:", X.shape)
    wavelet_tree = WaveletTree(dyadic_tree, X, 0, X.shape[-1])
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))
    print("took script {0:.4f} seconds to run".format(end_time-init_time))

    projections = invert(wavelet_tree, X)
    np.save('./line/results/mnist_inverse.npy', projections)

if __name__ == "__main__":
    main()