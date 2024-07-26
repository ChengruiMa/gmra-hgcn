# Note: Plot embedding in matplotlab
# SYSTEM IMPORTS
from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import time

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_
print(sys.path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree

original_sys_path = sys.path.copy()

# Add the parent directory of 'hgcn' to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from hgcn
from hgcn.manifolds.poincare import PoincareBall

# Restore the original sys.path
sys.path = original_sys_path


# Simple Helix Shape
def helix():
    # Load points from 'helix.txt'
    return np.loadtxt('../helix.txt', delimiter=',')

# The low-dimensional features for each point at an arbitrary scale (i.e. 0) are stored inside the wavelet nodes themselves. 
# When you want to extract the features for points in the dataset at a known scale, you need to traverse the tree to the 
# depth for that scale (i.e. 0) and then inspect all the nodes at that scale to get the features.

# Note: When you're at the correct depth, a single point only will be contained within
# one of the nodes at that level, you need to traverse down to that depth and then go through each node one at a time. 
# If you want to get more than a single point you just have to aggregate across nodes. 
# Be warned: the dimensionality at one node at depth d might be different than another node also at depth d!
def get_nodes_at_depth(node, depth):
    #Return a list of all nodes at depth in tree
    #subroutine to best_depth, start by passing in root node and depth you want
    #TODO: what happens when depth > max depth of node (return [])
    if depth == 0:
        return [node]
    result = []
    for child in node.children:
        result+=get_nodes_at_depth(child,depth-1)
    return result

def best_depth(node):
    #Find the list of WaveletNodes that exist at the deepest level where all nodes have the same dimension
    #TODO: What happens if node.basis is empty, does this work still?
    depth_counter = 1
    #root will satisfy best depth parameters since all nodes are present in root
    best_nodes = [node]
    best_dim = node.basis.shape[1]
    while True:
        nodes = get_nodes_at_depth(node, depth_counter)

        #check if this set is "good" - all nodes have the same dimension
        dims = {x.basis.shape[1] for x in nodes}

        num_dims = len(dims)

        dim = dims.pop()

        #if num_dims is 1, then all nodes have the same dimension (good). need to make sure its not 0
        if num_dims == 1 and not dim == 0:
            best_nodes = nodes
            best_dim = dim
        else:
            #if this is a bad depth, the previous depth was BEST. return those nodes
            return best_nodes, best_dim
        depth_counter += 1

def get_hyperbolic_embeddings(tree, manifold, c):
    nodes, dim = best_depth(tree.root)

    basis = pt.vstack([pt.tensor(node.basis) for node in nodes])
    idxs = pt.hstack([pt.tensor(node.idxs) for node in nodes])
    sigmas = pt.hstack([pt.tensor(node.sigmas[:-1]) for node in nodes])

    # Project embeddings to the Poincaré ball
    euclidean_embeddings = pt.multiply(basis, sigmas.reshape((basis.shape[0], 1)))
    hyperbolic_embeddings = manifold.expmap0(euclidean_embeddings, c)

    reordered_embs = pt.zeros_like(hyperbolic_embeddings)
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        reordered_embs[new_idx] = hyperbolic_embeddings[idx]
    
    return reordered_embs

def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    args = parser.parse_args()

    print("loading data")
    start_time = time.time()
    # Generate the helix dataset using the helix function
    X = helix() # Using 10,000 pts here imported from the helix.txt file
    X = pt.from_numpy(X.astype(np.float32))
    # Print the 3D points
    print("First 15 3D Points:")
    print(X[:15, :])
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
    wavelet_tree = WaveletTree(dyadic_tree, X, 0, X.shape[-1])
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))
    print("took script {0:.4f} seconds to run".format(end_time-init_time))

    # Set up the Poincaré ball manifold
    manifold = PoincareBall()
    c = 1.0  # curvature of the hyperbolic space

    # Set the desired scale to 0 (the roughest scale)
    desired_scale = 0

    print("Extracting hyperbolic embeddings at scale {0}".format(desired_scale))
    start_time = time.time()
    embeddings = get_hyperbolic_embeddings(wavelet_tree, manifold, c)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time - start_time))

    # Output the embeddings to a text file
    output_dir = "./helix/results"
    output_path = os.path.join(output_dir, "hyperbolic_dram.txt")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in embeddings:
            file.write(" ".join(map(str, embedding.tolist())) + "\n")

    print("Hyperbolic embeddings saved to {0}".format(output_path))

if __name__ == "__main__":
    main()
