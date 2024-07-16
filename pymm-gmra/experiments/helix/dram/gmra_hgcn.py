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

hgcn_path = "/scratch/f0071gk/mcas-gmra/hgcn"
if hgcn_path not in sys.path:
    sys.path.append(hgcn_path)

import numpy as np

def load_hyperbolic_embeddings(dataset='cora', model='HGCN', task='lp'):
    logs_dir = os.path.join(hgcn_path, "logs", task)
    runs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    if not runs:
        raise FileNotFoundError(f"No training runs found for task {task}")
    
    latest_run = max(runs)  # Get the most recent run
    run_dir = os.path.join(logs_dir, latest_run)
    
    # Find the subdirectory (assuming it's always "0" in this case)
    sub_dirs = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if not sub_dirs:
        raise FileNotFoundError(f"No subdirectories found in {run_dir}")
    
    sub_dir = sub_dirs[0]  # Assume it's always "0"
    
    embeddings_path = os.path.join(run_dir, sub_dir, "embeddings.npy")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    
    return np.load(embeddings_path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree

# Simple Helix Shape
def helix():
    # Load points from 'helix.txt'
    return np.loadtxt('./helix/helix.txt', delimiter=',')

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

def get_embeddings(tree):
    nodes, dim = best_depth(tree.root)

    basis = np.vstack([node.basis for node in nodes])
    idxs = np.hstack([node.idxs for node in nodes])
    sigmas = np.hstack([node.sigmas[:-1] for node in nodes])

    print(f"Basis shape: {basis.shape}")
    print(f"Idxs shape: {idxs.shape}")
    print(f"Sigmas shape: {sigmas.shape}")
    print(f"Max index in idxs: {np.max(idxs)}")
    print(f"Min index in idxs: {np.min(idxs)}")

    if basis.shape[0] != sigmas.shape[0]:
        print("Warning: Mismatch between basis and sigmas shapes. Adjusting sigmas.")
        if sigmas.shape[0] < basis.shape[0]:
            repeat_factor = basis.shape[0] // sigmas.shape[0] + 1
            sigmas = np.tile(sigmas, repeat_factor)[:basis.shape[0]]
        else:
            sigmas = sigmas[:basis.shape[0]]
        print(f"Adjusted sigmas shape: {sigmas.shape}")

    embeddings = np.multiply(basis, sigmas.reshape((basis.shape[0], 1)))

    max_idx = np.max(idxs)
    reordered_embs = np.zeros((max_idx + 1, embeddings.shape[1]))
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        if new_idx < embeddings.shape[0]:
            reordered_embs[new_idx] = embeddings[idx]
        else:
            print(f"Warning: Index {new_idx} is out of bounds. Skipping.")

    print(f"Final embeddings shape: {reordered_embs.shape}")
    return reordered_embs

def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', 'pubmed', 'disease'], help="Dataset to use")
    parser.add_argument("--model", type=str, default="HGCN", choices=['HGCN', 'HNN', 'GCN', 'GAT'], help="Model used for embeddings")
    parser.add_argument("--task", type=str, default="lp", choices=['lp', 'nc'], help="Task performed (lp: link prediction, nc: node classification)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where the covertree is saved")
    args = parser.parse_args()

    print("loading data")
    start_time = time.time()
    X = load_hyperbolic_embeddings(args.dataset, args.model, args.task)
    X = pt.from_numpy(X.astype(np.float32))
    print(f"Loaded embeddings for {args.dataset} using {args.model} for task {args.task}")
    print("First 15 Hyperbolic Embeddings:")
    print(X[:15, :])
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    covertree_path = os.path.join(args.data_dir, f"{args.dataset}_{args.model}_{args.task}_covertree.json")
    if not os.path.exists(covertree_path):
        raise FileNotFoundError(f"Covertree file not found at {covertree_path}. Please run covertree_build_hgcn.py first.")

    print("loading covertree from [%s]" % covertree_path)
    start_time = time.time()
    cover_tree = CoverTree(covertree_path)
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

    # Set the desired scale to 0 (the roughest scale)
    desired_scale = 0

    print("Extracting low-dimensional embeddings at scale {0}".format(desired_scale))
    start_time = time.time()
    embeddings = get_embeddings(wavelet_tree)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time - start_time))

    # Output the embeddings to a text file
    output_dir = f"./results/{args.dataset}_{args.model}_{args.task}"
    output_path = os.path.join(output_dir, "gmra_embeddings.txt")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in embeddings:
            if np.any(embedding):  # Only write non-zero embeddings
                file.write(" ".join(map(str, embedding)) + "\n")

    print("Low-dimensional embeddings saved to {0}".format(output_path))
    print(f"Number of non-zero embeddings: {np.sum(np.any(embeddings, axis=1))}")

if __name__ == "__main__":
    main()