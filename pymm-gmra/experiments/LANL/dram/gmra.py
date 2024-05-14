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

# Read Data
def read_data(file_path):
    # Initialize an empty list to store the data
    data_list = []

    # Open the file and read its contents
    with open(file_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into three values using a comma as the delimiter
            values = line.strip().split(' ')
            
            # Convert each value to a float and append to the list
            float_values = [float(val) for val in values[1:]]
            data_list.append(float_values)

    # Convert the list of lists to a PyTorch tensor
    tensor_data = torch.tensor(data_list)

    return tensor_data

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
    #returns the embeddings matrix and the idxs map
    nodes, dim = best_depth(tree.root)

    #aggregate nodes along depth
    #need basis, idxs, and sigmas
    basis = np.vstack([node.basis for node in nodes])
    idxs = np.hstack([node.idxs for node in nodes])
    sigmas = np.hstack([node.sigmas[:-1] for node in nodes])

    #TODO check dimensions of basis, idxs, sigmas
    #expect: basis nxd where n is 1000 (num nodes) and d is the best dim
    #idxs is a column vector of length 1000 (node idxs corresponding to the elements in the basis)
    #sigmas is a column vector of length 1000 (scaling factors for each basis vector)
    embeddings = np.multiply(basis, sigmas.reshape((basis.shape[0],1)))

    #we need to reorder embeddings based on sigmas
    reordered_embs = np.zeros(embeddings.shape)
    for idx in range(len(idxs)):
        new_idx = idxs[idx]
        reordered_embs[new_idx] = embeddings[idx]
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
    X = read_data("/scratch/f006dg0/end/pymm-gmra/experiments/LANL/dataset/auth_10mil_4_256.txt")
    # X = pt.from_numpy(X.astype(np.float32))
    # Print the 3D points
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

    # Set the desired scale to 0 (the roughest scale)
    desired_scale = 0

    print("Extracting low-dimensional embeddings at scale {0}".format(desired_scale))
    start_time = time.time()
    embeddings = get_embeddings(wavelet_tree)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time - start_time))

    # Output the embeddings to a text file
    output_dir = "./helix/results"
    output_path = os.path.join(output_dir, "dram.txt")

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the file in 'w' mode (write mode)
    with open(output_path, 'w') as file:
        for embedding in embeddings:
            file.write(" ".join(map(str, embedding)) + "\n")

    print("Low-dimensional embeddings saved to {0}".format(output_path))

if __name__ == "__main__":
    main()
