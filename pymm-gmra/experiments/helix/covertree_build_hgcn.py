# SYSTEM IMPORTS
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

hgcn_path = "/scratch/f0071gk/mcas-gmra/hgcn"
if hgcn_path not in sys.path:
    sys.path.append(hgcn_path)

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree

def load_hyperbolic_embeddings(dataset='cora', model='HGCN', task='lp'):
    logs_dir = os.path.join(hgcn_path, "logs", task)
    runs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    if not runs:
        raise FileNotFoundError(f"No training runs found for task {task}")
    
    latest_run = max(runs)  # Get the most recent run
    embeddings_path = os.path.join(logs_dir, latest_run, "0/embeddings.npy")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")
    
    return np.load(embeddings_path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', 'pubmed', 'disease'], help="Dataset to use")
    parser.add_argument("--model", type=str, default="HGCN", choices=['HGCN', 'HNN', 'GCN', 'GAT'], help="Model used for embeddings")
    parser.add_argument("--task", type=str, default="lp", choices=['lp', 'nc'], help="Task performed (lp: link prediction, nc: node classification)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the covertree")
    parser.add_argument("--validate", action="store_true", help="Validate the covertree after building")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X = load_hyperbolic_embeddings(args.dataset, args.model, args.task)
    X_pt = pt.from_numpy(X.astype(np.float32))
    print("done")

    cover_tree = CoverTree(max_scale=3)

    for pt_idx in tqdm(list(range(X_pt.shape[0])),
                       desc="building covertree"):
        cover_tree.insert_pt(pt_idx, X_pt)

    if args.validate:
        print("validating covertree...this may take a while")
        assert(cover_tree.validate(X_pt))

    filename = f"{args.dataset}_{args.model}_{args.task}_covertree.json"
    filepath = os.path.join(args.data_dir, filename)

    print("serializing covertree to [%s]" % filepath)
    cover_tree.save(filepath)

if __name__ == "__main__":
    main()