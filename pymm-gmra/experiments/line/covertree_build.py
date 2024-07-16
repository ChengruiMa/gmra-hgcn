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

# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree

# Simple Line Shape
def line():
    # Load points from 'line.txt'
    return np.loadtxt('./line.txt', delimiter=',')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="directory to where covertree will serialize itself to")
    parser.add_argument("--validate", action="store_true",
                        help="if enabled, perform an expensive tree validate operation")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    X = line()
    X = X.astype(np.float32)
    X_pt = pt.from_numpy(X)
    print("done")

    cover_tree = CoverTree(max_scale=2)

    for pt_idx in tqdm(list(range(X_pt.shape[0])),
                       desc="building covertree"):
        cover_tree.insert_pt(pt_idx, X_pt)

    if(args.validate):
        print("validating covertree...this may take a while")
        assert(cover_tree.validate(X_pt))

    filename = "line_covertree.json"
    filepath = os.path.join(args.data_dir, filename)

    print("serializing covertree to [%s]" % filepath)
    cover_tree.save(filepath)

if __name__ == "__main__":
    main()
