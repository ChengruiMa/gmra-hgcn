import sys
import os
import numpy as np
import torch
from stellargraph import datasets
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# Add HGCN to the Python path
sys.path.append('/scratch/f0071gk/hgcn')

from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name

# Load Cora dataset
dataset = datasets.Cora()
graph, node_subjects = dataset.load()

# Prepare data in the format HGCN expects
adj = graph.to_adjacency_matrix()
features = graph.node_features()

# Convert adjacency matrix to sparse format
adj_sparse = sp.csr_matrix(adj)

# Split the graph for link prediction
edge_splitter_test = EdgeSplitter(graph)
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(p=0.1, method="global")

edge_splitter_train = EdgeSplitter(graph_test, graph)
graph_train, examples, labels = edge_splitter_train.train_test_split(p=0.1, method="global")
examples_train, examples_val, labels_train, labels_val = train_test_split(examples, labels, train_size=0.75, test_size=0.25, random_state=123)

# Set up args
args = parser.parse_args()
args.task = 'lp'
args.dataset = 'cora'
args.model = 'HGCN'
args.dropout = 0.5
args.lr = 0.01
args.dim = 16
args.num_layers = 2
args.act = 'relu'
args.bias = 1
args.weight_decay = 0.001
args.manifold = 'PoincareBall'
args.log_freq = 5
args.cuda = 0
args.c = None
args.epochs = 200

# Prepare data in HGCN format
data = {
    'adj_train': adj_sparse,
    'features': features,
    'labels': np.array([node_subjects[node] for node in graph.nodes()]),
    'idx_train': np.arange(len(examples_train)),
    'idx_val': np.arange(len(examples_train), len(examples_train) + len(examples_val)),
    'idx_test': np.arange(len(examples_train) + len(examples_val), len(graph.nodes())),
    'train_edges': examples_train,
    'train_edges_false': examples_train[labels_train == 0],
    'val_edges': examples_val,
    'val_edges_false': examples_val[labels_val == 0],
    'test_edges': examples_test,
    'test_edges_false': examples_test[labels_test == 0]
}

# Model and optimizer
model = LPModel(args)
optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Train model
best_val_metrics = model.init_metric_dict()
best_test_metrics = None
best_emb = None

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    embeddings = model.encode(data['features'], data['adj_train'])
    train_metrics = model.compute_metrics(embeddings, data, 'train')
    train_metrics['loss'].backward()
    optimizer.step()
    
    if (epoch + 1) % args.log_freq == 0:
        model.eval()
        embeddings = model.encode(data['features'], data['adj_train'])
        val_metrics = model.compute_metrics(embeddings, data, 'val')
        
        if model.has_improved(best_val_metrics, val_metrics):
            best_test_metrics = model.compute_metrics(embeddings, data, 'test')
            best_emb = embeddings.cpu()
            best_val_metrics = val_metrics
        
        print(f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, Val AUC: {val_metrics['AUC']:.4f}")

# Save embeddings
output_dir = "/scratch/f0071gk/mcas-gmra/pymm-gmra/experiments/stellargraph_ecai_demos/"
output_file = f"cora_hgcn_{args.dim}.txt"
output_path = os.path.join(output_dir, output_file)

with open(output_path, 'w') as f:
    for node_id, embedding in zip(graph.nodes(), best_emb.detach().numpy()):
        f.write(f"{node_id} {' '.join(map(str, embedding))}\n")

print(f"HGCN Embeddings saved to {output_path}")
print("Test set results:", best_test_metrics)