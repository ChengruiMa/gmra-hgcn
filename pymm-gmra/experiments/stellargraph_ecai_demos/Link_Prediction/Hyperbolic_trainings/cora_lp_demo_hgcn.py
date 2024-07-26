import sys
import os
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

# Add HGCN to the Python path
sys.path.append('/scratch/f0071gk/mcas-gmra/hgcn')

from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data_lp
from utils.train_utils import get_dir_name
from manifolds.poincare import PoincareBall

# Set up args
args = parser.parse_args()
args.task = 'lp'
args.dataset = 'cora'
args.model = 'HGCN'
args.dropout = 0.5
args.lr = 0.001
args.num_layers = 2
args.act = 'relu'
args.bias = 1
args.weight_decay = 0.001
args.manifold = 'PoincareBall'
args.log_freq = 5
args.cuda = 0
args.c = 1.0  # Set a fixed curvature
args.epochs = 500
args.use_feats = True
args.data_path = None
args.optimizer = 'Adam'

# Load data
data = load_data_lp(args.dataset, args.use_feats, args.data_path)

# Get dimensions
args.n_nodes, args.feat_dim = data['features'].shape

# Convert to PyTorch tensors
for key in data.keys():
    if sp.issparse(data[key]):
        data[key] = torch.FloatTensor(data[key].todense())
    elif isinstance(data[key], np.ndarray):
        data[key] = torch.FloatTensor(data[key])
    else:
        data[key] = torch.FloatTensor(data[key])

# Split edges for link prediction
num_edges = data['adj_train'].sum().item() // 2
num_nodes = data['adj_train'].shape[0]

all_edges = []
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        if data['adj_train'][i, j] == 1:
            all_edges.append((i, j))

train_edges, test_edges = train_test_split(all_edges, test_size=0.1, random_state=42)
train_edges, val_edges = train_test_split(train_edges, test_size=0.1, random_state=42)

# Create negative edges
def create_neg_edges(pos_edges, num_nodes, num_neg):
    neg_edges = []
    while len(neg_edges) < num_neg:
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j and data['adj_train'][i, j] == 0 and (i, j) not in neg_edges:
            neg_edges.append((i, j))
    return neg_edges

train_edges_false = create_neg_edges(train_edges, num_nodes, len(train_edges))
val_edges_false = create_neg_edges(val_edges, num_nodes, len(val_edges))
test_edges_false = create_neg_edges(test_edges, num_nodes, len(test_edges))

args.nb_false_edges = len(train_edges_false)
args.nb_edges = len(train_edges)

# Prepare data for HGCN
data['train_edges'] = torch.LongTensor(train_edges)
data['train_edges_false'] = torch.LongTensor(train_edges_false)
data['val_edges'] = torch.LongTensor(val_edges)
data['val_edges_false'] = torch.LongTensor(val_edges_false)
data['test_edges'] = torch.LongTensor(test_edges)
data['test_edges_false'] = torch.LongTensor(test_edges_false)

# Set up CUDA
args.cuda = int(args.cuda and torch.cuda.is_available())
if args.cuda:
    torch.cuda.set_device(args.cuda)
device = 'cuda:' + str(args.cuda) if args.cuda else 'cpu'
args.device = device

# Move data to device
for key in data:
    if torch.is_tensor(data[key]):
        data[key] = data[key].to(device)

def train_hgcn(dim):
    args.dim = dim
    
    # Model and optimizer
    model = LPModel(args)
    model.to(device)
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Train model
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    print(f"Training HGCN with dimension {dim}")
    print(f"Features shape: {data['features'].shape}")
    print(f"Adj_train shape: {data['adj_train'].shape}")
    print(f"Train edges shape: {data['train_edges'].shape}")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train'])
        
        nan_count = torch.isnan(embeddings).sum().item()
        if nan_count > 0:
            print("NaNs detected in embeddings. Reinitializing model and continuing training.")
            model = LPModel(args)
            model.to(device)
            optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            continue

        try:
            train_metrics = model.compute_metrics(embeddings, data, 'train')
            if torch.isnan(train_metrics['loss']):
                print("NaN loss detected. Skipping this epoch.")
                continue
            train_metrics['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except RuntimeError as e:
            print(f"Error in epoch {epoch}: {str(e)}")
            continue

        if (epoch + 1) % args.log_freq == 0:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(data['features'], data['adj_train'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')
            
            scheduler.step(val_metrics['loss'])
            
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                best_val_metrics = val_metrics
            
            print(f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, Val ROC: {val_metrics['roc']:.4f}, Val AP: {val_metrics['ap']:.4f}")

    # Save embeddings
    if best_emb is not None:
        output_dir = "/scratch/f0071gk/mcas-gmra/pymm-gmra/experiments/stellargraph_ecai_demos/"
        output_file = f"cora_lp_hgcn_{dim}.txt"
        output_path = os.path.join(output_dir, output_file)

        with open(output_path, 'w') as f:
            for node_id, embedding in enumerate(best_emb.numpy()):
                f.write(f"{node_id} {' '.join(map(str, embedding))}\n")

        print(f"HGCN Embeddings saved to {output_path}")
        print("Test set results:", best_test_metrics)
    else:
        print("Training failed to produce valid embeddings.")
    
    return best_emb, best_test_metrics

# Train HGCN for different dimensions
dimensions = [32, 64, 128, 256]
results = {}

for dim in dimensions:
    best_emb, best_test_metrics = train_hgcn(dim)
    results[dim] = {'embeddings': best_emb, 'metrics': best_test_metrics}

# Link prediction using hyperbolic distance
def hyperbolic_distance(u, v):
    diff = u - v
    norm_u = torch.norm(u)
    norm_v = torch.norm(v)
    norm_diff = torch.norm(diff)
    return torch.acosh(1 + 2 * (norm_diff**2) / ((1 - norm_u**2) * (1 - norm_v**2)))

def link_examples_to_features(link_examples, embeddings):
    return torch.stack([hyperbolic_distance(embeddings[src], embeddings[dst]) for src, dst in link_examples])

def train_link_prediction_model(link_examples, link_labels, embeddings):
    clf = LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=2000)
    link_features = link_examples_to_features(link_examples, embeddings)
    clf.fit(link_features.numpy().reshape(-1, 1), link_labels)
    return clf

def evaluate_link_prediction_model(clf, link_examples, link_labels, embeddings):
    link_features = link_examples_to_features(link_examples, embeddings)
    predictions = clf.predict_proba(link_features.numpy().reshape(-1, 1))[:, 1]
    return roc_auc_score(link_labels, predictions)

# Perform link prediction for each dimension
for dim, result in results.items():
    embeddings = result['embeddings']
    
    # Prepare data
    train_examples = torch.cat([data['train_edges'], data['train_edges_false']], dim=0)
    train_labels = torch.cat([torch.ones(data['train_edges'].shape[0]), torch.zeros(data['train_edges_false'].shape[0])])
    val_examples = torch.cat([data['val_edges'], data['val_edges_false']], dim=0)
    val_labels = torch.cat([torch.ones(data['val_edges'].shape[0]), torch.zeros(data['val_edges_false'].shape[0])])
    test_examples = torch.cat([data['test_edges'], data['test_edges_false']], dim=0)
    test_labels = torch.cat([torch.ones(data['test_edges'].shape[0]), torch.zeros(data['test_edges_false'].shape[0])])

    # Train and evaluate
    clf = train_link_prediction_model(train_examples, train_labels, embeddings)
    val_score = evaluate_link_prediction_model(clf, val_examples, val_labels, embeddings)
    test_score = evaluate_link_prediction_model(clf, test_examples, test_labels, embeddings)

    print(f"Dimension {dim}:")
    print(f"Validation ROC AUC: {val_score:.4f}")
    print(f"Test ROC AUC: {test_score:.4f}")
    print()

print("Link prediction completed for all dimensions.")