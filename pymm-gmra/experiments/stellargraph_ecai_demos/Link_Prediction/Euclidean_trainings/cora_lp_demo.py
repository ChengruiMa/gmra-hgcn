
import sys
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split

dataset = datasets.Cora()
graph, _ = dataset.load(str_node_ids=True)

# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(graph)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)

# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(graph_test, graph)
graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global"
)
(
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25, random_state=123)

p = 1.0
q = 1.0
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = multiprocessing.cpu_count()

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

# Original:
input_file = "/scratch/f0071gk/mcas-gmra/pymm-gmra/experiments/stellargraph_ecai_demos/Link_Prediction/results/cora_lp_demo.txt"
# Reduced:
# input_file = "/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/cora_lp_demo_dram.txt"
    
data_dict = {}
with open(input_file, 'r') as file:
    for line in file:
        entries = line.strip().split(' ')
        node_id = entries[0]
        float_values = np.array([float(entry) for entry in entries[1:]])
        data_dict[node_id] = float_values

def node2vec_embedding(graph, name):
    dimensions = [32, 64, 128, 256]
    for dim in dimensions:
        rw = BiasedRandomWalk(graph)
        walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
        print(f"Number of random walks for '{name}': {len(walks)}")

        model = Word2Vec(
            walks,
            vector_size=dim,
            window=window_size,
            min_count=0,
            sg=1,
            workers=workers,
            epochs=num_iter,
        )

        # Define the output directory and file name
        output_dir = "/scratch/f0071gk/mcas-gmra/pymm-gmra/experiments/stellargraph_ecai_demos/"
        output_file = f"cora_lp_demo_{dim}.txt"  # Use dim instead of dimensions
        output_path = os.path.join(output_dir, output_file)

        # Check if the output directory exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get the list of node IDs from the Word2Vec model
        node_ids = model.wv.index_to_key

        with open(output_path, 'w') as file:
            for node_id, embedding in zip(node_ids, model.wv.vectors):
                # Write node ID and embedding to the file
                file.write(f"{node_id} {' '.join(map(str, embedding))}\n")

        print(f"Embeddings saved to {output_path}")

    print("All dimensions processed.")

    def get_embedding(u):
        # return data_dict[u] # For the loaded data
        return model.wv[u]

    return get_embedding

embedding_train = node2vec_embedding(graph_train, "Train Graph")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]

# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )

    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

# from sklearn.metrics import accuracy_score

# # Define a function to evaluate accuracy
# def evaluate_accuracy(clf, link_examples_test, link_labels_test, get_embedding, binary_operator):
#     link_features_test = link_examples_to_features(link_examples_test, get_embedding, binary_operator)
#     predictions = clf.predict(link_features_test)
#     accuracy = accuracy_score(link_labels_test, predictions)
#     return accuracy

# # Train and evaluate the classifier
# clf = train_link_prediction_model(examples_train, labels_train, embedding_train, np.multiply)
# accuracy = evaluate_accuracy(clf, examples_model_selection, labels_model_selection, embedding_train, np.multiply)

# # Print the accuracy
# print(f"Accuracy: {accuracy:.2f}")

# Stopped Here for Edge Embeddings
# Uncomment the following Lines For Link Prediction Results

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0

def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }

binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]

results = [run_link_prediction(op) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])

print(f"Best result from '{best_result['binary_operator'].__name__}'")

pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")

### Evaluate the best model using the test set
# Now that we've trained and selected our best model, we use a test set of embeddings and calculate a final evaluation score.

embedding_test = node2vec_embedding(graph_test, "Test Graph")

test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_test,
    best_result["binary_operator"],
)

print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
)