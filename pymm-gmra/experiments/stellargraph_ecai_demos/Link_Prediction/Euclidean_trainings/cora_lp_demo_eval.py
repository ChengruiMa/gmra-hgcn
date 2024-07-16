import numpy as np
import pandas as pd
from stellargraph import datasets
from stellargraph.data import EdgeSplitter
import multiprocessing
from sklearn.model_selection import train_test_split
import time

"""## Load the dataset

The Cora dataset is a homogeneous network where all nodes are papers and edges between nodes are citation links, e.g. paper A cites paper B.
"""

dataset = datasets.Cora()
graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)

"""## Construct splits of the input data

We have to carefully split the data to avoid data leakage and evaluate the algorithms correctly:

* For computing node embeddings, a **Train Graph** (`graph_train`)
* For training classifiers, a classifier **Training Set** (`examples_train`) of positive and negative edges that weren't used for computing node embeddings
* For choosing the best classifier, an **Model Selection Test Set** (`examples_model_selection`) of positive and negative edges that weren't used for computing node embeddings or training the classifier
* For the final evaluation, a **Test Graph** (`graph_test`) to compute test node embeddings with more edges than the Train Graph, and a **Test Set** (`examples_test`) of positive and negative edges not used for neither computing the test node embeddings or for classifier training or model selection

###  Test Graph

We begin with the full graph and use the `EdgeSplitter` class to produce:

* Test Graph
* Test set of positive/negative link examples

The Test Graph is the reduced graph we obtain from removing the test set of links from the full graph.
"""

# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(graph)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)

"""### Train Graph
This time, we use the `EdgeSplitter` on the Test Graph, and perform a train/test split on the examples to produce:

* Train Graph
* Training set of link examples
* Set of link examples for model selection
"""

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
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)

"""## Node2Vec

We use Node2Vec [[1]](#refs), to calculate node embeddings. These embeddings are learned in such a way to ensure that nodes that are close in the graph remain close in the embedding space. Node2Vec first involves running random walks on the graph to obtain our context pairs, and using these to train a Word2Vec model.

These are the set of parameters we can use:

* `p` - Random walk parameter "p"
* `q` - Random walk parameter "q"
* `dimensions` - Dimensionality of node2vec embeddings
* `num_walks` - Number of walks from each node
* `walk_length` - Length of each random walk
* `window_size` - Context window size for Word2Vec
* `num_iter` - number of SGD iterations (epochs)
* `workers` - Number of workers for Word2Vec
"""

p = 1.0
q = 1.0
dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = multiprocessing.cpu_count()

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

# Original:
# input_file1 = "/scratch/f006dg0/stellargraph_ecai_demos/cora_lp_demo_32.txt"
# input_file1 = "/scratch/f006dg0/stellargraph_ecai_demos/cora_lp_demo_64.txt"
# input_file1 = "/scratch/f006dg0/stellargraph_ecai_demos/cora_lp_demo_128.txt"
input_file1 = "/scratch/f006dg0/stellargraph_ecai_demos/cora_lp_demo_256.txt"

# Reduced:
# input_file = "/scratch/f006dg0/stellargraph_ecai_demos/Reduced_Embeddings/cora_lp_demo_32_dram.txt"
# input_file = "/thayerfs/home/f006dg0/cora_lp_demo_64_dram.txt"
# input_file = "/thayerfs/home/f006dg0/cora_lp_demo_128_dram.txt"
input_file = "/thayerfs/home/f006dg0/cora_lp_demo_256_dram.txt"

data_dict = {}
with open(input_file1, 'r') as file:
    for line in file:
        entries = line.strip().split(' ')
        node_id = entries[0]
        # float_values = np.array([float(entry) for entry in entries[1:]])
        data_dict[node_id] = None # float_values # For Orginal Only

# Open the text file in read mode
with open(input_file, 'r') as file2:
    lines = file2.readlines()  # Read all lines from the file

# Loop through the lines and assign them to the dictionary keys
for idx, line in enumerate(lines):
    key = list(data_dict.keys())[idx]  # Get the key corresponding to the current index
    entries = line.strip().split(' ')
    float_values = np.array([float(entry) for entry in entries])
    data_dict[key] = float_values

def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_iter,
    )

    def get_embedding(u):
        return data_dict[u] # For the loaded data
        # return model.wv[u]

    return get_embedding

embedding_train = node2vec_embedding(graph_train, "Train Graph")

"""## Train and evaluate the link prediction model

There are a few steps involved in using the Word2Vec model to perform link prediction:
1. We calculate link/edge embeddings for the positive and negative edge samples by applying a binary operator on the embeddings of the source and target nodes of each sampled edge.
2. Given the embeddings of the positive and negative examples, we train a logistic regression classifier to predict a binary value indicating whether an edge between two nodes should exist or not.
3. We evaluate the performance of the link classifier for each of the 4 operators on the training data with node embeddings calculated on the **Train Graph** (`graph_train`), and select the best classifier.
4. The best classifier is then used to calculate scores on the test data with node embeddings calculated on the **Test Graph** (`graph_test`).

Below are a set of helper functions that let us repeat these steps for each of the binary operators.
"""

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
    # Start time
    start_time = time.time()

    predicted_labels = clf.predict(link_features_test)

    # Calculate time taken
    time_taken = time.time() - start_time

    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)

    return score

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])

"""We consider 4 different operators:

* *Hadamard*
* $L_1$
* $L_2$
* *average*

The paper [[1]](#refs) provides a detailed description of these operators. All operators produce link embeddings that have equal dimensionality to the input node embeddings (128 dimensions for our example).
"""

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

# Evaluate all models using the test set
test_results = []
for result in results:
    embedding_test = node2vec_embedding(graph_test, "Test Graph")
    test_score = evaluate_link_prediction_model(
        result["classifier"],
        examples_test,
        labels_test,
        embedding_test,
        result["binary_operator"],
    )
    test_results.append((result["binary_operator"].__name__, test_score))

# Display results for all trained models on the test set
test_results_df = pd.DataFrame(test_results, columns=("Binary Operator", "ROC AUC Score"))

print("Results for all trained models on the test set:")
print(test_results_df)