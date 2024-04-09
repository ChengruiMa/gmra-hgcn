# -*- coding: utf-8 -*-
"""watch-your-step-embeddings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/embeddings/watch-your-step-embeddings.ipynb

# Node representation learning with Watch Your Step

> This notebook demonstrates how to use the StellarGraph implementation of [Watch Your Step](https://arxiv.org/abs/1710.09599).

<table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/embeddings/watch-your-step-embeddings.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/embeddings/watch-your-step-embeddings.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>
"""

# Commented out IPython magic to ensure Python compatibility.
# install StellarGraph if running on Google Colab
import sys
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None

from stellargraph.core import StellarGraph
from stellargraph.mapper import AdjacencyPowerGenerator
from stellargraph.layer import WatchYourStep
from stellargraph.losses import graph_log_likelihood
from stellargraph import datasets
from stellargraph.utils import plot_history

from matplotlib import pyplot as plt
from tensorflow.keras import optimizers, Model, layers, regularizers
import tensorflow as tf
from sklearn import preprocessing, feature_extraction, model_selection
from IPython.display import display, HTML

import networkx as nx
import random
import numpy as np
import pandas as pd
import os

tf.random.set_seed(1234)

"""## Loading in the data

(See [the "Loading from Pandas" demo](../basics/loading-pandas.ipynb) for details on how data can be loaded.)
"""

dataset = datasets.Cora()
display(HTML(dataset.description))
G, subjects = dataset.load()

"""## Creating the model

We create an `AdjacencyPowerGenerator` which loops through the rows of the first `num_powers` of the adjacency matrix.
"""

generator = AdjacencyPowerGenerator(G, num_powers=10)

"""Next, we use the `WatchYourStep` class to create trainable node embeddings and expected random walks."""

wys = WatchYourStep(
    generator,
    num_walks=80,
    embedding_dimension=128,
    attention_regularizer=regularizers.l2(0.5),
)
x_in, x_out = wys.in_out_tensors()

"""We use the graph log likelihood as our loss function."""

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=graph_log_likelihood, optimizer=tf.keras.optimizers.Adam(1e-3))

"""## Training

We now create a training generator and fit our model.
"""

epochs = 100

batch_size = 10
train_gen = generator.flow(batch_size=batch_size, num_parallel_calls=10)

history = model.fit(
    train_gen, epochs=epochs, verbose=1, steps_per_epoch=int(len(G.nodes()) // batch_size)
)

plot_history(history)

"""## Visualizing Embeddings

Now we use TSNE to visualize the embeddings.
"""

embeddings = wys.embeddings()

import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

nodelist = list(G.nodes())

labels = subjects.loc[nodelist]
target_encoding = OneHotEncoder(sparse_output=False)
label_vectors = target_encoding.fit_transform(labels.values.reshape(-1, 1))

transform = TSNE

trans = transform(n_components=2)
emb_transformed = pd.DataFrame(trans.fit_transform(embeddings), index=nodelist)

emb_transformed["label"] = np.argmax(label_vectors, 1)

alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["label"].astype("category"),
    cmap="jet",
    alpha=alpha,
)

ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(
    "{} visualization of Watch Your Step embeddings for cora dataset".format(
        transform.__name__
    )
)
plt.savefig("/scratch/f006dg0/mcas-gmra/experiments/results/watch_your_step_embeddings.png")

"""## Classification

Here, we predict the class of a node by performing a weighted average of the training labels, with the weights determined by the similarity of that node's embedding with the training node embeddings.
"""

# choose a random set of training nodes by permuting the labels and taking the first 300.
shuffled_idx = np.random.permutation(label_vectors.shape[0])
train_node_idx = shuffled_idx[:300]
test_node_idx = shuffled_idx[300:]

training_labels = label_vectors.copy()
training_labels[test_node_idx] = 0

d = embeddings.shape[1] // 2

predictions = np.dot(
    np.exp(np.dot(embeddings[:, :d], embeddings[:, d:].transpose())), training_labels
)

np.mean(
    np.argmax(predictions[test_node_idx], 1) == np.argmax(label_vectors[test_node_idx], 1)
)

"""<table><tr><td>Run the latest release of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/embeddings/watch-your-step-embeddings.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/embeddings/watch-your-step-embeddings.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>"""