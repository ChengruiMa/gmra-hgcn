# -*- coding: utf-8 -*-
"""complex.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dzNBFMqTfrNdGk0x0oLtFWdReCEFalk2

# ComplEx on WN18 and FB15K

This notebook reproduces the experiments done in the paper that introduced the ComplEx algorith: Complex Embeddings for Simple Link Prediction, Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and Guillaume Bouchard, ICML 2016. http://jmlr.org/proceedings/papers/v48/trouillon16.pdf

In table 2, the paper reports five metrics measured on the WN18 and FB15K datasets: "raw" MRR (mean reciprocal rank), "filtered" MRR and filtered Hits at {1, 3, 10}. This notebook measures all of these, as well as raw Hits at {1, 3, 10}.

<table><tr><td>Run the master version of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/knowledge-graphs/complex.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/knowledge-graphs/complex.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>
"""

# Commented out IPython magic to ensure Python compatibility.
# install StellarGraph if running on Google Colab
import sys

from stellargraph import datasets, utils
from tensorflow.keras import callbacks, optimizers, losses, metrics, regularizers, Model
import numpy as np
import pandas as pd

from stellargraph.mapper import KGTripleGenerator
from stellargraph.layer import ComplEx

"""## Initialisation

We need to set up our model parameters, like the number of epochs to train for, and the dimension of the embedding vectors we compute for each node and for each edge type.

The evaluation is performed in three steps:

1. Load the data
2. Train a model
3. Evaluate the model

The paper says that it used:
- the AdaGrad optimiser for 1000 epochs with an early stopping criterion evaluated every 50 epochs, but we've found using the Adam optimiser allows for much fewer epochs
- an embedding dimension of 150 or 200, since they had close results
- 10 negative samples (corrupted edges) per positive edge, which gives noticably improved performance on FB15k compared to using 1, and but not for WN18 (the paper evaluated 1, 2, 5 and 10 negative samples)
"""

epochs = 50
embedding_dimension = 200
negative_samples = 10

"""## WN18

The paper uses the WN18 and FB15k datasets for validation. These datasets are not good for evaluating algorithms because they contain "inverse relations", where `(s, r1, o)` implies `(o, r2, s)` for a pair of relation types `r1` and `r2` (for instance, `_hyponym` ("is more specific than") and `_hypernym` ("is more general than") in WN18), however, they work fine to demonstrate StellarGraph's functionality, and are appropriate to compare against the published results.

### Load the data

The dataset comes with a defined train, test and validation split, each consisting of subject, relation, object triples. We can load a `StellarGraph` object with all of the triples, as well as the individual splits as Pandas DataFrames, using the `load` method of the `WN18` dataset.
"""

wn18 = datasets.WN18()
wn18_graph, wn18_train, wn18_test, wn18_valid = wn18.load()

print(wn18_graph.info())

"""### Train a model

The ComplEx algorithm consists of some embedding layers and a scoring layer, but the `ComplEx` object means these details are invisible to us. The `ComplEx` model consumes "knowledge-graph triples", which can be produced in the appropriate format using `KGTripleGenerator`.
"""

wn18_gen = KGTripleGenerator(
    wn18_graph, batch_size=len(wn18_train) // 100  # ~100 batches per epoch
)

wn18_complex = ComplEx(
    wn18_gen,
    embedding_dimension=embedding_dimension,
    embeddings_regularizer=regularizers.l2(1e-7),
)

wn18_inp, wn18_out = wn18_complex.in_out_tensors()

wn18_model = Model(inputs=wn18_inp, outputs=wn18_out)

wn18_model.compile(
    optimizer=optimizers.Adam(lr=0.001),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=[metrics.BinaryAccuracy(threshold=0.0)],
)

"""Inputs for training are produced by calling the `KGTripleGenerator.flow` method, this takes a dataframe with `source`, `label` and `target` columns, where each row is a true edge in the knowledge graph.  The `negative_samples` parameter controls how many random edges are created for each positive edge to use as negative examples for training."""

wn18_train_gen = wn18_gen.flow(
    wn18_train, negative_samples=negative_samples, shuffle=True
)
wn18_valid_gen = wn18_gen.flow(wn18_valid, negative_samples=negative_samples)

wn18_es = callbacks.EarlyStopping(monitor="val_loss", patience=10)
wn18_history = wn18_model.fit(
    wn18_train_gen, validation_data=wn18_valid_gen, epochs=epochs, callbacks=[wn18_es]
)

utils.plot_history(wn18_history)

"""### Evaluate the model

We've now trained a model, so we can apply the evaluation procedure from the paper to it. This is done by taking each test edge `E = (s, r, o)`, and scoring it against all mutations `(s, r, n)` and `(n, r, o)` for every node `n` in the graph, that is, doing a prediction for every one of these edges similar to `E`. The "raw" rank is the number of mutated edges that have a higher predicted score than the true `E`.
"""

wn18_raw_ranks, wn18_filtered_ranks = wn18_complex.rank_edges_against_all_nodes(
    wn18_gen.flow(wn18_test), wn18_graph
)

# helper function to compute metrics from a dictionary of name -> array of ranks
def results_as_dataframe(name_to_results):
    return pd.DataFrame(
        name_to_results.values(),
        columns=["mrr", "hits at 1", "hits at 3", "hits at 10"],
        index=name_to_results.keys(),
    )


def summarise(name_to_ranks):
    return results_as_dataframe(
        {
            name: (
                np.mean(1 / ranks),
                np.mean(ranks <= 1),
                np.mean(ranks < 3),
                np.mean(ranks <= 10),
            )
            for name, ranks in name_to_ranks.items()
        }
    )

summarise({"raw": wn18_raw_ranks, "filtered": wn18_filtered_ranks})

"""For comparison, Table 2 in the paper gives the following results for WN18 (`NaN` denotes values the paper does not include). All of the numbers are similar:"""

results_as_dataframe(
    {"raw": (0.587, None, None, None), "filtered": (0.941, 0.936, 0.945, 0.947)}
)

"""## FB15k

Now that we know the process, we can apply the model on the FB15k dataset in the same way.

### Loading the data
"""

fb15k = datasets.FB15k()
fb15k_graph, fb15k_train, fb15k_test, fb15k_valid = fb15k.load()

print(fb15k_graph.info())

"""### Train a model"""

fb15k_gen = KGTripleGenerator(
    fb15k_graph, batch_size=len(fb15k_train) // 100  # ~100 batches per epoch
)

fb15k_complex = ComplEx(
    fb15k_gen,
    embedding_dimension=embedding_dimension,
    embeddings_regularizer=regularizers.l2(1e-8),
)

fb15k_inp, fb15k_out = fb15k_complex.in_out_tensors()

fb15k_model = Model(inputs=fb15k_inp, outputs=fb15k_out)
fb15k_model.compile(
    optimizer=optimizers.Adam(lr=0.001),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=[metrics.BinaryAccuracy(threshold=0.0)],
)

fb15k_train_gen = fb15k_gen.flow(
    fb15k_train, negative_samples=negative_samples, shuffle=True
)
fb15k_valid_gen = fb15k_gen.flow(fb15k_valid, negative_samples=negative_samples)

fb15k_es = callbacks.EarlyStopping(monitor="val_loss", patience=10)
fb15k_history = fb15k_model.fit(
    fb15k_train_gen, validation_data=fb15k_valid_gen, epochs=epochs, callbacks=[fb15k_es]
)

utils.plot_history(fb15k_history)

"""### Evaluate the model"""

fb15k_raw_ranks, fb15k_filtered_ranks = fb15k_complex.rank_edges_against_all_nodes(
    fb15k_gen.flow(fb15k_test), fb15k_graph
)

summarise({"raw": fb15k_raw_ranks, "filtered": fb15k_filtered_ranks})

"""For comparison, Table 2 in the paper gives the following results for FB15k:"""

results_as_dataframe(
    {"raw": (0.242, None, None, None), "filtered": (0.692, 0.599, 0.759, 0.850)}
)

"""<table><tr><td>Run the master version of this notebook:</td><td><a href="https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/link-prediction/knowledge-graphs/complex.ipynb" alt="Open In Binder" target="_parent"><img src="https://mybinder.org/badge_logo.svg"/></a></td><td><a href="https://colab.research.google.com/github/stellargraph/stellargraph/blob/master/demos/link-prediction/knowledge-graphs/complex.ipynb" alt="Open In Colab" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a></td></tr></table>"""