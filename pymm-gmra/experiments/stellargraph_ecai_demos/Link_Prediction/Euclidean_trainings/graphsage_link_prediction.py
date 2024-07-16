import sys
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow import keras
from stellargraph import datasets
from IPython.display import display, HTML

# Import Dataset
dataset = datasets.Cora()
display(HTML(dataset.description))
G, _ = dataset.load(subject_as_feature=True)

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

batch_size = 20
epochs = 20
num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [20, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

# Stack the GraphSAGE and prediction layers into a Keras model, and specify the loss
model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

# Evaluate the initial (untrained) model on the train and test set:
init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Train the model:
history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Extract node embeddings from the trained model
node_embeddings_model = keras.Model(inputs=x_inp, outputs=x_out)
node_embeddings = node_embeddings_model.predict(train_gen.flow(G_train.nodes()))

import numpy as np

# Save node embeddings and corresponding labels to files
np.savetxt('/scratch/f006dg0/stellargraph_ecai_demos/Link_Prediction/results/node_embeddings.csv', node_embeddings, delimiter=',')
np.savetxt('/scratch/f006dg0/stellargraph_ecai_demos/Link_Prediction/results/node_labels.csv', edge_labels_train, delimiter=',')

from sklearn.metrics import accuracy_score

# Predict labels for the train and test sets
# train_predictions = model.predict(train_flow)
test_predictions = model.predict(test_flow)

# Convert predicted probabilities to binary labels (0 or 1)
# train_pred_labels = (train_predictions > 0.5).astype(int).flatten()
test_pred_labels = (test_predictions > 0.5).astype(int).flatten()

# Get true labels for train and test sets
# train_true_labels = edge_labels_train
test_true_labels = edge_labels_test

# Calculate accuracy scores
# train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
test_accuracy = accuracy_score(test_true_labels, test_pred_labels)

# Print the accuracy scores
# print("\nTrain Set Accuracy using sklearn.metrics.accuracy_score:", train_accuracy)
print("\nTest Set Accuracy using sklearn.metrics.accuracy_score:", test_accuracy)
