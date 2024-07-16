import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML

dataset = datasets.Cora()
display(HTML(dataset.description))
G, _ = dataset.load(subject_as_feature=True)

print(G.info())

edge_splitter_test = EdgeSplitter(G)

G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

edge_splitter_train = EdgeSplitter(G_test)

G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

print(G_train.info())
print(G_test.info())

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

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)
test_embeddings = model.predict(test_flow)

# Add this import statement at the beginning of your code
from sklearn.metrics import accuracy_score

# Train the model and get the training history
history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

# Get the predicted labels for the test data
predicted_labels = model.predict(test_flow)
predicted_labels_binary = (predicted_labels > 0.5).astype(int)  # Assuming binary classification

# Calculate the accuracy score
accuracy = accuracy_score(edge_labels_test, predicted_labels_binary)

# Print the accuracy score
print("\nAccuracy Score:", accuracy)

# import os

# # Output the embeddings to a text file
# output_dir = "/scratch/f006dg0/experiments/Link_Prediction/results"
# output_path = os.path.join(output_dir, "cora_links_example.txt")

# # Create the directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Open the file in 'w' mode (write mode)
# with open(output_path, 'w') as file:
#     for embedding in test_embeddings:
#         file.write(" ".join(map(str, embedding)) + "\n")

"""
# Plot the training history:
sg.utils.plot_history(history)

# Evaluate the trained model on test citation links:
train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
"""