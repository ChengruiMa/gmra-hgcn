import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import sys
sys.path.append('/scratch/f0071gk/mcas-gmra/pymm-gmra')
from hgcn.manifolds.poincare import PoincareBall

# Initialize the Poincaré ball manifold
manifold = PoincareBall()
c = 1.0  # curvature, adjust if you used a different value

def load_embeddings(file_path):
    return np.loadtxt(file_path)

def hyperbolic_distance(x, y):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return manifold.sqdist(x, y, c).sqrt().item()

class HyperbolicKMeans(KMeans):
    def _euclidean_distances(self, X, Y):
        return np.array([[hyperbolic_distance(x, y) for y in Y] for x in X])

def classification_task(embeddings, labels, is_hyperbolic=False):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    if is_hyperbolic:
        # For hyperbolic space, we need to use a custom kernel
        def hyperbolic_kernel(X, Y):
            return np.array([[np.exp(-hyperbolic_distance(x, y)) for y in Y] for x in X])
        clf = SVC(kernel=hyperbolic_kernel)
    else:
        clf = SVC()
    
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return accuracy_score(y_test, predictions)

def clustering_task(embeddings, true_labels, n_clusters, is_hyperbolic=False):
    if is_hyperbolic:
        kmeans = HyperbolicKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return adjusted_rand_score(true_labels, cluster_labels)

def link_prediction_task(embeddings, is_hyperbolic=False):
    n_samples = embeddings.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Create a graph where nodes are connected if they're among 5 nearest neighbors
    if is_hyperbolic:
        distances = np.array([[hyperbolic_distance(x, y) for y in embeddings] for x in embeddings])
        nn = NearestNeighbors(n_neighbors=5, metric='precomputed')
        nn.fit(distances)
    else:
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(embeddings)
    
    graph = nn.kneighbors_graph(mode='connectivity')
    
    # Split into train and test
    train_graph = graph[:n_train, :n_train]
    test_graph = graph[n_train:, :n_train]
    
    # Train a new nearest neighbors model on the training embeddings
    if is_hyperbolic:
        train_distances = distances[:n_train, :n_train]
        nn_pred = NearestNeighbors(n_neighbors=5, metric='precomputed')
        nn_pred.fit(train_distances)
    else:
        nn_pred = NearestNeighbors(n_neighbors=5)
        nn_pred.fit(embeddings[:n_train])
    
    # Predict links for test nodes
    if is_hyperbolic:
        test_distances = distances[n_train:, :n_train]
        pred_graph = nn_pred.kneighbors_graph(test_distances, mode='connectivity')
    else:
        pred_graph = nn_pred.kneighbors_graph(embeddings[n_train:], mode='connectivity')
    
    # Compute accuracy
    accuracy = np.mean(test_graph.toarray() == pred_graph.toarray())
    return accuracy

def visualize_embeddings(euclidean, hyperbolic, output_file):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(euclidean[:, 0], euclidean[:, 1])
    plt.title("Euclidean Embeddings")
    plt.subplot(122)
    plt.scatter(hyperbolic[:, 0], hyperbolic[:, 1])
    plt.title("Hyperbolic Embeddings")
    plt.savefig(output_file)
    plt.close()

def main():
    # Load embeddings
    euclidean_embeddings = load_embeddings('./helix/results/dram.txt')
    hyperbolic_embeddings = load_embeddings('./helix/results/hyperbolic_dram.txt')

    print("Euclidean embedding shape:", euclidean_embeddings.shape)
    print("Hyperbolic embedding shape:", hyperbolic_embeddings.shape)

    # Generate some dummy labels for demonstration
    n_samples = euclidean_embeddings.shape[0]
    euclidean_labels = np.random.randint(0, 5, size=n_samples)
    hyperbolic_labels = np.random.randint(0, 5, size=n_samples)

    # Classification task
    euclidean_accuracy = classification_task(euclidean_embeddings, euclidean_labels)
    hyperbolic_accuracy = classification_task(hyperbolic_embeddings, hyperbolic_labels, is_hyperbolic=True)
    print(f"Classification Accuracy - Euclidean: {euclidean_accuracy:.4f}, Hyperbolic: {hyperbolic_accuracy:.4f}")

    # Clustering task
    euclidean_ari = clustering_task(euclidean_embeddings, euclidean_labels, n_clusters=5)
    hyperbolic_ari = clustering_task(hyperbolic_embeddings, hyperbolic_labels, n_clusters=5, is_hyperbolic=True)
    print(f"Clustering ARI - Euclidean: {euclidean_ari:.4f}, Hyperbolic: {hyperbolic_ari:.4f}")

    # Link prediction task
    euclidean_link_accuracy = link_prediction_task(euclidean_embeddings)
    hyperbolic_link_accuracy = link_prediction_task(hyperbolic_embeddings, is_hyperbolic=True)
    print(f"Link Prediction Accuracy - Euclidean: {euclidean_link_accuracy:.4f}, Hyperbolic: {hyperbolic_link_accuracy:.4f}")

    # Visualize embeddings
    visualize_embeddings(euclidean_embeddings, hyperbolic_embeddings, "embedding_comparison.png")
    print("Embedding visualization saved as 'embedding_comparison.png'")

if __name__ == "__main__":
    main()