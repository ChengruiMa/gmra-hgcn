import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
import torch

# Load the original HGCN embeddings
original_embeddings = np.load('/path/to/original/embeddings.npy')

# Load the GMRA embeddings
gmra_embeddings = np.loadtxt('./results/cora_HGCN_lp/gmra_embeddings.txt')

# Visualization
plt.figure(figsize=(15, 5))

# Original embeddings (using t-SNE for visualization)
plt.subplot(121)
tsne = TSNE(n_components=2, random_state=42)
original_2d = tsne.fit_transform(original_embeddings)
plt.scatter(original_2d[:, 0], original_2d[:, 1], alpha=0.5)
plt.title('Original HGCN Embeddings (t-SNE)')

# GMRA embeddings
plt.subplot(122)
plt.scatter(gmra_embeddings[:, 0], gmra_embeddings[:, 1], alpha=0.5)
plt.title('GMRA Embeddings (2D projection)')

plt.tight_layout()
plt.savefig('embedding_comparison.png')
plt.close()

# Link Prediction Evaluation
# You'll need to adapt this part based on your specific link prediction setup
def evaluate_link_prediction(embeddings, edges_pos, edges_neg):
    # This is a placeholder function. You'll need to implement the actual evaluation logic
    # based on how you're doing link prediction with the original HGCN embeddings
    pass

# Load your test edges (positive and negative)
# edges_pos = ...
# edges_neg = ...

# Evaluate original embeddings
original_score = evaluate_link_prediction(original_embeddings, edges_pos, edges_neg)

# Evaluate GMRA embeddings
gmra_score = evaluate_link_prediction(gmra_embeddings, edges_pos, edges_neg)

print(f"Original HGCN ROC-AUC: {original_score}")
print(f"GMRA ROC-AUC: {gmra_score}")