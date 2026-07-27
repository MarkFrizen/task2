import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
embeddings = np.load("embeddings.npy")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

# 2D визуализация через PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 8))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.7)
for i, txt in enumerate(docs):
    if i < 10:
        plt.annotate(f"Doc{i}", (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8)
plt.title("PCA 2D проекция")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# 3D визуализация через t-SNE или PCA
if len(embeddings) > 2:
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    emb_3d = tsne.fit_transform(embeddings)
    method = "t-SNE"
else:
    pca3 = PCA(n_components=3)
    emb_3d = pca3.fit_transform(embeddings)
    method = "PCA"
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], alpha=0.7)
for i, txt in enumerate(docs):
    if i < 10:
        ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2], f"Doc{i}", fontsize=8)
ax.set_title(f"{method} 3D проекция")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()