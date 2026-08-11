import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Загружаем эмбеддинги, тексты чанков и метаданные
embeddings = np.load("embeddings.npy")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Определяем уникальные имена файлов и назначаем каждому свой цвет
sources = [m['source'] for m in metadatas]
unique_sources = list(set(sources))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
source_to_color = {src: colors[i] for i, src in enumerate(unique_sources)}

# Группируем индексы чанков по файлам и сортируем по порядковому номеру чанка
indices_by_source = {}
for src in unique_sources:
    idx_list = [i for i, m in enumerate(metadatas) if m['source'] == src]
    idx_list.sort(key=lambda i: metadatas[i]['chunk_id'])
    indices_by_source[src] = idx_list

# Построение двумерной проекции методом главных компонент
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(14, 10))

# Рисуем все точки чанков и подписываем их номерами
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=color, alpha=0.7, s=30)
    plt.annotate(f"Чанк {i+1}", (emb_2d[i, 0], emb_2d[i, 1]),
                 fontsize=5, alpha=0.5, ha='center', va='center')

# Соединяем чанки одного файла линиями в порядке их следования
for src in unique_sources:
    indices = indices_by_source[src]
    if len(indices) > 1:
        points = np.array([emb_2d[i] for i in indices])
        plt.plot(points[:, 0], points[:, 1], color=source_to_color[src],
                 linewidth=1.5, alpha=0.7)

# Оформляем график: заголовок, подписи осей и сетка
plt.title("PCA 2D проекция с траекториями чанков")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle='--', alpha=0.3)

# Добавляем легенду с названиями файлов
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=source_to_color[src], label=src)
                   for src in unique_sources]
plt.legend(handles=legend_elements, loc='best', fontsize=8)
plt.tight_layout()
plt.show()

# Построение трёхмерной проекции
# Если чанков больше двух, используем t-SNE, иначе PCA
if len(embeddings) > 2:
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    emb_3d = tsne.fit_transform(embeddings)
    method = "t-SNE"
else:
    pca3 = PCA(n_components=3)
    emb_3d = pca3.fit_transform(embeddings)
    method = "PCA"
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем все точки в 3D и подписываем их
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    ax.scatter(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
               color=color, alpha=0.6, s=25)
    ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
            f"Чанк {i+1}", fontsize=4, alpha=0.3, ha='center', va='center')

# Соединяем чанки одного файла линиями в трёхмерном пространстве
for src in unique_sources:
    indices = indices_by_source[src]
    if len(indices) > 1:
        points = np.array([emb_3d[i] for i in indices])
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                color=source_to_color[src], linewidth=1.5, alpha=0.7)

# Оформляем трёхмерный график
ax.set_title(f"{method} 3D проекция с траекториями чанков")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Легенда для 3D
legend_elements_3d = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=source_to_color[src],
                                 label=src, markersize=8)
                      for src in unique_sources]
ax.legend(handles=legend_elements_3d, loc='best', fontsize=8)
plt.tight_layout()
plt.show()