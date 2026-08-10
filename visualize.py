# Импортируем библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Загружаем эмбеддинги, тексты и метаданные
embeddings = np.load("embeddings.npy")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Получаем уникальные имена файлов и цвета
sources = [m['source'] for m in metadatas]
unique_sources = list(set(sources))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
source_to_color = {src: colors[i] for i, src in enumerate(unique_sources)}

# Группируем индексы чанков по файлам, сортируем по chunk_id
indices_by_source = {}
for src in unique_sources:
    idx_list = [i for i, m in enumerate(metadatas) if m['source'] == src]
    # Сортируем по chunk_id, чтобы порядок был правильный
    idx_list.sort(key=lambda i: metadatas[i]['chunk_id'])
    indices_by_source[src] = idx_list

# Вычисляем центроиды для каждого файла
centroids = {}
for src in unique_sources:
    indices = indices_by_source[src]
    centroids[src] = np.mean(embeddings[indices], axis=0)

# 2D PCA проекция с линиями между чанками и от центроида
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(14, 10))

# Рисуем все точки и подписываем их
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=color, alpha=0.7, s=30)
    plt.annotate(f"Чанк {i+1}", (emb_2d[i, 0], emb_2d[i, 1]),
                 fontsize=5, alpha=0.5, ha='center', va='center')

# Для каждого файла соединяем его чанки линиями в порядке chunk_id
for src in unique_sources:
    indices = indices_by_source[src]
    if len(indices) > 1:
        # Берём координаты точек в 2D
        points = np.array([emb_2d[i] for i in indices])
        # Рисуем ломаную линию
        plt.plot(points[:, 0], points[:, 1], color=source_to_color[src],
                 linewidth=1.5, alpha=0.7, label=f"Траектория {src}")

# Проецируем центроиды и рисуем их звёздочками
centroids_2d = {}
for src in unique_sources:
    centroid_vec = centroids[src].reshape(1, -1)
    centroid_2d = pca.transform(centroid_vec)[0]
    centroids_2d[src] = centroid_2d
    plt.scatter(centroid_2d[0], centroid_2d[1],
                marker='*', s=300, color=source_to_color[src], edgecolors='black', linewidth=1)
    plt.annotate(f"Центроид {src}", (centroid_2d[0], centroid_2d[1]),
                 fontsize=9, fontweight='bold', ha='center', va='bottom')

# Соединяем каждую точку с центроидом своего файла пунктиром
for i in range(len(docs)):
    src = metadatas[i]['source']
    centroid_xy = centroids_2d[src]
    plt.plot([emb_2d[i, 0], centroid_xy[0]], [emb_2d[i, 1], centroid_xy[1]],
             color=source_to_color[src], linestyle=':', linewidth=0.8, alpha=0.3)

# Добавляем стрелки главных компонент (как раньше)
for i in range(2):
    scale = 2.0
    vec = pca.components_[i] * np.sqrt(pca.explained_variance_[i]) * scale
    mean_2d = np.mean(emb_2d, axis=0)
    plt.arrow(mean_2d[0], mean_2d[1], vec[0], vec[1],
              head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.8)
    plt.text(mean_2d[0] + vec[0]*1.1, mean_2d[1] + vec[1]*1.1,
             f'PC{i+1}', color='red', fontsize=12, fontweight='bold')
plt.title("PCA 2D с линиями траекторий чанков и связями с центроидами")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle='--', alpha=0.3)

# Легенда: только файлы и центроиды, без дублирования траекторий
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=source_to_color[src], label=src)
                   for src in unique_sources]
legend_elements.append(plt.Line2D([0], [0], marker='*', color='w',
                                  markerfacecolor='black', markersize=10, label='Центроиды'))
plt.legend(handles=legend_elements, loc='best', fontsize=8)
plt.tight_layout()
plt.show()

# 3D проекция (PCA или t-SNE) с линиями
if len(embeddings) > 2:
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    emb_3d = tsne.fit_transform(embeddings)
    method = "t-SNE"
    # Для t-SNE центроиды не показываем, но траектории покажем
    use_centroids = False
else:
    pca3 = PCA(n_components=3)
    emb_3d = pca3.fit_transform(embeddings)
    method = "PCA"
    use_centroids = True
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем точки
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    ax.scatter(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
               color=color, alpha=0.6, s=25)
    ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
            f"Чанк {i+1}", fontsize=4, alpha=0.3, ha='center', va='center')

# Рисуем траектории для каждого файла в 3D
for src in unique_sources:
    indices = indices_by_source[src]
    if len(indices) > 1:
        points = np.array([emb_3d[i] for i in indices])
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
                color=source_to_color[src], linewidth=1.5, alpha=0.7)

# Если метод PCA, показываем центроиды и связи с ними
if use_centroids:
    centroids_3d = {}
    for src in unique_sources:
        centroid_vec = centroids[src].reshape(1, -1)
        centroid_3d = pca3.transform(centroid_vec)[0]
        centroids_3d[src] = centroid_3d
        ax.scatter(centroid_3d[0], centroid_3d[1], centroid_3d[2],
                   marker='*', s=300, color=source_to_color[src], edgecolors='black', linewidth=1)
        ax.text(centroid_3d[0], centroid_3d[1], centroid_3d[2],
                f"Центроид {src}", fontsize=8, fontweight='bold')

        # Соединяем каждую точку с центроидом пунктиром
        for i in indices_by_source[src]:
            ax.plot([emb_3d[i, 0], centroid_3d[0]],
                    [emb_3d[i, 1], centroid_3d[1]],
                    [emb_3d[i, 2], centroid_3d[2]],
                    color=source_to_color[src], linestyle=':', linewidth=0.8, alpha=0.3)
else:
    ax.text2D(0.05, 0.95, "Центроиды не показаны для t-SNE", transform=ax.transAxes, fontsize=10, color='gray')
ax.set_title(f"{method} 3D проекция с траекториями чанков")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Легенда
legend_elements_3d = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=source_to_color[src],
                                 label=src, markersize=8)
                      for src in unique_sources]
if use_centroids:
    legend_elements_3d.append(plt.Line2D([0], [0], marker='*', color='w',
                                         markerfacecolor='black', markersize=10, label='Центроиды'))
ax.legend(handles=legend_elements_3d, loc='best', fontsize=8)
plt.tight_layout()
plt.show()