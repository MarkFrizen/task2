# Импортируем нужные библиотеки
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

# Загружаем заранее сохранённые эмбеддинги, тексты чанков и их метаданные
embeddings = np.load("embeddings.npy")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

# Определяем уникальные имена файлов, из которых взяты чанки
sources = [m['source'] for m in metadatas]
unique_sources = list(set(sources))

# Назначаем каждому файлу свой цвет из палитры tab10
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sources)))
source_to_color = {src: colors[i] for i, src in enumerate(unique_sources)}

# Строим двумерную проекцию методом главных компонент
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# Создаём новый график для 2D визуализации
plt.figure(figsize=(14, 10))

# Проходим по всем чанкам и рисуем каждую точку с подписью
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1], color=color, alpha=0.7, s=30)
    plt.annotate(f"Чанк {i+1}", (emb_2d[i, 0], emb_2d[i, 1]),
                 fontsize=5, alpha=0.5, ha='center', va='center')

# Добавляем заголовок, подписи осей и сетку
plt.title("PCA 2D проекция все чанки подписаны цвет по файлу")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, linestyle='--', alpha=0.3)

# Рисуем легенду с названиями исходных файлов
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=source_to_color[src], label=src)
                   for src in unique_sources]
plt.legend(handles=legend_elements, loc='best', fontsize=8)

# Показываем график
plt.tight_layout()
plt.show()

# Переходим к трёхмерной визуализации
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

# Создаём трёхмерный график
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем каждую точку в 3D и подписываем её
for i in range(len(docs)):
    color = source_to_color[metadatas[i]['source']]
    ax.scatter(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
               color=color, alpha=0.7, s=30)
    ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2],
            f"Чанк {i+1}", fontsize=4, alpha=0.4, ha='center', va='center')

# Добавляем заголовок и подписи осей
ax.set_title(f"{method} 3D проекция все чанки подписаны")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Легенда для трёхмерного графика
from mpl_toolkits.mplot3d.art3d import Line3DCollection
legend_elements_3d = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=source_to_color[src],
                                 label=src, markersize=8)
                      for src in unique_sources]
ax.legend(handles=legend_elements_3d, loc='best', fontsize=8)

# Показываем второй график
plt.tight_layout()
plt.show()