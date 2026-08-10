import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка сохранённых документов и их эмбеддингов
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
embeddings = np.load("embeddings.npy")
# Локальный кэш для офлайн-работы
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
# Вычисление косинусного сходства между запросом и документами
"""
Вычисляет и выводит косинусное сходство и расстояние между запросом и каждым документом.
"""
def show_cosine_distances(query: str):
    q_emb = model.encode([query])
    similarities = cosine_similarity(q_emb, embeddings)[0]
    distances = 1 - similarities
    print(f"\nЗапрос: '{query}'\n")
    for i, (doc, sim, dist) in enumerate(zip(docs, similarities, distances)):
        print(f"Документ {i}:")
        print(f"  Текст: {doc[:100]}...")
        print(f"  Сходство: {sim:.4f}")
        print(f"  Расстояние: {dist:.4f}\n")
if __name__ == "__main__":
    test_query = "машинное обучение"
    show_cosine_distances(test_query)