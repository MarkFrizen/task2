import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

embeddings = np.load("embeddings.npy")
model = SentenceTransformer('all-MiniLM-L6-v2')

def show_cosine_distances(query: str):
    """Выводит косинусное сходство и расстояние для каждого документа."""
    q_emb = model.encode([query])
    similarities = cosine_similarity(q_emb, embeddings)[0]
    distances = 1 - similarities

    print(f"\nЗапрос: '{query}'\n")
    for i, (doc, sim, dist) in enumerate(zip(docs, similarities, distances)):
        print(f"Документ {i}:")
        print(f"  Текст: {doc[:100]}...")
        print(f"  Косинусное сходство: {sim:.4f}")
        print(f"  Косинусное расстояние: {dist:.4f}\n")

if __name__ == "__main__":
    test_query = "машинное обучение"
    show_cosine_distances(test_query)