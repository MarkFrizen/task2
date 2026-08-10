import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Загрузка документов, метаданных и эмбеддингов
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)
with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)
embeddings = np.load("embeddings.npy")

# Нормализация векторов для скалярного произведения
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
dim = embeddings.shape[1]

# Создание FAISS-индекса с внутренним скалярным произведением
index = faiss.IndexFlatIP(dim)
index.add(embeddings_norm)

# Локальный кэш для офлайн-работы
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

# Поиск похожих документов через FAISS — единый интерфейс search()
def search(query: str, top_k: int = 10):
    q_emb = model.encode([query])
    q_emb_norm = q_emb / np.linalg.norm(q_emb)
    scores, indices = index.search(q_emb_norm, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "text": docs[idx],
            "metadata": metadatas[idx],
            "score": float(score)
        })
    return results

if __name__ == "__main__":
    print("\n=== Семантический поиск (FAISS) ===\n")
    while True:
        query = input("Введите запрос (или 'exit' для выхода): ").strip()
        if query.lower() in ("exit", "quit", "q"):
            print("Выход из программы.")
            break
        if not query:
            print("Запрос не может быть пустым. Попробуйте ещё раз.\n")
            continue
        print(f"\nЗапрос: {query}\n")
        results = search(query)
        if not results:
            print("Ничего не найдено.\n")
        else:
            print("Результаты:")
            for i, hit in enumerate(results, start=1):
                print(f"{i}. {hit['metadata']['source']} чанк {hit['metadata']['chunk_id']}")
                print(f"   {hit['text'][:150]}...")
                print(f"   score: {hit['score']:.4f}\n")
