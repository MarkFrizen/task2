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

# Поиск по косинусному сходству — единый интерфейс search()
def search(query: str, top_k: int = 5):
    q_emb = model.encode([query])
    similarities = cosine_similarity(q_emb, embeddings)[0]
    distances = 1 - similarities
    # Сортировка по убыванию сходства
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "text": docs[idx],
            "metadata": metadatas[idx],
            "score": float(similarities[idx])
        })
    return results

with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

if __name__ == "__main__":
    print("\n=== Семантический поиск (косинусное сходство) ===\n")
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
