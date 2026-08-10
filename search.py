import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Локальный кэш для офлайн-работы
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

# Подключение к серверу Qdrant
client = QdrantClient(host="localhost", port=6333)

# Инициализация модели и клиента Qdrant
def search(query: str, top_k: int = 10):
    vec = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name="my_docs",
        query=vec,
        limit=top_k,
        with_payload=True
    )
    return results.points
if __name__ == "__main__":
    print("\n=== Семантический поиск ===\n")
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
                print(f"{i}. {hit.payload['source']} чанк {hit.payload['chunk_id']}")
                print(f"   {hit.payload['text'][:150]}...")
                print(f"   score: {hit.score:.4f}\n")