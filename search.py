import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Локальный кэш для офлайн-работы
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')

# Инициализация модели
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

# Подключение к серверу Qdrant
client = QdrantClient(host="localhost", port=6333)

"""
Ищет top_k наиболее похожих документов на запрос через Qdrant.
"""
def search(query: str, top_k: int = 5):
    vec = model.encode([query]).tolist()
    results = client.query_points(
        collection_name="my_docs",
        query=vec,
        limit=top_k,
        with_payload=True
    )
    return results.points
if __name__ == "__main__":
    print("\n=== Семантический поиск ===")

    # Запрос ввода напрямую от пользователя
    query = input("Введите ваш запрос: ").strip()

    # Защита от пустого ввода
    if not query:
        print("Запрос не может быть пустым. Используется тестовая фраза.")
        query = "пример запроса для поиска"
    print(f"\nЗапрос: {query}\n")
    results = search(query)
    if not results:
        print("Ничего не найдено.")
    else:
        print("Результаты:")
        for i, hit in enumerate(results, start=1):
            # Безопасное получение данных из payload
            source = hit.payload.get('source', 'Неизвестно')
            chunk_id = hit.payload.get('chunk_id', 'N/A')
            text_preview = hit.payload.get('text', '')[:150]
            print(f"{i}. {source} чанк {chunk_id}")
            print(f"   {text_preview}...")
            print(f"   score: {hit.score:.4f}\n")