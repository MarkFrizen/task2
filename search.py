import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# --- ИСПРАВЛЕНИЕ: Объявляем список предустановленных запросов ---
PRESET_QUERIES = [
    "Как работает семантический поиск?",
    "Что такое векторные представления текста?",
    "Как настроить Qdrant для поиска документов?",
    "Примеры использования Sentence Transformers",
    "Сравнение полнотекстового и семантического поиска"
]
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
    print("Введите запрос:\n")

    # Теперь эта строка не вызовет ошибку, так как PRESET_QUERIES определен
    for i, q in enumerate(PRESET_QUERIES, start=1):
        print(f"  {i}. {q}")
    print("  0. Ввести свой запрос")
    choice = input("\nВаш выбор: ").strip()
    if choice == "0":
        query = input("Введите запрос: ").strip()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(PRESET_QUERIES):
                query = PRESET_QUERIES[idx]
            else:
                print("Неверный номер, используется первый запрос по умолчанию.")
                query = PRESET_QUERIES
        except ValueError:
            print("Неверный ввод, используется первый запрос по умолчанию.")
            query = PRESET_QUERIES
    print(f"\nЗапрос: {query}\n")
    results = search(query)
    if not results:
        print("Ничего не найдено.")
    else:
        print("Результаты:")
        for i, hit in enumerate(results, start=1):
            # Добавлена проверка на наличие ключей в payload для избежания ошибок
            source = hit.payload.get('source', 'Неизвестно')
            chunk_id = hit.payload.get('chunk_id', 'N/A')
            text_preview = hit.payload.get('text', '')[:150]
            print(f"{i}. {source} чанк {chunk_id}")
            print(f"   {text_preview}...")
            print(f"   score: {hit.score:.4f}\n")