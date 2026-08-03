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
# Набор готовых запросов для демонстрации
PRESET_QUERIES = [
    "машинное обучение",
    "векторы",
    "поиск по смыслу",
    "эмбеддинги",
    "документ",
    "предложения"
]
"""
Ищет top_k наиболее похожих документов на запрос через Qdrant.
"""
def search(query: str, top_k: int = 5):
    vec = model.encode([query])[0].tolist()
    results = client.query_points(
        collection_name="my_docs",
        query=vec,
        limit=top_k,
        with_payload=True
    )
    return results.points
if __name__ == "__main__":
    print("\n=== Семантический поиск ===")
    print("Выберите запрос или 0 для ручного ввода:\n")
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
                print("Неверный номер")
                query = PRESET_QUERIES[0]
        except ValueError:
            print("Неверный ввод")
            query = PRESET_QUERIES[0]
    print(f"\nЗапрос: {query}\n")
    results = search(query)
    if not results:
        print("Ничего не найдено.")
    else:
        print("Результаты:")
        for i, hit in enumerate(results, start=1):
            print(f"{i}. {hit.payload['source']} чанк {hit.payload['chunk_id']}")
            print(f"   {hit.payload['text'][:150]}...")
            print(f"   score: {hit.score:.4f}\n")