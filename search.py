from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)

# Список предопределённых запросов (можно дополнить)
PRESET_QUERIES = [
    "машинное обучение",
    "векторы",
    "поиск по смыслу",
    "эмбеддинги",
    "документ",
    "предложения"
]

def search(query, top_k=5):
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
    print("Выберите запрос (введите номер) или 0 для ручного ввода:\n")
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
                print("Неверный номер, использую первый запрос.")
                query = PRESET_QUERIES[0]
        except ValueError:
            print("Неверный ввод, использую первый запрос.")
            query = PRESET_QUERIES[0]

    print(f"\nЗапрос: {query}\n")
    results = search(query)

    if not results:
        print("Ничего не найдено.")
    else:
        print("Результаты:")
        for i, hit in enumerate(results, start=1):
            print(f"{i}. {hit.payload['source']} (чанк {hit.payload['chunk_id']})")
            print(f"   {hit.payload['text'][:150]}...")
            print(f"   score: {hit.score:.4f}\n")
