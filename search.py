from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)

def search(query, top_k=5):
    vec = model.encode([query])[0].tolist()
    # Используем query_points (новый метод)
    results = client.query_points(
        collection_name="my_docs",
        query=vec,
        limit=top_k,
        with_payload=True
    )
    return results.points   # точки с payload и score

if __name__ == "__main__":
    q = input("Введите запрос: ")
    res = search(q)
    print("\nРезультаты:")
    for i, hit in enumerate(res):
        print(f"{i+1}. {hit.payload['source']} (чанк {hit.payload['chunk_id']})")
        print(f"   {hit.payload['text'][:150]}...")
        print(f"   score: {hit.score:.4f}\n")
