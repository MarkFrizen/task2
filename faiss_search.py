import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# FAISS: поиск по векторам
# -----------------------------------------------------------------------------

with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

with open("metadatas.pkl", "rb") as f:
    metadatas = pickle.load(f)

embeddings = np.load("embeddings.npy")
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_norm)

model = SentenceTransformer('all-MiniLM-L6-v2')

def search_faiss(query: str, top_k: int = 5):
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
    test_query = "машинное обучение"
    print(f"Поиск через FAISS по запросу: '{test_query}'\n")
    results = search_faiss(test_query)
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['metadata']['source']} чанк {res['metadata']['chunk_id']}")
        print(f"   {res['text'][:150]}...")
        print(f"   score: {res['score']:.4f}\n")
