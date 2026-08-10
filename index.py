import os
import uuid
import pickle
import numpy as np
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)
docs = []
metadatas = []
ids = []
doc_folder = "./documents"
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена")
    exit(1)

# --- НАСТРОЙКИ ЧАНКИНГА ---
chunk_size = 800          # целевая длина чанка в символах
overlap_size = 150        # перекрытие между соседними чанками

def split_into_blocks(text: str) -> list:
    """Разбиваем текст на блоки по пустым строкам (абзацы)."""
    blocks = text.split('\n\n')
    return [block.strip() for block in blocks if block.strip()]

def is_block_empty(block: str) -> bool:
    """Проверяем, является ли блок пустым или содержит только маркеры разметки."""
    import re
    cleaned = re.sub(r'[-*_]{3,}', '', block).strip()
    return len(cleaned) == 0
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
    blocks = split_into_blocks(text)
    if not blocks:
        continue

    # Собираем чанки из блоков, разбивая по пустым строкам
    current_chunk_blocks = []
    current_len = 0
    chunk_id = 0
    for block in blocks:
        if is_block_empty(block):
            continue
        block_len = len(block)

        # Добавляем блок в текущий чанк
        current_chunk_blocks.append(block)
        current_len += block_len

        # Если чанк превысил лимит, закрываем его
        if current_len >= chunk_size and len(current_chunk_blocks) > 1:
            chunk = "\n\n".join(current_chunk_blocks).strip()
            if len(chunk) > 5:
                docs.append(chunk)
                metadatas.append({"source": filename, "chunk_id": chunk_id, "text": chunk})
                ids.append(str(uuid.uuid4()))
                chunk_id += 1
            # Оставляем последний блок как начало следующего чанка (overlap)
            current_chunk_blocks = [current_chunk_blocks[-1]]
            current_len = len(current_chunk_blocks[0])

    # Последний чанк
    if current_chunk_blocks:
        chunk = "\n\n".join(current_chunk_blocks).strip()
        if len(chunk) > 5:
            docs.append(chunk)
            metadatas.append({"source": filename, "chunk_id": chunk_id, "text": chunk})
            ids.append(str(uuid.uuid4()))
if not docs:
    print("Не найдено ни одного чанка")
    exit(1)
print(f"Сгенерировано {len(docs)} чанков")
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))
]
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")