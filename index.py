import os
import uuid
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
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

def split_into_lines(text: str):
    """Разбиваем по строкам, но не теряем переносы как смысловые границы."""
    # splitlines() корректно обрабатывает разные переносы строк
    return [line.strip() for line in text.splitlines() if line.strip()]

def is_numbered_line(line: str) -> bool:
    """Проверяем, начинается ли строка с нумерации (1., 1.1., A., и т.п.)"""
    import re
    return bool(re.match(r'^\s*\d+(\.\d+)*\.?\s+', line)) or bool(re.match(r'^\s*[A-Za-z]\.\s+', line))
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
    lines = split_into_lines(text)
    if not lines:
        continue

    # Собираем чанки из строк, стараясь не резать пронумерованные пункты
    current_chunk_lines = []
    current_len = 0
    chunk_id = 0
    for line in lines:
        line_len = len(line)

        # Если строка — это пронумерованный пункт, и текущий чанк уже почти полон,
        # то лучше начать новый чанк, чтобы не резать пункт посередине.
        if is_numbered_line(line) and current_len > 0 and current_len + line_len > chunk_size:
            # Сохраняем накопленный чанк
            chunk = "\n".join(current_chunk_lines).strip()
            if len(chunk) > 5:
                docs.append(chunk)
                metadatas.append({"source": filename, "chunk_id": chunk_id, "text": chunk})
                ids.append(str(uuid.uuid4()))
                chunk_id += 1
            # Начинаем новый чанк с этой строки
            current_chunk_lines = [line]
            current_len = line_len
        else:
            # Просто добавляем строку в текущий чанк
            current_chunk_lines.append(line)
            current_len += line_len

            # Если чанк превысил лимит, закрываем его
            if current_len >= chunk_size and len(current_chunk_lines) > 1:
                chunk = "\n".join(current_chunk_lines).strip()
                if len(chunk) > 5:
                    docs.append(chunk)
                    metadatas.append({"source": filename, "chunk_id": chunk_id, "text": chunk})
                    ids.append(str(uuid.uuid4()))
                    chunk_id += 1
                # Оставляем последнюю строку как начало следующего чанка
                current_chunk_lines = [current_chunk_lines[-1]]
                current_len = len(current_chunk_lines[0])

    # Последний чанк
    if current_chunk_lines:
        chunk = "\n".join(current_chunk_lines).strip()
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