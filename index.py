#!/usr/bin/env python3
"""
Скрипт для индексации текстовых документов в Qdrant.

Загружает текстовые файлы из папки documents, разбивает их на чанки,
кодировает каждый чанк в вектор с помощью SentenceTransformer и загружает
векторы в коллекцию Qdrant для семантического поиска. Также сохраняет
данные локально в формате pickle и numpy.
"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
import uuid
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Модель для кодирования текстов в эмбеддинги
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
# Подключение к серверу Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384

# Создание коллекции в Qdrant
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)
docs = []
metadatas = []
ids = []
doc_folder = "./documents"

# Проверка существования папки с документами
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена")
    exit(1)

# Загрузка и разбиение документов на чанки по предложениям
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        chunks = text.replace('\n', ' ').split('.')
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) > 5:
                docs.append(chunk)
                metadatas.append({"source": filename, "chunk_id": i, "text": chunk})
                ids.append(str(uuid.uuid4()))
if not docs:
    print("Не найдено ни одного чанка")
    exit(1)
print(f"Сгенерировано {len(docs)} чанков")

# Кодирование чанков в векторные представления
embeddings = model.encode(docs, show_progress_bar=True)

# Формирование точек для загрузки в Qdrant
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))]

# Загрузка векторов в Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# Сохранение данных локально для последующего использования
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")