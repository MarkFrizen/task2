import os
import uuid
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Модель для кодирования текстов
model = SentenceTransformer('all-MiniLM-L6-v2')
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

# Проверка существования папки
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена")
    exit(1)

# Загрузка и разбиение документов на чанки
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

# Кодирование в векторы
embeddings = model.encode(docs, show_progress_bar=True)

# Формирование точек для загрузки
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))]

# Загрузка в Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# Сохранение данных локально
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")