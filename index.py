import os
import uuid
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Инициализация модели
model = SentenceTransformer('all-MiniLM-L6-v2')
# Подключение к Qdrant
client = QdrantClient(host="localhost", port=6333)

# Настройки коллекции
collection_name = "my_docs"
vector_size = 384

# Пересоздаём коллекцию (если существует, удалится)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Списки для хранения данных
docs = []
metadatas = []
ids = []
doc_folder = "./documents"
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена. Создайте её и положите .txt файлы.")
    exit(1)

# Обработка файлов
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        # Разбиваем по предложениям (можно заменить на более продвинутый сплиттер)
        chunks = text.replace('\n', ' ').split('.')
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if len(chunk) > 5:
                docs.append(chunk)
                metadatas.append({
                    "source": filename,
                    "chunk_id": i,
                    "text": chunk
                })
                ids.append(str(uuid.uuid4()))
if not docs:
    print("Не найдено ни одного чанка. Проверьте содержимое папки ./documents.")
    exit(1)
print(f"Сгенерировано {len(docs)} чанков")
embeddings = model.encode(docs, show_progress_bar=True)

# Подготовка точек для Qdrant
points = [
    PointStruct(
        id=ids[i],
        vector=embeddings[i].tolist(),
        payload=metadatas[i]
    )
    for i in range(len(docs))
]

# Загрузка в Qdrant
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# ---------- СОХРАНЕНИЕ ДАННЫХ ДЛЯ ДАЛЬНЕЙШЕГО ИСПОЛЬЗОВАНИЯ ----------
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")