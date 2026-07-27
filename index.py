import os
import uuid
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Инициализация эмбеддинг-модели
# -----------------------------------------------------------------------------
# all-MiniLM-L6-v2: Лёгкая модель, оптимальная для семантического
# поиска. Предобучена на огромных корпусах текстов.
model = SentenceTransformer('all-MiniLM-L6-v2')
# -----------------------------------------------------------------------------

# Подключение к Qdrant
# -----------------------------------------------------------------------------
# Qdrant — векторная база данных, специализирующаяся на поиске по аналогии.
# Работает как сервер на localhost:6333
client = QdrantClient(host="localhost", port=6333)
# -----------------------------------------------------------------------------

# Настройки коллекции
# -----------------------------------------------------------------------------
collection_name = "my_docs"
vector_size = 384
# -----------------------------------------------------------------------------

# Пересоздание коллекции
# -----------------------------------------------------------------------------
# Distance.COSINE: Используется косинусное расстояние для сравнения векторов.
# Векторный размер 384 соответствует выходу модели SentenceTransformer.
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)
# -----------------------------------------------------------------------------

# Сбор данных из файлов
# -----------------------------------------------------------------------------
# Списки для накопления данных перед загрузкой в Qdrant
docs = []
metadatas = []
ids = []
doc_folder = "./documents"
# -----------------------------------------------------------------------------

# Проверка существования папки с документами
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена. Создайте её и положите .txt файлы.")
    exit(1)

# Обработка файлов
# -----------------------------------------------------------------------------
# Для каждого .txt файла:
#   1. Читаем содержимое
#   2. Убираем переносы строк
#   3. Разбиваем по точкам
#   4. Для каждого чанка сохраняем текст, метаданные и UUID
# -----------------------------------------------------------------------------
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        # Разбиваем по предложениям
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

# Проверка на пустую коллекцию
if not docs:
    print("Не найдено ни одного чанка. Проверьте содержимое папки ./documents.")
    exit(1)
print(f"Сгенерировано {len(docs)} чанков")

# Генерация эмбеддингов
# -----------------------------------------------------------------------------
# Преобразуем все текстовые чанки в векторы
embeddings = model.encode(docs, show_progress_bar=True)
# -----------------------------------------------------------------------------

# Подготовка точек для загрузки в Qdrant
# -----------------------------------------------------------------------------
points = [
    PointStruct(
        id=ids[i],
        vector=embeddings[i].tolist(),
        payload=metadatas[i]
    )
    for i in range(len(docs))
]
# -----------------------------------------------------------------------------

# Загрузка точек в Qdrant
# -----------------------------------------------------------------------------
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")
# -----------------------------------------------------------------------------

# Сохранение данных для последующего использования
# -----------------------------------------------------------------------------
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")