import os
import uuid
# Импорт библиотеки для работы с предобученными моделями трансформеров
# Используется модель 'all-MiniLM-L6-v2' для генерации 384-мерных векторных эмбеддингов
from sentence_transformers import SentenceTransformer
# Импорт клиента для взаимодействия с векторной базой данных Qdrant
from qdrant_client import QdrantClient
# Импорт типов данных для конфигурации векторов и точек в Qdrant
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Инициализация модели для генерации эмбеддингов
# Модель 'all-MiniLM-L6-v2' — лёгкая и быстрая, подходит для семантического поиска
model = SentenceTransformer('all-MiniLM-L6-v2')
# Подключение к локальному экземпляру Qdrant, который должен быть запущен на порту 6333
client = QdrantClient(host="localhost", port=6333)

# Настройки коллекции
# Имя коллекции для хранения векторизованных документов
collection_name = "my_docs"
# Размерность векторов: модель 'all-MiniLM-L6-v2' генерирует 384-мерные векторы
vector_size = 384
# Создание (или пересоздание) коллекции в Qdrant
# - vectors_config: конфигурация векторного пространства
# - size: размерность векторов (384)
# - distance: метрика расстояния (косинусное сходство для семантического поиска)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Списки для хранения данных перед вставкой в базу
# docs — текстовые чанки, metadatas — метаданные, ids — уникальные идентификаторы
docs = []
metadatas = []
ids = []
# Путь к папке с документами
doc_folder = "./documents"

# Проверка существования папки с документами
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена. Создайте её и положите .txt файлы.")
    exit(1)

# Обход всех файлов в папке documents
for filename in os.listdir(doc_folder):
    # Обрабатываем только файлы с расширением .txt
    if not filename.endswith(".txt"):
        continue
    # Открытие и чтение текстового файла с кодировкой UTF-8
    with open(os.path.join(doc_folder, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        # Предварительная обработка текста:
        # - замена символов новой строки на пробелы
        # - разбиение по точкам для получения чанков
        chunks = text.replace('\n', ' ').split('.')
        # Обработка каждого чанка
        for i, chunk in enumerate(chunks):
            # Очистка чанка от лишних пробелов
            chunk = chunk.strip()
            # Добавляем только непустые и достаточно длинные чанки (длиной > 5 символов)
            # Это предотвращает добавление коротких обрывков текста
            if len(chunk) > 5:
                docs.append(chunk)
                # Формирование метаданных для каждого чанка
                metadatas.append({
                    "source": filename,        # Имя исходного файла
                    "chunk_id": i,             # Порядковый номер чанка в файле
                    "text": chunk              # Сам текст чанка (для удобства поиска)
                })
                # Генерация уникального UUID для каждой точки в базе данных
                ids.append(str(uuid.uuid4()))

# Проверка: если не удалось создать ни одного чанка, завершаем работу
if not docs:
    print("Не найдено ни одного чанка. Проверьте содержимое папки ./documents.")
    exit(1)

# Вывод количества созданных чанков
print(f"Сгенерировано {len(docs)} чанков")
# Генерация векторных эмбеддингов для всех чанков с отображением прогресса
embeddings = model.encode(docs, show_progress_bar=True)

# Формирование списка точек для вставки в Qdrant
# Каждая точка состоит из:
# - id: уникальный идентификатор
# - vector: векторное представление чанка
# - payload: метаданные (источник, номер чанка, текст)
points = [
    PointStruct(
        id=ids[i],
        vector=embeddings[i].tolist(),
        payload=metadatas[i]
    )
    for i in range(len(docs))
]

# Выполнение пакетной вставки всех точек в коллекцию
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")