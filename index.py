import os
import uuid
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab', quiet=True)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------- Настройки модели и базы данных ----------
# Папка для кэша модели, чтобы не качать каждый раз
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')

# Загружаем лёгкую модель для получения эмбеддингов
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

# Подключаемся к локально запущенному Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384  # размерность векторов у all-MiniLM-L6-v2

# Пересоздаём коллекцию в Qdrant (старые данные удаляются)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Параметры семантического разбиения на чанки
chunk_size = 800              # максимальная длина чанка в символах
overlap_size = 150            # сколько символов перекрывать между соседними чанками
similarity_threshold = 0.7    # порог схожести предложений для разрыва

# ---------- Функции для работы с текстом ----------

def split_into_sentences(text: str) -> list:
    """Разбивает текст на предложения, используя библиотеку NLTK.
       Поддерживает русский и английский языки."""
    try:
        return sent_tokenize(text, language='russian')
    except:
        return sent_tokenize(text, language='english')

def semantic_chunking(text: str, chunk_size: int, overlap_size: int, threshold: float) -> list:
    """
    Основная функция семантического чанкинга.
    Сначала текст делится на абзацы по пустым строкам.
    Внутри каждого абзаца предложения группируются по смыслу:
    если соседние предложения похожи (косинусное расстояние выше порога),
    они остаются в одном чанке, иначе — создаётся разрыв.
    Перекрытие обеспечивается добавлением последних символов предыдущего чанка.
    В конце удаляются дубликаты чанков.
    """
    # Разбиваем на абзацы
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []
    all_chunks = []
    for para in paragraphs:
        # Получаем предложения внутри абзаца
        sentences = split_into_sentences(para)
        if not sentences:
            continue

        # Считаем эмбеддинги для всех предложений абзаца
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)

        # Начинаем первый чанк с первого предложения
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            # Вычисляем косинусное сходство между текущим и предыдущим предложением
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )

            # Если сходство высокое и размер не превышен — добавляем предложение
            if sim >= threshold and (current_len + len(sentences[i]) <= chunk_size):
                current_chunk.append(sentences[i])
                current_len += len(sentences[i])
            else:
                # Иначе закрываем текущий чанк
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    all_chunks.append(chunk_text)

                # Формируем перекрытие: берём последние overlap_size символов из закрытого чанка
                overlap_text = chunk_text[-overlap_size:] if len(chunk_text) > overlap_size else chunk_text
                # Новый чанк начинается с перекрытия + текущее предложение
                current_chunk = [overlap_text + " " + sentences[i]] if overlap_text else [sentences[i]]
                current_len = len(current_chunk[0])

        # Добавляем последний чанк абзаца
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                all_chunks.append(chunk_text)

    # Убираем дубликаты, сохраняя порядок
    seen = set()
    unique_chunks = []
    for ch in all_chunks:
        if ch not in seen:
            seen.add(ch)
            unique_chunks.append(ch)
    return unique_chunks

# ---------- Чтение всех текстовых файлов из папки documents ----------
docs = []         # список текстов чанков (с разделителями)
metadatas = []    # метаданные для каждого чанка (источник, номер, текст)
ids = []          # уникальные идентификаторы для Qdrant
doc_folder = "./documents"
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена")
    exit(1)
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(doc_folder, filename)
    text = None
    # Пробуем разные кодировки, чтобы правильно прочитать файл
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        print(f"Пропущен {filename} — не удалось определить кодировку")
        continue

    # Применяем семантическое разбиение к содержимому файла
    chunks = semantic_chunking(text, chunk_size, overlap_size, similarity_threshold)
    if not chunks:
        print(f"Файл {filename} не дал чанков")
        continue

    # Сохраняем каждый чанк с метаданными
    for idx, chunk in enumerate(chunks):
        # Добавляем разделитель для визуального отделения чанков (не влияет на поиск)
        chunk_with_sep = chunk + "\n\n---\n\n"
        docs.append(chunk_with_sep)
        metadatas.append({
            "source": filename,
            "chunk_id": idx,
            "text": chunk
        })
        ids.append(str(uuid.uuid4()))
if not docs:
    print("Не найдено ни одного чанка")
    exit(1)
print(f"Сгенерировано {len(docs)} чанков")

# ---------- Получение эмбеддингов и загрузка в Qdrant ----------

# Кодируем все чанки в векторы с помощью той же модели
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)

# Формируем точки для загрузки в Qdrant
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))
]

# Загружаем векторы в коллекцию
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# ---------- Сохранение данных на диск для дальнейшего использования ----------
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")