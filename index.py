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

# ---------- Настройки ----------
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384

# Пересоздаём коллекцию
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Параметры семантического чанкинга
chunk_size = 800                # максимальная длина чанка в символах (мягкое ограничение)
overlap_size = 150              # перекрытие в символах (для сохранения контекста на границах)
similarity_threshold = 0.6      # порог косинусного сходства для разрыва (настраивайте)

# ---------- Функции для семантического разбиения ----------
def split_into_sentences(text: str) -> list:
    """
    Разбивает текст на предложения, используя простые правила.
    Учитывает точки, восклицательные и вопросительные знаки,
    а также заглавные буквы после них (для английского и русского).
    """
    import re
    # Шаблон: разделитель .!? + пробел или конец строки, после которого заглавная буква
    # Это не идеально, но достаточно для большинства текстов
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-Я])', text)
    # Если не сработало (нет заглавных после точки), разбиваем просто по .!? с пробелом
    if len(sentences) == 1:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    # Удаляем пустые и лишние пробелы
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def semantic_chunking(text: str, chunk_size: int, overlap_size: int, threshold: float) -> list:
    """
    Основная функция семантического разбиения текста на чанки.
    Возвращает список строк (чанков).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    # Получаем эмбеддинги для всех предложений за один вызов модели
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
    chunks = []

    # Начинаем с первого предложения
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    for i in range(1, len(sentences)):
        prev_emb = embeddings[i-1]
        curr_emb = embeddings[i]
        # Косинусное сходство (векторы уже нормализованы, можно использовать dot)
        similarity = np.dot(prev_emb, curr_emb) / (np.linalg.norm(prev_emb) * np.linalg.norm(curr_emb))

        # Решение: добавлять или разбивать
        if similarity >= threshold and (current_len + len(sentences[i]) <= chunk_size):
            # Смысл близок и размер позволяет – добавляем
            current_chunk.append(sentences[i])
            current_len += len(sentences[i])
        else:
            # Закрываем текущий чанк
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Формируем перекрытие: берём последние предложения из текущего чанка,
            # чтобы суммарная длина не превышала overlap_size
            overlap_sentences = []
            temp_len = 0
            for s in reversed(current_chunk):
                if temp_len + len(s) <= overlap_size:
                    overlap_sentences.insert(0, s)
                    temp_len += len(s)
                else:
                    # если одно предложение длиннее overlap_size, берём его целиком
                    if temp_len == 0:
                        overlap_sentences.insert(0, s)
                    break

            # Новый чанк начинается с перекрывающихся предложений + текущее
            current_chunk = overlap_sentences + [sentences[i]]
            current_len = sum(len(s) for s in current_chunk)

    # Добавляем последний чанк
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

# ---------- Чтение и обработка документов ----------
docs = []
metadatas = []
ids = []
doc_folder = "./documents"
if not os.path.exists(doc_folder):
    print(f"Папка {doc_folder} не найдена")
    exit(1)
for filename in os.listdir(doc_folder):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(doc_folder, filename)
    text = None

    # Пробуем разные кодировки
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        print(f"Пропущен файл {filename}: не удалось определить кодировку")
        continue

    # Применяем семантическое разбиение
    text_chunks = semantic_chunking(text, chunk_size, overlap_size, similarity_threshold)
    if not text_chunks:
        print(f"Файл {filename} не дал чанков (возможно, пустой)")
        continue

    # Добавляем каждый чанк в глобальные списки
    for idx, chunk in enumerate(text_chunks):
        # Добавляем разделитель для улучшения читаемости (необязательно)
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

# ---------- Векторизация и загрузка в Qdrant ----------
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))
]
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# ---------- Сохранение данных на диск ----------
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.pkl, metadatas.pkl и embeddings.npy")