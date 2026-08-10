import os
import uuid
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

# Скачиваем данные для токенизации (делается один раз)
nltk.download('punkt_tab', quiet=True)

# Отключаем обращения в интернет для офлайн-работы
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# ---------- Настройки модели и базы данных ----------
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384

# Пересоздаём коллекцию (старые данные удаляются)
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Параметры семантического чанкинга
chunk_size = 800              # максимальная длина чанка в символах
overlap_size = 150            # перекрытие в символах между соседними чанками
similarity_threshold = 0.7    # порог схожести предложений для разрыва

# ---------- Функции для работы с текстом ----------
def split_into_sentences(text: str) -> list:
    """Разбивает текст на предложения с помощью NLTK (русский/английский)."""
    try:
        return sent_tokenize(text, language='russian')
    except:
        return sent_tokenize(text, language='english')

def semantic_chunking(text: str, chunk_size: int, overlap_size: int, threshold: float) -> list:
    """
    Основная функция семантического чанкинга.
    Сначала текст делится на абзацы по пустым строкам (два переноса).
    Внутри каждого абзаца предложения группируются по смыслу:
    если косинусное сходство между соседними предложениями выше порога
    и размер не превышен, они остаются в одном чанке, иначе — разрыв.
    Перекрытие добавляется с помощью последних символов предыдущего чанка.
    В конце удаляются дубликаты.
    """
    # Разбиваем на абзацы по двум переносам строк (стандарт Markdown)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []

    all_chunks = []

    for para in paragraphs:
        # Получаем предложения внутри абзаца
        sentences = split_into_sentences(para)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue

        # Считаем эмбеддинги для всех предложений абзаца
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)

        # Начинаем первый чанк с первого предложения
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            # Косинусное сходство между предыдущим и текущим предложением
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )

            # Если сходство высокое и размер не превышен — добавляем
            if sim >= threshold and (current_len + len(sentences[i]) <= chunk_size):
                current_chunk.append(sentences[i])
                current_len += len(sentences[i])
            else:
                # Закрываем текущий чанк
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    all_chunks.append(chunk_text)

                # Перекрытие: последние `overlap_size` символов из закрытого чанка
                overlap_text = chunk_text[-overlap_size:] if len(chunk_text) > overlap_size else chunk_text
                # Новый чанк начинается с перекрытия + текущее предложение
                current_chunk = [overlap_text + " " + sentences[i]] if overlap_text else [sentences[i]]
                current_len = len(current_chunk[0])

        # Последний чанк абзаца
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
        print(f"Пропущен {filename} — не удалось определить кодировку")
        continue

    # Применяем семантическое разбиение (переносы строк сохраняются внутри абзацев)
    chunks = semantic_chunking(text, chunk_size, overlap_size, similarity_threshold)
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        print(f"Файл {filename} не дал чанков")
        continue

    # Сохраняем каждый чанк с метаданными
    for idx, chunk in enumerate(chunks):
        docs.append(chunk)   # теперь чанк содержит пробелы, но не сломанные переносы
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