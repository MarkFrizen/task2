# Импорт необходимых библиотек
import os
import re
import uuid
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi

# Скачиваем данные для разбиения на предложения
nltk.download('punkt_tab', quiet=True)

# Отключаем интернет-запросы для работы офлайн
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Настройки модели и базы данных
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)

client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384

# Удаляем старую коллекцию и создаём новую
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Параметры чанкинга
chunk_size = 800
overlap_size = 0          # перекрытие отключено, чтобы не было дублей
similarity_threshold = 0.7

# Функция очистки текста: убирает только кавычки, запятые в конце и ссылки
def clean_text(text: str) -> str:
    # Удаляем ссылки типа [reference:цифры]
    text = re.sub(r'\[reference:\d+\]', '', text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Убираем начальные и конечные кавычки (всех видов)
        if line.startswith(('"', "'", '«', '“', '„')) and line.endswith(('"', "'", '»', '”', '“')):
            line = line[1:-1]
        # Убираем запятую в конце строки
        if line.endswith(','):
            line = line[:-1]
        # Убираем точку с запятой в конце
        if line.endswith(';'):
            line = line[:-1]
        line = line.strip()
        if line:  # сохраняем все непустые строки, включая ---, *** и т.д.
            cleaned_lines.append(line)
    # Собираем обратно с двойными переносами между строками (сохраняем структуру)
    return '\n\n'.join(cleaned_lines)

# Разбивает текст на предложения через NLTK
def split_into_sentences(text: str) -> list:
    try:
        return sent_tokenize(text, language='russian')
    except:
        return sent_tokenize(text, language='english')

# Проверяет, является ли строка заголовком
def is_heading(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith('#'):
        return True
    if re.match(r'^\d+\.\s+', stripped):
        return True
    return False

# Удаляет повторяющиеся предложения внутри списка (в пределах одного абзаца)
def deduplicate_sentences(sentences: list) -> list:
    seen = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

# Основная функция семантического чанкинга
def semantic_chunking(text: str, chunk_size: int, threshold: float) -> list:
    # Разбиваем на абзацы по двойным переносам строк
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []
    all_chunks = []

    for para in paragraphs:
        # Если абзац — заголовок, добавляем его целиком
        if is_heading(para):
            if len(para) <= chunk_size:
                all_chunks.append(para)
            else:
                sents = split_into_sentences(para)
                sents = [s.strip() for s in sents if s.strip()]
                if sents:
                    all_chunks.append(" ".join(sents))
            continue

        # Обычный абзац: разбиваем на предложения и удаляем дубли внутри абзаца
        sentences = split_into_sentences(para)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = deduplicate_sentences(sentences)
        if not sentences:
            continue

        # Получаем эмбеддинги для каждого предложения
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)

        # Собираем чанки без перекрытия
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            # Вычисляем косинусное сходство между соседними предложениями
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )

            # Если сходство высокое и размер не превышен, добавляем предложение
            if sim >= threshold and (current_len + len(sentences[i]) <= chunk_size):
                current_chunk.append(sentences[i])
                current_len += len(sentences[i])
            else:
                # Иначе закрываем чанк и начинаем новый с текущего предложения
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    all_chunks.append(chunk_text)
                current_chunk = [sentences[i]]
                current_len = len(sentences[i])

        # Добавляем последний чанк
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                all_chunks.append(chunk_text)

    # Глобальная дедупликация чанков (точные дубликаты)
    seen = set()
    unique_chunks = []
    for ch in all_chunks:
        if ch not in seen:
            seen.add(ch)
            unique_chunks.append(ch)
    return unique_chunks

# Чтение всех текстовых файлов из папки documents
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

    # Пробуем разные кодировки для чтения файла
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

    # Очищаем текст от лишних символов (кавычки, запятые, ссылки)
    text = clean_text(text)

    # Применяем семантический чанкинг
    chunks = semantic_chunking(text, chunk_size, similarity_threshold)
    chunks = [c for c in chunks if c.strip()]  # убираем только совсем пустые чанки
    if not chunks:
        print(f"Файл {filename} не дал чанков")
        continue

    # Сохраняем все чанки без исключений
    for idx, chunk in enumerate(chunks):
        docs.append(chunk)
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

# Получение эмбеддингов и загрузка в Qdrant
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))
]
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# Построение BM25 индекса для текстового поиска
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)
with open("tokenized_docs.pkl", "wb") as f:
    pickle.dump(tokenized_docs, f)
print(f"Создан BM25 индекс по {len(docs)} чанкам")

# Сохранение данных на диск
with open("docs.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(docs):
        if i > 0:
            f.write("\n---\n\n")
        f.write(chunk)
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
with open("metadatas.pkl", "wb") as f:
    pickle.dump(metadatas, f)
np.save("embeddings.npy", embeddings)
print("Данные сохранены в docs.txt, docs.pkl, metadatas.pkl и embeddings.npy")