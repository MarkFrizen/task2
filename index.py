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
MODEL_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_cache')
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
client = QdrantClient(host="localhost", port=6333)
collection_name = "my_docs"
vector_size = 384
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Параметры чанкинга
chunk_size = 800
overlap_size = 150
similarity_threshold = 0.7

# ---------- Функции ----------
def split_into_sentences(text: str) -> list:
    try:
        return sent_tokenize(text, language='russian')
    except:
        return sent_tokenize(text, language='english')

def is_heading(text: str) -> bool:
    """Проверяет, является ли строка заголовком (Markdown или нумерованным)."""
    stripped = text.strip()
    # Markdown-заголовки: начинаются с #, ##, ### и т.д.
    if stripped.startswith('#'):
        return True
    # Нумерованные списки: начинаются с цифры, точки и пробела
    if re.match(r'^\d+\.\s+', stripped):
        return True
    # Также можно проверить на маркеры списков типа "- ", "* "
    return False

def semantic_chunking(text: str, chunk_size: int, overlap_size: int, threshold: float) -> list:
    # Разбиваем на абзацы по двойным переносам
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []
    all_chunks = []
    for para in paragraphs:
        # Пропускаем пустые и разделители
        if para in ("", "---", "***", "___"):
            continue

        # Если абзац — заголовок, добавляем его как целый чанк
        if is_heading(para):
            # Добавляем заголовок, если он не слишком длинный (можно обрезать, но обычно короткий)
            if len(para) <= chunk_size:
                all_chunks.append(para)
            else:
                # Если заголовок длинный — разбиваем как обычный текст (редко)
                sentences = split_into_sentences(para)
                sentences = [s.strip() for s in sentences if s.strip()]
                if sentences:
                    # Просто объединяем все предложения в один чанк
                    all_chunks.append(" ".join(sentences))
            continue  # переходим к следующему абзацу

        # Обычный абзац — разбиваем на предложения
        sentences = split_into_sentences(para)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
        current_chunk = [sentences[0]]
        current_len = len(sentences[0])
        for i in range(1, len(sentences)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            if sim >= threshold and (current_len + len(sentences[i]) <= chunk_size):
                current_chunk.append(sentences[i])
                current_len += len(sentences[i])
            else:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    all_chunks.append(chunk_text)
                overlap_text = chunk_text[-overlap_size:] if len(chunk_text) > overlap_size else chunk_text
                current_chunk = [overlap_text + " " + sentences[i]] if overlap_text else [sentences[i]]
                current_len = len(current_chunk[0])
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                all_chunks.append(chunk_text)

    # Убираем дубликаты
    seen = set()
    unique_chunks = []
    for ch in all_chunks:
        if ch not in seen:
            seen.add(ch)
            unique_chunks.append(ch)
    return unique_chunks

# ---------- Чтение файлов ----------
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
    chunks = semantic_chunking(text, chunk_size, overlap_size, similarity_threshold)

    # Фильтруем пустые и чанки-разделители
    chunks = [c for c in chunks if c.strip() and c.strip() != "---"]
    if not chunks:
        print(f"Файл {filename} не дал чанков")
        continue
    for idx, chunk in enumerate(chunks):
        if any(phrase in chunk for phrase in STOP_PHRASES):
            continue
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

# ---------- Векторизация и загрузка ----------
embeddings = model.encode(docs, show_progress_bar=True, batch_size=32)
points = [
    PointStruct(id=ids[i], vector=embeddings[i].tolist(), payload=metadatas[i])
    for i in range(len(docs))
]
client.upsert(collection_name=collection_name, points=points)
print(f"Загружено {len(points)} векторов в коллекцию '{collection_name}'")

# ---------- Сохранение ----------
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
print("Данные сохранены в docs.txt (с разделителями), docs.pkl, metadatas.pkl и embeddings.npy")