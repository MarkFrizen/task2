# Семантический поиск по документам

Поисковая система с семантическим разбиением на чанки, векторным представлением и хранением в Qdrant.

## Возможности

- Семантическое разбиение текста на чанки с учётом схожести предложений
- Векторное представление через `sentence-transformers` (all-MiniLM-L6-v2)
- Поиск в векторной БД Qdrant (локальный режим)
- Альтернативный поиск через FAISS
- Визуализация встраиваний (PCA / t-SNE)

## Требования

- Python 3.12+
- Docker и Docker Compose (для Qdrant)

## Установка

```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Скачивание токенов NLTK
python3 -c "import nltk; nltk.download('punkt_tab')"
```

## Запуск Qdrant

```bash
docker compose up -d
```

## Использование

```bash
source venv/bin/activate

# Индексация документов (index.py)
python index.py

# Поиск
python search.py "запрос"

# Визуализация
python visualize.py

# Демонстрация косинусного сходства
python cosine_demo.py

# Поиск через FAISS
python faiss_search.py
```

## Структура файлов

| Файл | Назначение |
|---|---|
| `index.py` | Чанкирование, эмбеддинг, загрузка в Qdrant |
| `search.py` | Поиск по запросу в Qdrant |
| `faiss_search.py` | Поиск через FAISS |
| `visualize.py` | Визуализация эмбеддингов |
| `cosine_demo.py` | Демонстрация косинусного сходства |
| `documents/` | Входные текстовые файлы |

## Зависимости

См. `requirements.txt`. Фиксированные версии для воспроизводимости.
