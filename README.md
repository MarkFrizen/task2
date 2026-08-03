# Семантический поиск — Офлайн-режим

Все программы в этом проекте работают **полностью без интернета**.

## Что сделано для офлайн-работы

| Компонент | Статус |
|---|---|
| Модель `all-MiniLM-L6-v2` | Скачана в `models_cache/` |
| `faiss-cpu` | Установлен в venv |
| `sentence-transformers` | Установлен в venv |
| `qdrant/qdrant` Docker-образ | Сохранён локально |
| `scikit-learn`, `matplotlib`, `numpy` | Установлены в venv |

## Использование

### 1. Виртуальное окружение
```bash
source venv/bin/activate
```

### 2. Индексация документов
```bash
python3 index.py
```
Сканирует `./documents/`, создаёт векторы и загружает в Qdrant.

### 3. Поиск через Qdrant
```bash
docker compose up -d   # запустить Qdrant (один раз)
python3 search.py      # интерактивный поиск
```

### 4. Поиск через FAISS (без Qdrant)
```bash
python3 faiss_search.py
```

### 5. Косинусное сходство
```bash
python3 cosine_demo.py
```

### 6. Визуализация (нужен GUI/X11)
```bash
python3 visualize.py
```

## Офлайн-инструкция (перенос на другой ПК)

1. Скопировать **всю папку проекта** (включая `models_cache/`)
2. На новом ПК:
   ```bash
   source venv/bin/activate
   pip install faiss-cpu
   docker pull qdrant/qdrant   # только если образ не установлен
   ```
3. Все скрипты работают без интернета.

## Замечания

- `docker compose up -d` не требует интернета — образ уже локальный
- Модель весит ~87 MB, находится в `models_cache/`
- Warning про HF Hub можно игнорировать — модель уже загружена локально
