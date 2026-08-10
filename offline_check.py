#!/usr/bin/env python3
"""
Проверка готовности проекта к полноценной офлайн-работе.
Проверяет:
  - наличие закэшированной модели и всех её файлов
  - наличие всех установленных pip-пакетов
  - способность загрузить модель без подключения к интернету
  - наличие локальных данных (docs.pkl, embeddings.npy, Qdrant storage)
"""

import os
import sys
import json
import importlib

# ═══════════════════════════════════════════════════════════
# ВАЖНО: все переменные офлайн-режима должны быть заданы
# ДО любых импортов sentence_transformers / transformers / huggingface_hub
# ═══════════════════════════════════════════════════════════
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE = os.path.join(SCRIPT_DIR, 'models_cache')
QDRANT_STORAGE = os.path.join(SCRIPT_DIR, 'qdrant_storage')

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"

errors = 0
warnings = 0

def check_pass(msg: str):
    print(f"  {PASS} {msg}")

def check_fail(msg: str):
    global errors
    errors += 1
    print(f"  {FAIL} {msg}")

def check_warn(msg: str):
    global warnings
    warnings += 1
    print(f"  {WARN} {msg}")


def check_model_cache():
    """Проверяет полноту кэша модели."""
    print("\n[1/5] Проверка кэша модели...")
    expected_files = [
        'config.json',
        'config_sentence_transformers.json',
        'model.safetensors',
        'modules.json',
        'sentence_bert_config.json',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'vocab.txt',
    ]

    # Ищем снэпшот
    snapshots_dir = os.path.join(MODEL_CACHE, 'models--sentence-transformers--all-MiniLM-L6-v2', 'snapshots')
    if not os.path.isdir(snapshots_dir):
        check_fail(f"Директория снэпшотов не найдена: {snapshots_dir}")
        return

    snapshot_dirs = [d for d in os.listdir(snapshots_dir)
                     if os.path.isdir(os.path.join(snapshots_dir, d))]
    if not snapshot_dirs:
        check_fail("Не найдено ни одного снэпшота модели")
        return

    snapshot_path = os.path.join(snapshots_dir, snapshot_dirs[0])
    print(f"  Найдено: {os.path.basename(snapshot_path)}/")

    for fname in expected_files:
        fpath = os.path.join(snapshot_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            check_pass(f"{fname} ({size:,} байт)")
        else:
            check_fail(f"Отсутствует: {fname}")


def check_pip_packages():
    """Проверяет, что все пакеты из requirements.txt установлены."""
    print("\n[2/5] Проверка установленных pip-пакетов...")
    req_file = os.path.join(SCRIPT_DIR, 'requirements.txt')
    if not os.path.isfile(req_file):
        check_fail("requirements.txt не найден")
        return

    required = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg = line.split('==')[0].split('[')[0].strip()
                if pkg:
                    required.append(pkg)

    # Маппинг pip-пакет → имя модуля для importlib
    PACKAGE_TO_MODULE = {
        'faiss-cpu': 'faiss',
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
    }

    missing = []
    for pkg in required:
        module_name = PACKAGE_TO_MODULE.get(pkg, pkg.replace('-', '_'))
        try:
            importlib.import_module(module_name)
            check_pass(pkg)
        except ImportError:
            missing.append(pkg)
            check_fail(pkg)

    if missing:
        print(f"\n  Установите недостающие пакеты:")
        for m in missing:
            print(f"    pip install {m}")


def check_model_load():
    """Проверяет способность загрузить модель в офлайн-режиме."""
    print("\n[3/5] Проверка загрузки модели в офлайн-режиме...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE)
        # Тестовый прогон
        vec = model.encode("тестовый запрос")
        check_pass(f"Модель загружена, вектор размерности {len(vec)}")
    except Exception as e:
        check_fail(f"Ошибка загрузки модели: {e}")


def check_local_data():
    """Проверяет наличие локальных данных."""
    print("\n[4/5] Проверка локальных данных...")
    data_files = {
        'docs.pkl': 'Список чанков документов',
        'metadatas.pkl': 'Метаданные чанков',
        'embeddings.npy': 'Эмбеддинги документов',
    }
    for fname, desc in data_files.items():
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            check_pass(f"{fname} — {desc} ({size:,} байт)")
        else:
            check_warn(f"{fname} — {desc} (файл отсутствует, запускайте index.py)")

    # Qdrant storage
    if os.path.isdir(QDRANT_STORAGE):
        check_pass(f"Qdrant storage ({QDRANT_STORAGE})")
    else:
        check_warn(f"Qdrant storage не найден (запустите docker-compose up)")


def check_scripts():
    """Проверяет наличие и синтаксис всех скриптов."""
    print("\n[5/5] Проверка скриптов...")
    scripts = [
        'index.py',
        'search.py',
        'cosine_demo.py',
        'faiss_search.py',
        'visualize.py',
        'offline_check.py',
    ]
    for script in scripts:
        fpath = os.path.join(SCRIPT_DIR, script)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    compile(f.read(), script, 'exec')
                check_pass(script)
            except SyntaxError as e:
                check_fail(f"{script}: {e}")
        else:
            check_warn(f"{script} — файл отсутствует")


if __name__ == '__main__':
    print("=" * 60)
    print("  ОФЛАЙН-ПРОВЕРКА ПРОЕКТА")
    print("=" * 60)

    check_model_cache()
    check_pip_packages()
    check_model_load()
    check_local_data()
    check_scripts()

    print("\n" + "=" * 60)
    if errors == 0 and warnings == 0:
        print(f"  {PASS} ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — проект готов к офлайн-работе")
    elif errors == 0:
        print(f"  {WARN} Есть предупреждения ({warnings}), но модель работает")
    else:
        print(f"  {FAIL} Ошибки: {errors}, Предупреждения: {warnings}")
        print("  Исправьте ошибки перед офлайн-использованием")
    print("=" * 60)

    sys.exit(1 if errors > 0 else 0)
