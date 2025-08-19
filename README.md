# Face Verify
facecheck from web-app

📦 Структура проекта
```commandline

mlface_verify/
├── manage.py
├── requirements.txt
├── README.md
├── .env.example
├── weights/                      # сюда кладём .pt и .onnx
├── mlface_verify/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   ├── wsgi.py
│   ├── logging_config.py
│   ├── logging_handlers.py
│   ├── decorators.py
│   ├── filters.py
├── verification/
│   ├── __init__.py
│   ├── apps.py
│   ├── urls.py
│   ├── forms.py
│   ├── views.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_io.py
│   │   ├── yolo_face_detector.py
│   │   ├── face_embedder.py
│   │   ├── matcher.py
│   ├── templates/verification/index.html
│   ├── static/verification/main.css
│   ├── static/verification/main.js
│   ├── tests/
│       ├── test_services.py
│       ├── test_views.py
│       └── data/ (пустая папка под тест-данные)
└── scripts/
    └── download_weights.py
```

# ML Face Verify (Django)

Тестовое приложение для верификации пользователя по фото документа (паспорт/права) и селфи.

## Возможности
- Загрузка фото документа
- Снятие селфи с веб-камеры (или загрузка из файла)
- Детекция лиц на базе **YOLOv11**
- Сопоставление лиц (ArcFace-совместимые эмбеддинги, **ONNX Runtime** GPU/CPU)
- MLflow-логирование метрик/артефактов (опционально)
- Профессиональное логирование (JSON-формат, ротация по времени/размеру, раздельные файлы ошибок/общих логов)

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
mkdir -p weights media/crops
```

🧠 Алгоритм верификации (коротко)

Пользователь загружает изображение документа (паспорт, права и т.п.) и делает селфи через камеру.

Бэкенд с помощью YOLO v11 (веса под детекцию лица в документе) находит портретную область на документе и вырезает её.

Для кропа из документа и для селфи извлекаем эмбеддинги ArcFace (InsightFace), нормализуем и считаем cosine distance.

Сравниваем с порогом (по умолчанию cosine_distance ≤ 0.35) и возвращаем: «Верификация пройдена / не пройдена» + метрики.

Если YOLO не нашёл область, срабатывает fallback: повторная детекция лиц InsightFace на полном изображении (мягкая деградация).


🔧 Инструменты

Django 4.2 — быстрый старт, класс‑бейзед вьюхи, формы, статика, шаблоны.

YOLO v11 (Ultralytics) — детекция портрета на документе (веса указываются через YOLO_WEIGHTS).

InsightFace + ONNX Runtime — эмбеддинги ArcFace, точное сравнение лиц.

OpenCV, Pillow, NumPy — работа с изображениями.

Профессиональный логгер — JSON‑адаптер, ротация по времени/размеру, отдельные файлы для ошибок и общего лога, HTTP‑коллектор, декораторы для автологирования вызовов.

MLflow (опционально) — метрики/артефакты/эксперименты.

Kubeflow/KServe (опционально) — продакшн‑деплой inference‑сервиса отдельно от Django: масштабирование, A/B, автоскейл, трейсинг.



🚀 Быстрый старт

```commandline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Инициализация Django
python manage.py migrate
python manage.py runserver
```