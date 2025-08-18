# facecheck
facecheck from web-app

📦 Структура проекта
```commandline
facecheck/
├── manage.py
├── README.md
├── requirements.txt
├── mlapp/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── verifier/
    ├── __init__.py
    ├── apps.py
    ├── urls.py
    ├── models.py
    ├── forms.py
    ├── views.py
    ├── services/
    │   ├── __init__.py
    │   ├── face_verify.py
    │   ├── ml_tracking.py
    │   └── utils.py
    ├── templates/
    │   ├── base.html
    │   └── verifier/
    │       └── index.html
    └── static/
        ├── css/
        │   └── styles.css
        └── js/
            └── camera.js

# (опционально для Kubeflow/KServe)
└── kserve/
    ├── inference_server.py
    ├── Dockerfile
    └── inferenceservice.yaml
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