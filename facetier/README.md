# FaceTier

Жёсткий разбор лица в Telegram (Mini App).

## Текущий статус (MVP в разработке)

- [x] Бот создан (@Facerst_bot)
- [x] Базовая структура проекта
- [x] FastAPI бэкенд с эндпоинтом `/api/analyze`
- [x] Mini App (HTML) с тёмным дизайном, загрузкой фото, полом/возрастом
- [x] Скелет анализатора лица на MediaPipe
- [ ] Реальный подсчёт всех метрик
- [ ] Деплой Mini App + бэкенда
- [ ] Подключение WebApp кнопки в боте
- [ ] PDF-отчёт
- [ ] Оплата (RollyPay) — позже

## Структура

```
facetier/
├── bot/                  # Telegram бот (aiogram)
│   └── main.py
├── backend/
│   └── app/
│       ├── main.py       # FastAPI
│       └── analysis/
│           └── face_analyzer.py
├── miniapp/
│   └── public/
│       └── index.html    # Mini App
├── .env
└── README.md
```

## Как запустить локально (для разработки)

### 1. Бот
```bash
cd bot
python main.py
```

### 2. Бэкенд + Mini App
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

После этого Mini App будет доступен по адресу:
`http://localhost:8000/app`

## Следующий шаг

Нужно задеплоить бэкенд + статику Mini App на публичный HTTPS-адрес (Railway / Render / Vercel + бэкенд), после чего прописать `MINIAPP_URL` в `.env` и перезапустить бота.
