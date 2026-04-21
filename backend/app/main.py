from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .core.config import settings
from .api import recommendations, interactions, search
from .api import auth

# Добавляем эти два импорта:
from .db.repository import db
from .ml.recommender import recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Инициализация базы данных...")
    db.init()

    print("🧠 Загрузка ML моделей (это может занять пару секунд)...")
    recommender.load()

    print("✅ Сервер успешно запущен и готов к работе!")
    yield

    print("🛑 Сервер останавливается...")


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Подключаем сессии
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)

# Статика
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключаем роутеры (контроллеры)
app.include_router(auth.router)
app.include_router(recommendations.router)
app.include_router(interactions.router)
app.include_router(search.router)

# Главная страница (Фронтенд)
@app.get("/", include_in_schema=False)
def frontend():
    index = Path("static/index.html")
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Положи index.html в папку static/"})