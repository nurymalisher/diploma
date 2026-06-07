from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .core.config import settings
from .api import recommendations, interactions, search
from .api import auth
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

app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET)

Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Все backend endpoints теперь доступны через /api/...
app.include_router(auth.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(interactions.router, prefix="/api")
app.include_router(search.router, prefix="/api")

# React build из frontend/dist
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_ASSETS.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_ASSETS)),
        name="frontend-assets",
    )


@app.get("/", include_in_schema=False)
def serve_frontend_root():
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    return JSONResponse(
        {"message": "Frontend build не найден. Выполни: cd frontend && npm run build"},
        status_code=404,
    )


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_routes(full_path: str):
    if full_path.startswith("api/"):
        return JSONResponse(
            {"detail": "API endpoint not found"},
            status_code=404,
        )

    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))

    return JSONResponse(
        {"message": "Frontend build не найден. Выполни: cd frontend && npm run build"},
        status_code=404,
    )
