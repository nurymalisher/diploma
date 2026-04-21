from ..services import steam_api
from ..db.repository import db
from ..ml.recommender import recommender

async def get_personalized_recommendations(
    steamid: str,
    top_n: int = 20,
    min_reviews: int = 50,
    exclude_free: bool = False
) -> dict:
    # 1. Получаем купленные игры из Steam
    owned = await steam_api.get_owned_games(steamid)
    if not owned:
        return {"error": "Библиотека пуста или профиль приватный"}

    # 2. Получаем оценки и лайки из нашей БД
    db_weights = db.get_aggregated_weights(steamid)

    # 3. Генерируем рекомендации (модель сама обогатит профиль через _enrich_with_db)
    recs = recommender.get_recommendations(
        owned_games=owned,
        steamid=steamid,
        top_n=top_n,
        min_reviews=min_reviews,
        exclude_free=exclude_free,
        db_weights=db_weights if db_weights else None,
    )

    return {
        "library_size": len(owned),
        "recommendations": recs,
        "count": len(recs),
        "weights_used": {
            "tfidf": round(recommender.alpha, 3),
            "svd": round(recommender.beta, 3),
            "embeddings": round(recommender.gamma, 3),
        }
    }

async def get_recent_recommendations(steamid: str, top_n: int = 20) -> dict:
    recent = await steam_api.get_recently_played(steamid)
    if not recent:
        return {"error": "Нет активности за последние 2 недели"}

    recs = recommender.get_recommendations(
        owned_games=recent,
        steamid=steamid,
        top_n=top_n,
    )

    return {
        "mode": "recent_2_weeks",
        "based_on": len(recent),
        "recent_games": [g["name"] for g in recent],
        "recommendations": recs,
        "count": len(recs),
    }