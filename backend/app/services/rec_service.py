from ..services import steam_api
from ..db.repository import db
from ..ml.recommender import recommender


def _get_weights_used(steamid: str | None) -> dict:
    use_svd = steamid is not None and str(steamid) in recommender.user_to_idx

    if use_svd:
        return {
            "mode": "auth",
            "tfidf": round(recommender.auth_alpha, 3),
            "svd": round(recommender.auth_beta, 3),
            "embeddings": round(recommender.auth_gamma, 3),
        }

    return {
        "mode": "anon_or_no_svd_profile",
        "tfidf": round(recommender.anon_alpha, 3),
        "svd": round(recommender.anon_beta, 3),
        "embeddings": round(recommender.anon_gamma, 3),
    }


async def get_personalized_recommendations(
    steamid: str,
    top_n: int = 20,
    min_reviews: int = 50,
    exclude_free: bool = False
) -> dict:
    owned = await steam_api.get_owned_games(steamid)

    if not owned:
        return {"error": "Библиотека пуста или профиль приватный"}

    db_weights = db.get_aggregated_weights(steamid)

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
        "weights_used": _get_weights_used(steamid),
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
        "weights_used": _get_weights_used(steamid),
    }