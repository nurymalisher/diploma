from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..ml.recommender import recommender


router = APIRouter(tags=["search"])


@router.get("/search")
def search_games(q: str = "", limit: int = 8):
    if not q or len(q.strip()) < 2:
        return JSONResponse({"results": []})

    q_lower = q.strip().lower()
    results = []

    # Сначала ищем игры, которые начинаются с запроса
    for item in recommender.search_items:
        if item["name_lower"].startswith(q_lower):
            results.append(item)

            if len(results) >= limit:
                return JSONResponse({
                    "results": [_clean_item(item) for item in results]
                })

    # Потом ищем игры, где запрос встречается внутри названия
    for item in recommender.search_items:
        if q_lower in item["name_lower"] and not item["name_lower"].startswith(q_lower):
            results.append(item)

            if len(results) >= limit:
                break

    return JSONResponse({
        "results": [_clean_item(item) for item in results[:limit]]
    })


def _clean_item(item: dict) -> dict:
    return {
        "appid": item["appid"],
        "name": item["name"],
        "header_image": item["header_image"],
        "is_free": item["is_free"],
        "price_usd": item["price_usd"],
        "recommendations": item["recommendations"],
        "metacritic": item["metacritic"],
        "store_url": item["store_url"],
    }