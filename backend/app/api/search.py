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

    for appid, row in recommender.apps_meta.iterrows():
        name = str(row.get("name", ""))

        if q_lower in name.lower():
            results.append({
                "appid": int(appid),
                "name": name,
                "header_image": row.get("header_image"),
                "is_free": bool(row.get("is_free", False)),
                "price_usd": recommender._cents_to_usd(row.get("mat_final_price")),
                "recommendations": int(row.get("recommendations_total") or 0),
                "metacritic": row.get("metacritic_score"),
                "store_url": f"https://store.steampowered.com/app/{appid}",
            })

        if len(results) >= limit * 4:
            break

    results.sort(
        key=lambda x: (
            not x["name"].lower().startswith(q_lower),
            len(x["name"])
        )
    )

    return JSONResponse({"results": results[:limit]})