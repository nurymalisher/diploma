# from fastapi import APIRouter
# from fastapi.responses import JSONResponse
# from app.ml.recommender import recommender
#
# router = APIRouter(tags=["search"])
#
# @router.get("/search")
# def search_games(q: str = "", limit: int = 8):
#     if not q or len(q) < 2:
#         return JSONResponse({"results": []})
#
#     q_lower = q.lower()
#     results = []
#
#     # TODO: Оптимизировать этот поиск! (Уйти от iterrows)
#     for appid, row in recommender.apps_meta.iterrows():
#         name = str(row.get("name", ""))
#         if q_lower in name.lower():
#             results.append({
#                 "appid": int(appid),
#                 "name": name,
#                 "header_image": row.get("header_image"),
#                 "is_free": bool(row.get("is_free", False)),
#                 "price_usd": recommender._cents_to_usd(row.get("mat_final_price")),
#                 "genres": recommender.genres_map.get(int(appid), []) if hasattr(recommender, "genres_map") else [],
#             })
#         if len(results) >= limit * 4:
#             break
#
#     results.sort(key=lambda x: (
#         not x["name"].lower().startswith(q_lower),
#         len(x["name"])
#     ))
#
#     return JSONResponse({"results": results[:limit]})

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..db.repository import db

router = APIRouter(tags=["search"])

@router.get("/search")
def search_games(q: str = "", limit: int = 8):
    if not q or len(q) < 2:
        return JSONResponse({"results": []})

    # Передаем запрос в нижнем регистре
    results = db.search_games(q.lower(), limit)

    return JSONResponse({"results": results})