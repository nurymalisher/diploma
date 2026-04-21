from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse

from ..api.dependencies import get_current_steamid
from ..services import rec_service
from ..ml.recommender import recommender

# Без префикса, чтобы сохранить пути совместимыми с фронтендом
router = APIRouter(tags=["recommendations"])


@router.get("/recommendations")
async def recommendations_endpoint(
        request: Request,
        top_n: int = 20,
        min_reviews: int = 50,
        exclude_free: bool = False,
        steamid: str = Depends(get_current_steamid)
):
    result = await rec_service.get_personalized_recommendations(
        steamid, top_n, min_reviews, exclude_free
    )

    if "error" in result:
        return JSONResponse({
            "error": result["error"],
            "fix": "Steam → Настройки → Приватность → Библиотека игр: Публичная"
        })

    return JSONResponse({
        "steamid": steamid,
        "username": request.session.get("username"),
        **result
    })


@router.get("/recommendations/recent")
async def recommendations_recent_endpoint(
        top_n: int = 20,
        steamid: str = Depends(get_current_steamid)
):
    result = await rec_service.get_recent_recommendations(steamid, top_n)
    if "error" in result:
        return JSONResponse({"error": result["error"]})

    return JSONResponse({"steamid": steamid, **result})


@router.post("/recommendations/anon")
def recommendations_anon(data: dict):
    owned_games = data.get("owned_games", [])
    top_n = data.get("top_n", 20)

    if not owned_games:
        return JSONResponse({"recommendations": []})

    recs = recommender.get_recommendations(
        owned_games=owned_games,
        steamid=None,
        top_n=top_n,
        min_reviews=50,
    )
    return JSONResponse({"recommendations": recs})


@router.get("/similar/{appid}")
def get_similar_games(appid: int, top_n: int = 10):
    game_info = None
    if appid in recommender.apps_meta.index:
        g = recommender.apps_meta.loc[appid]
        game_info = {
            "appid": appid,
            "name": g.get("name"),
            "store_url": f"https://store.steampowered.com/app/{appid}",
        }

    similar = recommender.get_similar_games(appid, top_n=top_n)

    if not similar and not game_info:
        raise HTTPException(404, f"Игра appid={appid} не найдена")

    return JSONResponse({"game": game_info, "similar": similar, "count": len(similar)})