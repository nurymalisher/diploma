from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..api.dependencies import get_current_steamid, get_optional_steamid
from ..db.repository import db
from ..ml.recommender import recommender
from ..services import steam_api

router = APIRouter(tags=["interactions", "user"])

# -- ПРОФИЛЬ --
@router.get("/me")
async def my_profile(steamid: str = Depends(get_current_steamid)):
    profile = await steam_api.get_steam_profile(steamid)
    owned = await steam_api.get_owned_games(steamid)
    recent = await steam_api.get_recently_played(steamid)

    our_appids = set(recommender.apps_meta.index)
    matched = [g for g in owned if g["appid"] in our_appids]
    total_hours = sum(g["playtime_hours"] for g in owned)

    return JSONResponse({
        "steamid": steamid,
        **profile,
        "library": {
            "total_games": len(owned),
            "matched_in_db": len(matched),
            "total_hours": round(total_hours, 1),
        },
        "recent_games": recent[:5],
    })

# -- ДЕЙСТВИЯ С ИГРАМИ --
@router.post("/game/{appid}/rate")
async def rate_game(appid: int, data: dict, steamid: str = Depends(get_current_steamid)):
    rating = data.get("rating")
    if not rating or rating not in [1, 2, 3, 4, 5]:
        raise HTTPException(400, "Оценка должна быть от 1 до 5")
    result = db.add_interaction(steamid, appid, "rate", value=float(rating))
    return JSONResponse({"status": "ok", "appid": appid, "rating": rating, "weight": result["weight"]})

@router.delete("/game/{appid}/rate")
async def remove_rating(appid: int, steamid: str = Depends(get_current_steamid)):
    db.remove_interaction(steamid, appid, "rate")
    return JSONResponse({"status": "ok"})

@router.post("/game/{appid}/favorite")
async def add_favorite(appid: int, steamid: str = Depends(get_current_steamid)):
    db.add_interaction(steamid, appid, "favorite")
    return JSONResponse({"status": "ok", "favorite": True})

@router.delete("/game/{appid}/favorite")
async def remove_favorite(appid: int, steamid: str = Depends(get_current_steamid)):
    db.remove_interaction(steamid, appid, "favorite")
    return JSONResponse({"status": "ok", "favorite": False})

@router.post("/game/{appid}/wishlist")
async def add_wishlist(appid: int, steamid: str = Depends(get_current_steamid)):
    db.add_interaction(steamid, appid, "wishlist")
    return JSONResponse({"status": "ok", "wishlist": True})

@router.delete("/game/{appid}/wishlist")
async def remove_wishlist(appid: int, steamid: str = Depends(get_current_steamid)):
    db.remove_interaction(steamid, appid, "wishlist")
    return JSONResponse({"status": "ok", "wishlist": False})

@router.post("/game/{appid}/view")
async def track_view(appid: int, steamid: str | None = Depends(get_optional_steamid)):
    if steamid:
        db.add_interaction(steamid, appid, "view")
    return JSONResponse({"status": "ok"})

@router.get("/game/{appid}/status")
async def game_status(appid: int, steamid: str | None = Depends(get_optional_steamid)):
    if not steamid:
        return JSONResponse({"rating": None, "favorite": False, "wishlist": False})
    return JSONResponse(db.get_game_interaction(steamid, appid))

# -- СПИСКИ ПОЛЬЗОВАТЕЛЯ --
def _get_games_data(appids):
    games = []
    for appid in appids:
        if appid in recommender.apps_meta.index:
            g = recommender.apps_meta.loc[appid]
            games.append({
                "appid": appid,
                "name": g.get("name"),
                "header_image": g.get("header_image"),
                "price_usd": recommender._cents_to_usd(g.get("mat_final_price")),
                "store_url": f"https://store.steampowered.com/app/{appid}",
            })
    return games

@router.get("/my/favorites")
async def my_favorites(steamid: str = Depends(get_current_steamid)):
    appids = db.get_favorites(steamid)
    games = _get_games_data(appids)
    return JSONResponse({"favorites": games, "count": len(games)})

@router.get("/my/wishlist")
async def my_wishlist(steamid: str = Depends(get_current_steamid)):
    appids = db.get_wishlist(steamid)
    games = _get_games_data(appids)
    return JSONResponse({"wishlist": games, "count": len(games)})