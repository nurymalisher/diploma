import httpx
from ..core.config import settings


# async def get_steam_profile(steamid: str) -> dict:
#     try:
#         url = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             r = await client.get(url, params={"key": settings.STEAM_API_KEY, "steamids": steamid})
#
#         if r.status_code != 200 or not r.content:
#             return _default_profile(steamid)
#
#         players = r.json().get("response", {}).get("players", [])
#         if not players:
#             return _default_profile(steamid)
#
#         p = players[0]
#         return {
#             "username": p.get("personaname", f"User {steamid[-4:]}"),
#             "avatar": p.get("avatarfull"),
#             "profile_url": p.get("profileurl"),
#             "visibility": "public" if p.get("communityvisibilitystate") == 3 else "private",
#         }
#     except Exception:
#         return _default_profile(steamid)


async def get_owned_games(steamid: str) -> list[dict]:
    try:
        url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params={
                "key": settings.STEAM_API_KEY,
                "steamid": steamid,
                "include_appinfo": True,
                "include_played_free_games": True,
            })
        if r.status_code != 200 or not r.content:
            return []
        games = r.json().get("response", {}).get("games", [])
        return [
            {
                "appid": int(g["appid"]),
                "name": g.get("name", ""),
                "playtime_hours": round(g.get("playtime_forever", 0) / 60, 2),
            }
            for g in games
        ]
    except Exception:
        return []


async def get_recently_played(steamid: str) -> list[dict]:
    try:
        url = "https://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v1/"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url, params={"key": settings.STEAM_API_KEY, "steamid": steamid})
        if r.status_code != 200 or not r.content:
            return []
        games = r.json().get("response", {}).get("games", [])
        return [
            {
                "appid": int(g["appid"]),
                "name": g.get("name", ""),
                "playtime_hours": round(g.get("playtime_2weeks", 0) / 60, 2),
            }
            for g in games
        ]
    except Exception:
        return []


def _default_profile(steamid: str) -> dict:
    return {
        "username": f"User {steamid[-4:]}",
        "avatar": None,
        "profile_url": f"https://steamcommunity.com/profiles/{steamid}",
        "visibility": "unknown"
    }

async def get_steam_profile(steamid: str) -> dict | None:
    """Делает запрос к серверам Valve и забирает данные профиля"""
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {
        "key": settings.STEAM_API_KEY,  # Твой ключ из .env
        "steamids": steamid
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            data = response.json()
            players = data.get("response", {}).get("players", [])

            if players:
                p = players[0]
                return {
                    "steamid": steamid,
                    "username": p.get("personaname", "Unknown Player"),
                    "avatar": p.get("avatarfull", ""),
                    "profile_url": p.get("profileurl", "")
                }
        except Exception as e:
            print(f"Ошибка связи со Steam: {e}")

    return None