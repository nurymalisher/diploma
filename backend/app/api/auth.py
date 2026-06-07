import urllib.parse
import re

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from ..services.steam_api import get_steam_profile
from ..db.repository import db
from ..core.config import settings


router = APIRouter(tags=["auth"])


@router.get("/auth/steam")
def login_with_steam(request: Request):
    params = {
        "openid.ns": "http://specs.openid.net/auth/2.0",
        "openid.mode": "checkid_setup",
        "openid.return_to": f"{settings.BASE_URL}/api/auth/steam/callback",
        "openid.realm": settings.BASE_URL,
        "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
        "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
    }

    url = f"https://steamcommunity.com/openid/login?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=url)


@router.get("/auth/steam/callback")
async def steam_callback(request: Request):
    claimed_id = request.query_params.get("openid.claimed_id", "")
    match = re.search(r"id/(\d+)", claimed_id)

    if not match:
        return RedirectResponse(url=f"{settings.BASE_URL}/")

    steamid = match.group(1)

    profile_data = await get_steam_profile(steamid)

    if profile_data:
        with db._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO users (steamid, username, avatar, profile_url)
                VALUES (?, ?, ?, ?)
                """,
                (
                    profile_data["steamid"],
                    profile_data["username"],
                    profile_data["avatar"],
                    profile_data["profile_url"],
                ),
            )

    request.session["steamid"] = steamid

    return RedirectResponse(url=f"{settings.BASE_URL}/")


@router.get("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url=f"{settings.BASE_URL}/")
