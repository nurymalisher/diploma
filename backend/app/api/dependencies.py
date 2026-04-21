from fastapi import Request, HTTPException

def get_current_steamid(request: Request) -> str:
    steamid = request.session.get("steamid")
    if not steamid:
        raise HTTPException(401, "Войдите через Steam: GET /auth/steam")
    return steamid

def get_optional_steamid(request: Request) -> str | None:
    return request.session.get("steamid")