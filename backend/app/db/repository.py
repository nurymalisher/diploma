import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

# Путь к БД относительно корня проекта
DB_PATH = Path("data/users.db")

WEIGHTS = {
    "rate_5": 3.0, "rate_4": 2.0, "rate_3": 0.5,
    "rate_2": -1.0, "rate_1": -2.0,
    "favorite": 2.5, "wishlist": 1.5, "view": 0.3,
}


def _get_weight(action: str, value: float = None) -> float:
    if action == "rate" and value is not None:
        key = f"rate_{int(value)}"
        return WEIGHTS.get(key, 0.0)
    return WEIGHTS.get(action, 0.0)


class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = path

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript("""
                               -- Твои старые таблицы...
                               CREATE TABLE IF NOT EXISTS user_interactions
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   steamid
                                   TEXT
                                   NOT
                                   NULL,
                                   appid
                                   INTEGER
                                   NOT
                                   NULL,
                                   action
                                   TEXT
                                   NOT
                                   NULL,
                                   value
                                   REAL,
                                   weight
                                   REAL
                                   NOT
                                   NULL
                                   DEFAULT
                                   0,
                                   created_at
                                   TEXT
                                   NOT
                                   NULL
                                   DEFAULT (
                                   datetime
                               (
                                   'now'
                               )),
                                   updated_at TEXT NOT NULL DEFAULT
                               (
                                   datetime
                               (
                                   'now'
                               ))
                                   );
                               CREATE INDEX IF NOT EXISTS idx_interactions_steamid ON user_interactions(steamid);
                               CREATE INDEX IF NOT EXISTS idx_interactions_steamid_appid ON user_interactions(steamid, appid);
                               CREATE UNIQUE INDEX IF NOT EXISTS idx_interactions_unique ON user_interactions(steamid, appid, action);

                               CREATE TABLE IF NOT EXISTS users
                               (
                                   steamid
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   username
                                   TEXT,
                                   avatar
                                   TEXT,
                                   profile_url
                                   TEXT,
                                   cached_at
                                   TEXT
                                   NOT
                                   NULL
                                   DEFAULT (
                                   datetime
                               (
                                   'now'
                               ))
                                   );

                               -- 🚀 НОВАЯ ТАБЛИЦА ДЛЯ ИГР 
                               CREATE TABLE IF NOT EXISTS games
                               (
                                   appid
                                   INTEGER
                                   PRIMARY
                                   KEY,
                                   name
                                   TEXT
                                   NOT
                                   NULL,
                                   name_lower
                                   TEXT
                                   NOT
                                   NULL,
                                   header_image
                                   TEXT,
                                   is_free
                                   BOOLEAN,
                                   price_usd
                                   REAL,
                                   genres
                                   TEXT
                               );
                               -- Индекс для мгновенного поиска по названию
                               CREATE INDEX IF NOT EXISTS idx_games_name ON games(name_lower);
                               """)

    # 🚀 НОВЫЙ МЕТОД ДЛЯ ПОИСКА
    def search_games(self, query: str, limit: int = 8) -> list[dict]:
        with self._conn() as conn:
            # Ищем подстроку и сортируем: сначала те, что НАЧИНАЮТСЯ с запроса, потом остальные
            rows = conn.execute("""
                                SELECT *
                                FROM games
                                WHERE name_lower LIKE ?
                                ORDER BY CASE WHEN name_lower LIKE ? THEN 0 ELSE 1 END,
                                         length(name) LIMIT ?
                                """, (f"%{query}%", f"{query}%", limit)).fetchall()

        results = []
        for r in rows:
            results.append({
                "appid": r["appid"],
                "name": r["name"],
                "header_image": r["header_image"],
                "is_free": bool(r["is_free"]),
                "price_usd": r["price_usd"],
                "genres": r["genres"].split(",") if r["genres"] else []
            })
        return results

    # 🚀 НОВЫЙ МЕТОД ДЛЯ ЗАЛИВКИ ИГР (нужен только 1 раз)
    def import_games(self, games_list: list[tuple]):
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO games 
                (appid, name, name_lower, header_image, is_free, price_usd, genres)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, games_list)

    def add_interaction(self, steamid: str, appid: int, action: str, value: float = None) -> dict:
        weight = _get_weight(action, value)
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                         INSERT INTO user_interactions (steamid, appid, action, value, weight, created_at, updated_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(steamid, appid, action) DO
                         UPDATE SET
                             value = excluded.value, weight = excluded.weight, updated_at = excluded.updated_at
                         """, (steamid, appid, action, value, weight, now, now))
        return {"steamid": steamid, "appid": appid, "action": action, "value": value, "weight": weight}

    def remove_interaction(self, steamid: str, appid: int, action: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM user_interactions WHERE steamid=? AND appid=? AND action=?",
                         (steamid, appid, action))

    def get_user_interactions(self, steamid: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT appid, action, value, weight, updated_at FROM user_interactions WHERE steamid=? ORDER BY updated_at DESC",
                (steamid,)).fetchall()
        return [dict(r) for r in rows]

    def get_game_interaction(self, steamid: str, appid: int) -> dict:
        with self._conn() as conn:
            rows = conn.execute("SELECT action, value, weight FROM user_interactions WHERE steamid=? AND appid=?",
                                (steamid, appid)).fetchall()
        result = {"rating": None, "favorite": False, "wishlist": False}
        for row in rows:
            if row["action"] == "rate":
                result["rating"] = row["value"]
            elif row["action"] == "favorite":
                result["favorite"] = True
            elif row["action"] == "wishlist":
                result["wishlist"] = True
        return result

    def get_aggregated_weights(self, steamid: str) -> dict[int, float]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT appid, SUM(weight) as total_weight FROM user_interactions WHERE steamid=? GROUP BY appid",
                (steamid,)).fetchall()
        return {row["appid"]: row["total_weight"] for row in rows}

    def get_favorites(self, steamid: str) -> list[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT appid FROM user_interactions WHERE steamid=? AND action='favorite' ORDER BY updated_at DESC",
                (steamid,)).fetchall()
        return [r["appid"] for r in rows]

    def get_wishlist(self, steamid: str) -> list[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT appid FROM user_interactions WHERE steamid=? AND action='wishlist' ORDER BY updated_at DESC",
                (steamid,)).fetchall()
        return [r["appid"] for r in rows]

    def save_user(self, steamid: str, username: str, avatar: str, profile_url: str):
        with self._conn() as conn:
            conn.execute("""
                         INSERT INTO users (steamid, username, avatar, profile_url, cached_at)
                         VALUES (?, ?, ?, ?, datetime('now')) ON CONFLICT(steamid) DO
                         UPDATE SET
                             username = excluded.username, avatar = excluded.avatar, profile_url = excluded.profile_url, cached_at = excluded.cached_at
                         """, (steamid, username, avatar, profile_url))

    def get_user(self, steamid: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE steamid=?", (steamid,)).fetchone()
        return dict(row) if row else None

    def get_stats(self) -> dict:
        with self._conn() as conn:
            users = conn.execute("SELECT COUNT(DISTINCT steamid) FROM user_interactions").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM user_interactions").fetchone()[0]
            favs = conn.execute("SELECT COUNT(*) FROM user_interactions WHERE action='favorite'").fetchone()[0]
            rates = conn.execute("SELECT COUNT(*) FROM user_interactions WHERE action='rate'").fetchone()[0]
        return {"total_users": users, "total_interactions": total, "total_favorites": favs, "total_ratings": rates}


db = Database()