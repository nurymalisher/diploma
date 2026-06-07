"""
merge_steam_recommendations.py

Назначение:
    Подмешивает большой датасет Game Recommendations on Steam к текущему reviews_clean.csv.

Что нужно скачать:
    Kaggle: antonkozyriev/game-recommendations-on-steam
    Нужен файл recommendations.csv.

Куда положить:
    ../dataset/game_recommendations_on_steam/recommendations.csv

Запуск:
    python merge_steam_recommendations.py

После этого:
    python train.py

Скрипт делает backup старого reviews_clean.csv:
    reviews_clean_before_external.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("../backend/data")
EXTERNAL_PATH = Path("../dataset/game_recommendations_on_steam/recommendations.csv")

CURRENT_REVIEWS = DATA_DIR / "reviews_clean.csv"
BACKUP_REVIEWS = DATA_DIR / "reviews_clean_before_external.csv"
APPS_PATH = DATA_DIR / "apps_clean.csv"

# Чтобы не убить ноутбук/ПК, сначала берём часть большого датасета.
# Потом можно поднять до 3_000_000, 5_000_000 и т.д.
MAX_EXTERNAL_ROWS = 1_000_000

# Оставляем пользователей и игры, по которым есть хотя бы минимум взаимодействий.
MIN_USER_REVIEWS = 5
MIN_GAME_REVIEWS = 3

# Внешний датасет может иметь очень большие playtime. Ограничим выбросы.
MAX_PLAYTIME_HOURS = 20_000


def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def main():
    if not CURRENT_REVIEWS.exists():
        raise FileNotFoundError(f"Не найден {CURRENT_REVIEWS}")

    if not APPS_PATH.exists():
        raise FileNotFoundError(f"Не найден {APPS_PATH}")

    if not EXTERNAL_PATH.exists():
        raise FileNotFoundError(
            f"Не найден внешний датасет: {EXTERNAL_PATH}\n"
            "Скачай Kaggle antonkozyriev/game-recommendations-on-steam "
            "и положи recommendations.csv в указанную папку."
        )

    print("=" * 70)
    print("MERGE — добавление внешнего Steam recommendations dataset")
    print("=" * 70)

    apps = pd.read_csv(APPS_PATH)
    app_col = pick_col(apps, ["appid", "app_id"])
    apps[app_col] = apps[app_col].astype(int)
    known_appids = set(apps[app_col].astype(int))
    print(f"Игр в текущем каталоге: {len(known_appids):,}")

    old = pd.read_csv(CURRENT_REVIEWS)
    print(f"Текущих отзывов: {len(old):,}")

    if not BACKUP_REVIEWS.exists():
        old.to_csv(BACKUP_REVIEWS, index=False)
        print(f"Backup создан: {BACKUP_REVIEWS}")
    else:
        print(f"Backup уже есть: {BACKUP_REVIEWS}")

    print(f"Читаю внешний файл: {EXTERNAL_PATH}")
    ext = pd.read_csv(EXTERNAL_PATH, nrows=MAX_EXTERNAL_ROWS)
    print(f"Внешних строк загружено: {len(ext):,}")
    print(f"Колонки внешнего датасета: {list(ext.columns)}")

    col_app = pick_col(ext, ["app_id", "appid"])
    col_user = pick_col(ext, ["user_id", "steamid", "author_steamid"])
    col_rec = pick_col(ext, ["is_recommended", "voted_up", "recommended"])
    col_hours = pick_col(ext, ["hours", "playtime_hours", "author_playtime_forever"])
    col_helpful = pick_col(ext, ["helpful", "votes_up", "weighted_vote_score"])

    missing = []
    if col_app is None:
        missing.append("app_id/appid")
    if col_user is None:
        missing.append("user_id/steamid/author_steamid")
    if col_rec is None:
        missing.append("is_recommended/voted_up/recommended")
    if col_hours is None:
        missing.append("hours/playtime_hours/author_playtime_forever")

    if missing:
        raise ValueError(f"Не нашёл обязательные колонки: {missing}")

    out = pd.DataFrame()
    out["appid"] = ext[col_app].astype(int)
    out["author_steamid"] = ext[col_user].astype(str)

    rec = ext[col_rec]
    if rec.dtype == bool:
        out["voted_up"] = rec.astype(bool)
    else:
        out["voted_up"] = rec.astype(str).str.lower().isin(["true", "1", "yes", "recommended", "positive"])

    out["rating"] = out["voted_up"].astype(int)

    # hours обычно уже в часах. Если вдруг колонка author_playtime_forever — это минуты.
    if col_hours.lower() == "author_playtime_forever":
        out["playtime_hours"] = pd.to_numeric(ext[col_hours], errors="coerce").fillna(0) / 60.0
    else:
        out["playtime_hours"] = pd.to_numeric(ext[col_hours], errors="coerce").fillna(0)

    out["playtime_hours"] = out["playtime_hours"].clip(lower=0, upper=MAX_PLAYTIME_HOURS)

    if col_helpful:
        helpful = pd.to_numeric(ext[col_helpful], errors="coerce").fillna(0)
        if helpful.max() > 1:
            helpful = np.log1p(helpful)
            helpful = helpful / helpful.max() if helpful.max() > 0 else helpful
        out["weighted_vote_score"] = helpful.clip(0, 1)
    else:
        out["weighted_vote_score"] = 0.0

    # Только игры, которые есть в текущем каталоге apps_clean.
    before = len(out)
    out = out[out["appid"].isin(known_appids)]
    print(f"После пересечения с apps_clean: {before:,} → {len(out):,}")

    # Убираем мусор.
    out = out.dropna(subset=["appid", "author_steamid"])
    out = out[out["playtime_hours"] >= 0]
    out = out.drop_duplicates(subset=["author_steamid", "appid"], keep="last")

    # Чтобы user_id из внешнего датасета случайно не конфликтовал со SteamID из старого,
    # добавляем префикс.
    out["author_steamid"] = "ext_" + out["author_steamid"].astype(str)

    # Приводим старый датасет к тем же колонкам.
    for col in ["voted_up", "rating", "playtime_hours", "weighted_vote_score"]:
        if col not in old.columns:
            if col == "voted_up":
                old[col] = old.get("rating", 1).astype(int) > 0
            elif col == "rating":
                old[col] = old.get("voted_up", True).astype(int)
            else:
                old[col] = 0.0

    keep_cols = ["appid", "author_steamid", "voted_up", "rating", "playtime_hours", "weighted_vote_score"]
    combined = pd.concat([old[keep_cols], out[keep_cols]], ignore_index=True)

    print(f"До k-core: {len(combined):,} отзывов")
    while True:
        n = len(combined)

        uc = combined["author_steamid"].value_counts()
        users = uc[uc >= MIN_USER_REVIEWS].index
        combined = combined[combined["author_steamid"].isin(users)]

        ic = combined["appid"].value_counts()
        items = ic[ic >= MIN_GAME_REVIEWS].index
        combined = combined[combined["appid"].isin(items)]

        if len(combined) == n:
            break

    combined.to_csv(CURRENT_REVIEWS, index=False)

    print("=" * 70)
    print("ГОТОВО")
    print(f"Итоговых отзывов: {len(combined):,}")
    print(f"Пользователей: {combined['author_steamid'].nunique():,}")
    print(f"Игр во взаимодействиях: {combined['appid'].nunique():,}")
    print(f"Сохранено в: {CURRENT_REVIEWS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
