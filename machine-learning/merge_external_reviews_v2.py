"""
merge_external_reviews_v2.py

Цель:
    Нормально добавить большой датасет Game Recommendations on Steam
    к текущему backend/data/reviews_clean.csv.

Ожидаемый внешний файл:
    ../dataset/game_recommendations_on_steam/recommendations.csv

Запуск:
    python merge_external_reviews_v2.py

Важно:
    После успешного merge в train.py в STEP 1 должно быть отзывов намного больше,
    чем 128k. Например 500k+, 1M+ и т.д.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("../backend/data")
REVIEWS_PATH = DATA_DIR / "reviews_clean.csv"
BACKUP_PATH = DATA_DIR / "reviews_clean_original_backup.csv"
APPS_PATH = DATA_DIR / "apps_clean.csv"

EXT_PATH = Path("../dataset/game_recommendations_on_steam/recommendations.csv")

# Сколько внешних строк прочитать. Если ПК тянет, можно поставить 3_000_000 или 5_000_000.
MAX_EXTERNAL_ROWS = 3_000_000
CHUNK_SIZE = 300_000

# Не делаем слишком жёстко, иначе опять потеряем много данных.
MIN_USER_REVIEWS = 3
MIN_GAME_REVIEWS = 2

MAX_PLAYTIME_HOURS = 20_000


def find_col(df, variants):
    cols = {c.lower(): c for c in df.columns}
    for v in variants:
        if v.lower() in cols:
            return cols[v.lower()]
    return None


def normalize_external_chunk(chunk, known_appids, chunk_no):
    app_col = find_col(chunk, ["app_id", "appid"])
    user_col = find_col(chunk, ["user_id", "steamid", "author_steamid"])
    rec_col = find_col(chunk, ["is_recommended", "voted_up", "recommended"])
    hours_col = find_col(chunk, ["hours", "playtime_hours", "author_playtime_forever"])

    if not all([app_col, user_col, rec_col, hours_col]):
        raise ValueError(
            "Не найдены нужные колонки во внешнем датасете. "
            f"Есть колонки: {list(chunk.columns)}"
        )

    out = pd.DataFrame()
    out["appid"] = pd.to_numeric(chunk[app_col], errors="coerce")
    out = out.dropna(subset=["appid"])
    out["appid"] = out["appid"].astype(int)

    before = len(out)
    out = out[out["appid"].isin(known_appids)]
    if len(out) == 0:
        print(f"  chunk {chunk_no}: после пересечения с apps_clean 0 строк из {before:,}")
        return out

    chunk = chunk.loc[out.index]

    out["author_steamid"] = "ext_" + chunk[user_col].astype(str)

    rec = chunk[rec_col]
    if rec.dtype == bool:
        out["voted_up"] = rec.astype(bool)
    else:
        out["voted_up"] = rec.astype(str).str.lower().isin(
            ["true", "1", "yes", "recommended", "positive"]
        )

    out["rating"] = out["voted_up"].astype(int)

    hours = pd.to_numeric(chunk[hours_col], errors="coerce").fillna(0)
    if hours_col.lower() == "author_playtime_forever":
        hours = hours / 60.0
    out["playtime_hours"] = hours.clip(lower=0, upper=MAX_PLAYTIME_HOURS)

    helpful_col = find_col(chunk, ["helpful", "votes_up", "weighted_vote_score"])
    if helpful_col:
        helpful = pd.to_numeric(chunk[helpful_col], errors="coerce").fillna(0)
        if helpful.max() > 1:
            helpful = np.log1p(helpful)
            helpful = helpful / helpful.max() if helpful.max() > 0 else helpful
        out["weighted_vote_score"] = helpful.clip(0, 1)
    else:
        out["weighted_vote_score"] = 0.0

    out = out.drop_duplicates(subset=["author_steamid", "appid"], keep="last")
    print(f"  chunk {chunk_no}: {before:,} → {len(out):,} после appid-фильтра")
    return out


def kcore(df, min_user, min_game):
    print(f"K-core: user ≥ {min_user}, game ≥ {min_game}")
    i = 0
    while True:
        i += 1
        n0 = len(df)

        uc = df["author_steamid"].value_counts()
        keep_users = uc[uc >= min_user].index
        df = df[df["author_steamid"].isin(keep_users)]

        ic = df["appid"].value_counts()
        keep_items = ic[ic >= min_game].index
        df = df[df["appid"].isin(keep_items)]

        print(f"  iteration {i}: {n0:,} → {len(df):,}")

        if len(df) == n0:
            return df


def main():
    print("=" * 70)
    print("MERGE EXTERNAL REVIEWS V2")
    print("=" * 70)

    if not REVIEWS_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {REVIEWS_PATH}")
    if not APPS_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {APPS_PATH}")
    if not EXT_PATH.exists():
        raise FileNotFoundError(
            f"Не найден внешний файл: {EXT_PATH}\n"
            "Положи recommendations.csv именно в эту папку."
        )

    apps = pd.read_csv(APPS_PATH)
    app_col = find_col(apps, ["appid", "app_id"])
    apps[app_col] = pd.to_numeric(apps[app_col], errors="coerce")
    apps = apps.dropna(subset=[app_col])
    known_appids = set(apps[app_col].astype(int))
    print(f"Игр в apps_clean: {len(known_appids):,}")

    old = pd.read_csv(REVIEWS_PATH)
    print(f"Старых отзывов: {len(old):,}")
    print(f"Старых пользователей: {old['author_steamid'].nunique():,}")
    print(f"Старых игр: {old['appid'].nunique():,}")

    if not BACKUP_PATH.exists():
        old.to_csv(BACKUP_PATH, index=False)
        print(f"Backup создан: {BACKUP_PATH}")
    else:
        print(f"Backup уже существует: {BACKUP_PATH}")

    for col in ["voted_up", "rating", "playtime_hours", "weighted_vote_score"]:
        if col not in old.columns:
            if col == "voted_up":
                old[col] = old.get("rating", 1).astype(int) > 0
            elif col == "rating":
                old[col] = old.get("voted_up", True).astype(int)
            else:
                old[col] = 0.0

    keep_cols = ["appid", "author_steamid", "voted_up", "rating", "playtime_hours", "weighted_vote_score"]
    old = old[keep_cols].copy()
    old["appid"] = old["appid"].astype(int)
    old["author_steamid"] = old["author_steamid"].astype(str)

    parts = []
    total_read = 0
    chunk_no = 0

    for chunk in pd.read_csv(EXT_PATH, chunksize=CHUNK_SIZE):
        chunk_no += 1
        total_read += len(chunk)

        part = normalize_external_chunk(chunk, known_appids, chunk_no)
        if len(part) > 0:
            parts.append(part)

        if total_read >= MAX_EXTERNAL_ROWS:
            break

    if not parts:
        raise RuntimeError(
            "Внешние строки не добавились. Скорее всего app_id внешнего датасета "
            "не пересекается с appid в apps_clean или файл не тот."
        )

    ext = pd.concat(parts, ignore_index=True)
    print("=" * 70)
    print(f"Прочитано внешних строк: {total_read:,}")
    print(f"Внешних строк после appid-фильтра: {len(ext):,}")
    print(f"Внешних пользователей: {ext['author_steamid'].nunique():,}")
    print(f"Внешних игр: {ext['appid'].nunique():,}")

    combined = pd.concat([old, ext], ignore_index=True)
    combined = combined.drop_duplicates(subset=["author_steamid", "appid"], keep="last")

    print("=" * 70)
    print(f"До k-core всего отзывов: {len(combined):,}")
    print(f"Пользователей: {combined['author_steamid'].nunique():,}")
    print(f"Игр: {combined['appid'].nunique():,}")

    combined = kcore(combined, MIN_USER_REVIEWS, MIN_GAME_REVIEWS)

    combined.to_csv(REVIEWS_PATH, index=False)

    print("=" * 70)
    print("ГОТОВО")
    print(f"Итоговых отзывов: {len(combined):,}")
    print(f"Итоговых пользователей: {combined['author_steamid'].nunique():,}")
    print(f"Итоговых игр: {combined['appid'].nunique():,}")
    print(f"Сохранено: {REVIEWS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
