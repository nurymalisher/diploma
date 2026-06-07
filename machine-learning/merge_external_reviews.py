"""
merge_external_reviews_v3_random.py

Проблема v2:
    Брались первые 3 млн строк recommendations.csv.
    Внешний файл, похоже, отсортирован по app_id, поэтому попало только 217 игр.
    Из-за этого SVD стал хуже.

Что делает v3:
    Проходит по всему recommendations.csv чанками.
    Из каждого чанка берёт случайную долю строк.
    Так выборка покрывает намного больше разных игр.

Куда положить внешний файл:
    ../dataset/game_recommendations_on_steam/recommendations.csv

Запуск:
    python merge_external_reviews_v3_random.py
    python train.py

Важно:
    preprocess.py после этого НЕ запускать.
"""

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("../backend/data")
REVIEWS_PATH = DATA_DIR / "reviews_clean.csv"
ORIGINAL_BACKUP_PATH = DATA_DIR / "reviews_clean_original_backup.csv"
APPS_PATH = DATA_DIR / "apps_clean.csv"
EXT_PATH = Path("../dataset/game_recommendations_on_steam/recommendations.csv")

CHUNK_SIZE = 300_000

# Сколько внешних строк хотим примерно оставить ДО k-core.
# Если ПК тянет, можно поднять до 2_000_000.
TARGET_EXTERNAL_ROWS = 1_000_000

# Вероятность взять строку из каждого чанка.
# Для 41 млн строк 0.03 ≈ 1.2 млн строк до фильтров.
SAMPLE_FRAC = 0.03

MIN_USER_REVIEWS = 5
MIN_GAME_REVIEWS = 5

MAX_PLAYTIME_HOURS = 20_000
RANDOM_STATE = 42


def find_col(df, variants):
    cols = {c.lower(): c for c in df.columns}
    for v in variants:
        if v.lower() in cols:
            return cols[v.lower()]
    return None


def normalize_external_chunk(chunk, known_appids, rng):
    app_col = find_col(chunk, ["app_id", "appid"])
    user_col = find_col(chunk, ["user_id", "steamid", "author_steamid"])
    rec_col = find_col(chunk, ["is_recommended", "voted_up", "recommended"])
    hours_col = find_col(chunk, ["hours", "playtime_hours", "author_playtime_forever"])

    if not all([app_col, user_col, rec_col, hours_col]):
        raise ValueError(f"Не найдены нужные колонки. Есть: {list(chunk.columns)}")

    if SAMPLE_FRAC < 1.0:
        mask = rng.random(len(chunk)) < SAMPLE_FRAC
        chunk = chunk.loc[mask]
        if len(chunk) == 0:
            return pd.DataFrame()

    out = pd.DataFrame()
    out["appid"] = pd.to_numeric(chunk[app_col], errors="coerce")
    out = out.dropna(subset=["appid"])
    out["appid"] = out["appid"].astype(int)

    out = out[out["appid"].isin(known_appids)]
    if len(out) == 0:
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

    return out.drop_duplicates(subset=["author_steamid", "appid"], keep="last")


def kcore(df, min_user, min_game):
    print(f"K-core: user ≥ {min_user}, game ≥ {min_game}")
    while True:
        n0 = len(df)

        uc = df["author_steamid"].value_counts()
        keep_users = uc[uc >= min_user].index
        df = df[df["author_steamid"].isin(keep_users)]

        ic = df["appid"].value_counts()
        keep_items = ic[ic >= min_game].index
        df = df[df["appid"].isin(keep_items)]

        print(f"  {n0:,} → {len(df):,}")
        if len(df) == n0:
            return df


def main():
    print("=" * 70)
    print("MERGE EXTERNAL REVIEWS V3 RANDOM")
    print("=" * 70)

    if not EXT_PATH.exists():
        raise FileNotFoundError(f"Не найден внешний файл: {EXT_PATH}")
    if not APPS_PATH.exists():
        raise FileNotFoundError(f"Не найден apps_clean: {APPS_PATH}")

    apps = pd.read_csv(APPS_PATH, low_memory=False)
    app_col = find_col(apps, ["appid", "app_id"])
    known_appids = set(pd.to_numeric(apps[app_col], errors="coerce").dropna().astype(int))
    print(f"Игр в apps_clean: {len(known_appids):,}")

    # Берём исходный маленький датасет из backup, а не текущий испорченный merge.
    if ORIGINAL_BACKUP_PATH.exists():
        old = pd.read_csv(ORIGINAL_BACKUP_PATH)
        print(f"Старый датасет взят из backup: {ORIGINAL_BACKUP_PATH}")
    else:
        old = pd.read_csv(REVIEWS_PATH)
        old.to_csv(ORIGINAL_BACKUP_PATH, index=False)
        print(f"Backup создан: {ORIGINAL_BACKUP_PATH}")

    print(f"Старых отзывов: {len(old):,}")

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

    rng = np.random.default_rng(RANDOM_STATE)
    parts = []
    total_read = 0
    total_kept = 0

    for i, chunk in enumerate(pd.read_csv(EXT_PATH, chunksize=CHUNK_SIZE), start=1):
        total_read += len(chunk)
        part = normalize_external_chunk(chunk, known_appids, rng)

        if len(part) > 0:
            parts.append(part)
            total_kept += len(part)

        if i % 10 == 0:
            games_so_far = pd.concat(parts, ignore_index=True)["appid"].nunique() if parts else 0
            print(f"  chunks={i}, read={total_read:,}, kept={total_kept:,}, games={games_so_far:,}")

        if total_kept >= TARGET_EXTERNAL_ROWS:
            break

    if not parts:
        raise RuntimeError("Не получилось взять внешние строки. Проверь файл recommendations.csv.")

    ext = pd.concat(parts, ignore_index=True)
    ext = ext.drop_duplicates(subset=["author_steamid", "appid"], keep="last")

    print("=" * 70)
    print(f"Внешних строк после случайной выборки: {len(ext):,}")
    print(f"Внешних пользователей: {ext['author_steamid'].nunique():,}")
    print(f"Внешних игр: {ext['appid'].nunique():,}")

    combined = pd.concat([old, ext], ignore_index=True)
    combined = combined.drop_duplicates(subset=["author_steamid", "appid"], keep="last")

    print("=" * 70)
    print(f"До k-core: {len(combined):,}")
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
