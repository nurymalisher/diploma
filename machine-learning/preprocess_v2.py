"""
preprocess.py — очистка и подготовка данных перед обучением моделей.

Запускать ДО train.py:
    python preprocess.py
    python train.py

Что делает:
    1. Очищает applications.csv    → data/apps_clean.csv
    2. Очищает reviews.csv         → data/reviews_clean.csv
    3. Фильтрует эмбеддинги        → data/embeddings_clean.npy + map
    4. Выводит статистику до/после → понятно что убрали и почему

Всё сохраняется в папку data/ — train.py читает оттуда.
"""

import re
import html
import numpy as np
import pandas as pd
from pathlib import Path

CSV_DIR        = Path("../dataset/steam_dataset_2025_csv")
EMBEDDINGS_DIR = Path("../dataset/steam_dataset_2025_embeddings")
DATA_DIR       = Path("../backend/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Главные настройки предобработки
MIN_PLAYTIME_MINUTES = 1
MAX_PLAYTIME_MINUTES = 3_000_000
MIN_USER_REVIEWS = 5
MIN_GAME_REVIEWS = 3


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def clean_text(text: str) -> str:
    """
    Очищает текст описания игры:
    - Декодирует HTML entities (&amp; → &, &lt; → <)
    - Убирает HTML теги (<br>, <strong>, <i> и т.д.)
    - Убирает лишние пробелы и переносы строк
    - Приводит к нижнему регистру
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)                       # &amp; → &
    text = re.sub(r"<[^>]+>", " ", text)             # <br> → пробел
    text = re.sub(r"[^\w\s]", " ", text)             # знаки препинания → пробел
    text = re.sub(r"\s+", " ", text).strip()         # множественные пробелы
    text = text.lower()
    return text


def print_stat(label: str, before: int, after: int):
    removed = before - after
    pct     = removed / before * 100 if before > 0 else 0
    print(f"  {label}: {before:,} → {after:,}  (убрано {removed:,}, {pct:.1f}%)")


# =============================================================================
# STEP 1 — ОЧИСТКА ИГР
# =============================================================================

def clean_apps():
    print("\n" + "="*60)
    print("STEP 1 — Очистка applications.csv")
    print("="*60)

    apps = pd.read_csv(CSV_DIR / "applications.csv")
    n0   = len(apps)
    print(f"  Исходно записей: {n0:,}")

    # ── Оставляем только игры (убираем DLC, soundtrack, demo и т.д.) ─────────
    apps = apps[apps["type"] == "game"].copy()
    print_stat("Только type=game", n0, len(apps))

    # ── Убираем игры без названия ─────────────────────────────────────────────
    n = len(apps)
    apps = apps[apps["name"].notna() & (apps["name"].str.strip() != "")]
    print_stat("Есть название", n, len(apps))

    # ── Убираем игры без описания (нечего векторизировать) ────────────────────
    n = len(apps)
    apps = apps[apps["short_description"].notna() &
                (apps["short_description"].str.strip() != "")]
    print_stat("Есть описание", n, len(apps))

    # ── Убираем игры с очень коротким описанием (< 20 символов) ──────────────
    n = len(apps)
    apps = apps[apps["short_description"].str.len() >= 20]
    print_stat("Описание ≥ 20 символов", n, len(apps))

    # ── Убираем дубликаты по appid ────────────────────────────────────────────
    n = len(apps)
    apps = apps.drop_duplicates(subset="appid")
    print_stat("Уникальные appid", n, len(apps))

    # ── Убираем игры с явно мусорными ценами ──────────────────────────────────
    # Цена в центах: отрицательная или > $1000 (100000 центов) — аномалия
    n = len(apps)
    price_ok = (
        apps["mat_final_price"].isna() |
        apps["is_free"] |
        apps["mat_final_price"].between(0, 100_000)
    )
    apps = apps[price_ok]
    print_stat("Нормальная цена", n, len(apps))

    # ── Очищаем текст описания от HTML ───────────────────────────────────────
    print("  Очистка HTML из описаний...")
    apps["short_description"] = apps["short_description"].apply(clean_text)

    # ── Убираем дубликаты по описанию (asset flips — одно описание, разные ID)
    n = len(apps)
    apps = apps.drop_duplicates(subset="short_description", keep="first")
    print_stat("Уникальные описания (убраны asset flips)", n, len(apps))

    # ── Нормализуем числовые поля ─────────────────────────────────────────────
    apps["mat_final_price"]      = apps["mat_final_price"].fillna(0)
    apps["recommendations_total"] = apps["recommendations_total"].fillna(0)
    apps["metacritic_score"]     = apps["metacritic_score"].fillna(0)
    apps["is_free"]              = apps["is_free"].fillna(False)
    apps["appid"]                = apps["appid"].astype(int)

    apps.to_csv(DATA_DIR / "apps_clean.csv", index=False)
    print(f"\n  Итого игр после очистки: {len(apps):,}")
    print(f"  → data/apps_clean.csv")

    return apps


# =============================================================================
# STEP 2 — ОЧИСТКА ЖАНРОВ И КАТЕГОРИЙ
# =============================================================================

def clean_genres_categories(apps_clean: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 2 — Очистка жанров и категорий")
    print("="*60)

    valid_appids = set(apps_clean["appid"].astype(int))

    # ── Жанры ─────────────────────────────────────────────────────────────────
    app_genres = pd.read_csv(CSV_DIR / "application_genres.csv")
    genres_ref = pd.read_csv(CSV_DIR / "genres.csv")

    # Оставляем только жанры для валидных игр
    app_genres = app_genres[app_genres["appid"].isin(valid_appids)]

    # Нормализуем названия жанров — убираем не-ASCII (жанры на других языках)
    genres_ref["name_clean"] = genres_ref["name"].apply(
        lambda x: x.encode("ascii", errors="ignore").decode().strip()
        if isinstance(x, str) else ""
    )
    # Убираем жанры с пустым названием после нормализации
    genres_ref = genres_ref[genres_ref["name_clean"] != ""]

    merged = app_genres.merge(genres_ref, left_on="genre_id", right_on="id", how="inner")
    genres_agg = (
        merged.groupby("appid")["name_clean"]
        .apply(lambda x: " ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"name_clean": "genres_text"})
    )

    print(f"  Жанров в итоге: {genres_ref['name_clean'].nunique()} уникальных")
    genres_agg.to_csv(DATA_DIR / "genres_agg.csv", index=False)
    print(f"  → data/genres_agg.csv")

    # ── Категории ─────────────────────────────────────────────────────────────
    try:
        app_cats = pd.read_csv(CSV_DIR / "application_categories.csv")
        cats_ref = pd.read_csv(CSV_DIR / "categories.csv")
        name_col = "description" if "description" in cats_ref.columns else cats_ref.columns[-1]

        app_cats = app_cats[app_cats["appid"].isin(valid_appids)]
        cats_ref["name_clean"] = cats_ref[name_col].apply(
            lambda x: x.encode("ascii", errors="ignore").decode().strip()
            if isinstance(x, str) else ""
        )
        cats_ref = cats_ref[cats_ref["name_clean"] != ""]

        merged_c = app_cats.merge(cats_ref, left_on="category_id", right_on="id", how="inner")
        cats_agg = (
            merged_c.groupby("appid")["name_clean"]
            .apply(lambda x: " ".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"name_clean": "categories_text"})
        )
        cats_agg.to_csv(DATA_DIR / "categories_agg.csv", index=False)
        print(f"  → data/categories_agg.csv")
    except Exception as e:
        print(f"  Категории: {e}")

    return genres_agg


# =============================================================================
# STEP 3 — ОЧИСТКА ОТЗЫВОВ
# =============================================================================

def clean_reviews(apps_clean: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 3 — Очистка reviews.csv")
    print("="*60)

    valid_appids = set(apps_clean["appid"].astype(int))

    print("  Загрузка...")
    reviews = pd.read_csv(CSV_DIR / "reviews.csv", usecols=[
        "appid", "author_steamid", "voted_up",
        "author_playtime_forever", "language", "weighted_vote_score"
    ])
    n0 = len(reviews)
    print(f"  Исходно отзывов: {n0:,}")

    reviews["appid"]          = reviews["appid"].astype(int)
    reviews["author_steamid"] = reviews["author_steamid"].astype(str)

    # ── Оставляем только отзывы на игры из нашего датасета ───────────────────
    n = len(reviews)
    reviews = reviews[reviews["appid"].isin(valid_appids)]
    print_stat("Игры из датасета", n, len(reviews))

    # ── Убираем отзывы с нулевым временем игры ────────────────────────────────
    # Человек без часов не может честно оценить игру
    n = len(reviews)
    reviews = reviews[reviews["author_playtime_forever"] > 0]
    print_stat("playtime > 0 минут", n, len(reviews))

    # ── Минимальное время игры ──────────────────────────────────────────────
    n = len(reviews)
    reviews = reviews[reviews["author_playtime_forever"] >= MIN_PLAYTIME_MINUTES]
    print_stat(f"playtime ≥ {MIN_PLAYTIME_MINUTES} минут", n, len(reviews))

    # ── Убираем ботов и аномальных пользователей (> 50k часов в одной игре) ──
    # 50k часов = ~5.7 лет непрерывной игры → явная аномалия
    n = len(reviews)
    reviews = reviews[reviews["author_playtime_forever"] <= MAX_PLAYTIME_MINUTES]  # минуты
    print_stat("playtime ≤ 50k часов", n, len(reviews))

    # ── Убираем дубликаты (один пользователь — одна игра) ────────────────────
    n = len(reviews)
    reviews = reviews.drop_duplicates(subset=["author_steamid", "appid"], keep="last")
    print_stat("Уникальные user-item пары", n, len(reviews))

    # ── Итеративная k-core фильтрация ────────────────────────────────────────
    # После удаления редких игр у части пользователей тоже может стать мало отзывов.
    # Поэтому фильтрацию делаем в цикле до стабилизации.
    print(f"  K-core фильтрация: users ≥ {MIN_USER_REVIEWS}, games ≥ {MIN_GAME_REVIEWS}")
    before_kcore = len(reviews)
    iteration = 0
    while True:
        iteration += 1
        n_before_iter = len(reviews)

        user_counts = reviews["author_steamid"].value_counts()
        active_users = user_counts[user_counts >= MIN_USER_REVIEWS].index
        reviews = reviews[reviews["author_steamid"].isin(active_users)]

        game_counts = reviews["appid"].value_counts()
        active_games = game_counts[game_counts >= MIN_GAME_REVIEWS].index
        reviews = reviews[reviews["appid"].isin(active_games)]

        print(f"    итерация {iteration}: {n_before_iter:,} → {len(reviews):,}")

        if len(reviews) == n_before_iter:
            break

    print_stat("После k-core фильтрации", before_kcore, len(reviews))

    # ── Финальные поля ────────────────────────────────────────────────────────
    reviews["rating"]          = reviews["voted_up"].astype(int).astype(float)
    reviews["playtime_hours"]  = (reviews["author_playtime_forever"] / 60).round(2)

    reviews = reviews[[
        "author_steamid", "appid", "rating",
        "playtime_hours", "weighted_vote_score"
    ]]

    reviews.to_csv(DATA_DIR / "reviews_clean.csv", index=False)

    print(f"\n  Итого отзывов: {len(reviews):,}")
    print(f"  Пользователей: {reviews['author_steamid'].nunique():,}")
    print(f"  Игр:           {reviews['appid'].nunique():,}")
    print(f"  Положительных: {reviews['rating'].mean()*100:.1f}%")
    print(f"  → data/reviews_clean.csv")

    return reviews


# =============================================================================
# STEP 4 — ФИЛЬТРАЦИЯ ЭМБЕДДИНГОВ
# =============================================================================

def filter_embeddings(apps_clean: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 4 — Фильтрация эмбеддингов")
    print("="*60)

    emb_map = pd.read_csv(EMBEDDINGS_DIR / "applications_embedding_map.csv")
    emb_map["appid"] = emb_map["appid"].astype(int)
    print(f"  Эмбеддингов в map: {len(emb_map):,}")

    # Загружаем сырые эмбеддинги
    print("  Загрузка embeddings (может занять время)...")
    raw = np.fromfile(
        EMBEDDINGS_DIR / "applications_embeddings.npy",
        dtype=np.float32
    )
    n   = len(emb_map)
    dim = len(raw) // n
    embeddings = raw[:n * dim].reshape(n, dim)
    print(f"  Загружено: {embeddings.shape}")

    # Оставляем только эмбеддинги для игр которые прошли очистку
    valid_appids = set(apps_clean["appid"].astype(int))
    mask         = emb_map["appid"].isin(valid_appids)
    emb_map_clean = emb_map[mask].reset_index(drop=True)
    old_indices   = emb_map_clean["vector_index"].values
    embeddings_clean = embeddings[old_indices]

    # Переиндексируем — новые индексы 0..N
    emb_map_clean["vector_index"] = np.arange(len(emb_map_clean))

    print_stat("Эмбеддинги для чистых игр",
               len(emb_map), len(emb_map_clean))

    # L2-нормализация — стандарт для cosine similarity
    print("  L2-нормализация...")
    norms = np.linalg.norm(embeddings_clean, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # защита от нулевых векторов
    embeddings_clean = embeddings_clean / norms

    # Сохраняем
    np.save(DATA_DIR / "embeddings_clean.npy", embeddings_clean)
    emb_map_clean.to_csv(DATA_DIR / "embedding_map_clean.csv", index=False)

    print(f"  Итого эмбеддингов: {embeddings_clean.shape}")
    print(f"  → data/embeddings_clean.npy")
    print(f"  → data/embedding_map_clean.csv")

    return embeddings_clean, emb_map_clean


# =============================================================================
# STEP 5 — ИТОГОВАЯ СТАТИСТИКА
# =============================================================================

def print_summary(apps, reviews, embeddings):
    print("\n" + "="*60)
    print("ИТОГ — Данные готовы для обучения")
    print("="*60)

    print(f"""
  Файл                        Записей
  ─────────────────────────── ────────
  data/apps_clean.csv         {len(apps):>8,}  игр
  data/reviews_clean.csv      {len(reviews):>8,}  отзывов
  data/embeddings_clean.npy   {embeddings.shape[0]:>8,}  векторов ({embeddings.shape[1]}D)

  Теперь запускай:
      python train.py
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    apps    = clean_apps()
    genres  = clean_genres_categories(apps)
    reviews = clean_reviews(apps)
    emb, _  = filter_embeddings(apps)

    print_summary(apps, reviews, emb)
    print(f"Время очистки: {time.time()-t0:.1f}с")