"""
train.py — обучение гибридной рекомендательной системы.

Все зависимости — чистый Python.

Запуск:
    pip install -r requirements.txt
    python train.py

Три модели:
    1. TF-IDF           — content-based, обучаем TfidfVectorizer
    2. SVD (sklearn)    — collaborative filtering, TruncatedSVD на матрице отзывов
    3. Embeddings       — готовые векторы из датасета, baseline

Что сохраняется в models/:
    tfidf.pkl           — TfidfVectorizer + матрица
    svd.pkl             — TruncatedSVD + user/item матрицы
    embeddings.pkl      — эмбеддинги + маппинг appid→индекс
    hybrid_weights.pkl  — оптимальные веса α, β, γ
    apps_meta.pkl       — метаданные игр для UI
    metrics_report.csv  — таблица метрик для диплома
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

CSV_DIR        = Path("../dataset/steam_dataset_2025_csv")   # исходные данные
EMBEDDINGS_DIR = Path("../dataset/steam_dataset_2025_embeddings")
DATA_DIR       = Path("../backend/data")       # очищенные данные (после preprocess.py)
MODELS_DIR     = Path("../backend/models")
MODELS_DIR.mkdir(exist_ok=True)

# Проверяем что preprocess.py был запущен
if not (DATA_DIR / "apps_clean.csv").exists():
    print("⚠  Папка data/ не найдена. Сначала запусти: python preprocess.py")
    exit(1)




def _load_embeddings(path, emb_map):
    """
    Загружает эмбеддинги из файла.
    Файл не имеет numpy-заголовка — сырые float32.
    Форму берём из embedding_map: n_rows = len(map), dim = total / n_rows.
    """
    try:
        data = np.load(path, allow_pickle=False)
        print(f"  Эмбеддинги (numpy): {data.shape}")
        return data.astype(np.float32)
    except Exception:
        pass
    raw  = np.fromfile(path, dtype=np.float32)
    n    = len(emb_map)
    dim  = len(raw) // n
    data = raw[:n * dim].reshape(n, dim)
    print(f"  Эмбеддинги (raw float32): {data.shape}")
    return data

# =============================================================================
# STEP 1 — ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_data():
    print("\n" + "="*60)
    print("STEP 1 — Загрузка очищенных данных")
    print("="*60)

    # ── Игры (из preprocess.py) ───────────────────────────────────────────────
    apps = pd.read_csv(DATA_DIR / "apps_clean.csv")
    apps["appid"] = apps["appid"].astype(int)

    # ── Жанры и категории (уже агрегированы preprocess.py) ───────────────────
    genres_agg = pd.read_csv(DATA_DIR / "genres_agg.csv")
    apps = apps.merge(genres_agg, on="appid", how="left")
    apps["genres_text"] = apps["genres_text"].fillna("")

    if (DATA_DIR / "categories_agg.csv").exists():
        cats_agg = pd.read_csv(DATA_DIR / "categories_agg.csv")
        apps = apps.merge(cats_agg, on="appid", how="left")
        apps["categories_text"] = apps["categories_text"].fillna("")
    else:
        apps["categories_text"] = ""

    # ── Контент для TF-IDF ────────────────────────────────────────────────────
    apps["content"] = (
        apps["short_description"].fillna("") + " " +
        apps["genres_text"] + " " +
        apps["categories_text"]
    ).str.strip()

    apps = apps.set_index("appid")
    print(f"  Игр: {len(apps):,}")

    # ── Эмбеддинги (нормализованы и отфильтрованы preprocess.py) ─────────────
    embeddings = np.load(DATA_DIR / "embeddings_clean.npy")
    emb_map    = pd.read_csv(DATA_DIR / "embedding_map_clean.csv")
    appid_to_emb = dict(zip(emb_map["appid"].astype(int), emb_map["vector_index"].astype(int)))
    print(f"  Эмбеддинги: {embeddings.shape}")

    # ── Отзывы (очищены preprocess.py) ───────────────────────────────────────
    reviews = pd.read_csv(DATA_DIR / "reviews_clean.csv")
    reviews["appid"]          = reviews["appid"].astype(int)
    reviews["author_steamid"] = reviews["author_steamid"].astype(str)
    print(f"  Отзывов: {len(reviews):,} | Пользователей: {reviews['author_steamid'].nunique():,}")

    return apps, embeddings, appid_to_emb, reviews


# =============================================================================
# STEP 2 — TF-IDF (Content-Based)
# =============================================================================

def train_tfidf(apps: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 2 — TF-IDF (Content-Based)")
    print("="*60)

    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        ngram_range=(1, 2),   # "open world", "first person" — биграммы
        min_df=2,
        sublinear_tf=True,    # log(tf) вместо tf
    )
    tfidf_matrix = vectorizer.fit_transform(apps["content"])
    print(f"  Словарь: {len(vectorizer.vocabulary_):,} токенов")
    print(f"  Матрица: {tfidf_matrix.shape}")
    print(f"  Время:   {time.time()-t0:.1f}с")

    joblib.dump({
        "vectorizer":   vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "appids":       list(apps.index),
    }, MODELS_DIR / "tfidf.pkl", compress=3)
    print("  → models/tfidf.pkl")

    return vectorizer, tfidf_matrix, list(apps.index)


# =============================================================================
# STEP 3 — SVD из scikit-learn (Collaborative Filtering)
#
# Используем TruncatedSVD — это Matrix Factorization без C++ компилятора.
# Строим user-item матрицу: строки = пользователи, столбцы = игры,
# значения = log(1 + часы) * voted_up.
# TruncatedSVD разлагает матрицу на латентные факторы.
# =============================================================================

def train_svd(reviews: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 3 — SVD (Collaborative Filtering, scikit-learn)")
    print("="*60)

    t0 = time.time()

    # Кодируем пользователей и игры в числовые индексы
    user_series = reviews["author_steamid"].astype("category")
    item_series = reviews["appid"].astype("category")
    user_idx    = user_series.cat.codes.values
    item_idx    = item_series.cat.codes.values

    # Значение = log(1 + часы) для проголосовавших "за", -1 для "против"
    playtime   = reviews["playtime_hours"].fillna(0).values  # уже в часах после preprocess.py
    confidence = reviews["rating"].values * (1 + np.log1p(playtime))

    n_users = user_series.cat.categories.shape[0]
    n_items = item_series.cat.categories.shape[0]

    # Sparse матрица user × item
    user_item = csr_matrix(
        (confidence, (user_idx, item_idx)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    print(f"  User-item матрица: {user_item.shape}  ({user_item.nnz:,} ненулевых)")

    # TruncatedSVD — разложение на 100 латентных факторов
    # Это эквивалент SVD/ALS без необходимости компилятора
    n_components = 100
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    user_factors = svd.fit_transform(user_item)       # (n_users, 100)
    item_factors = svd.components_.T                   # (n_items, 100)

    # Нормализуем для cosine similarity
    user_factors = normalize(user_factors)
    item_factors = normalize(item_factors)

    elapsed = time.time() - t0
    print(f"  Объяснённая дисперсия: {svd.explained_variance_ratio_.sum():.3f}")
    print(f"  Время обучения: {elapsed:.1f}с")

    joblib.dump({
        "svd":            svd,
        "user_factors":   user_factors,   # (n_users, 100)
        "item_factors":   item_factors,   # (n_items, 100)
        "user_categories": user_series.cat.categories.tolist(),  # steamid → индекс
        "item_categories": item_series.cat.categories.tolist(),  # appid → индекс
    }, MODELS_DIR / "svd.pkl", compress=3)
    print("  → models/svd.pkl")

    return {
        "svd":            svd,
        "user_factors":   user_factors,
        "item_factors":   item_factors,
        "user_categories": user_series.cat.categories,
        "item_categories": item_series.cat.categories,
    }


# =============================================================================
# МЕТРИКИ
# =============================================================================

def ndcg_at_k(recs, relevant, k):
    dcg  = sum(1/np.log2(i+2) for i, x in enumerate(recs[:k]) if x in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg > 0 else 0.0

def precision_at_k(recs, relevant, k):
    return sum(1 for x in recs[:k] if x in relevant) / k

def recall_at_k(recs, relevant, k):
    if not relevant: return 0.0
    return sum(1 for x in recs[:k] if x in relevant) / len(relevant)

def evaluate_model(get_recs_fn, test_users, K=10):
    p_list, r_list, n_list = [], [], []
    for uid, liked, train in tqdm(test_users, desc="  Оценка", leave=False):
        if not liked: continue
        recs = get_recs_fn(uid, train)
        if not recs: continue
        rel = set(liked)
        p_list.append(precision_at_k(recs, rel, K))
        r_list.append(recall_at_k(recs, rel, K))
        n_list.append(ndcg_at_k(recs, rel, K))
    return {
        f"Precision@{K}": round(np.mean(p_list), 4) if p_list else 0.0,
        f"Recall@{K}":    round(np.mean(r_list), 4) if r_list else 0.0,
        f"NDCG@{K}":      round(np.mean(n_list), 4) if n_list else 0.0,
    }

def prepare_test_users(reviews, n_users=300):
    print(f"  Подготовка {n_users} тестовых пользователей...")
    liked  = reviews[reviews["rating"] == 1]
    counts = liked["author_steamid"].value_counts()
    cands  = counts[counts >= 10].index[:n_users]
    result = []
    for uid in cands:
        games = liked[liked["author_steamid"] == uid]["appid"].tolist()
        split = max(1, int(len(games) * 0.8))
        train = [{"appid": a, "playtime_hours": 10} for a in games[:split]]
        test  = games[split:]
        result.append((uid, test, train))
    return result


# =============================================================================
# STEP 4 — ОЦЕНКА ТРЁХ МОДЕЛЕЙ
# =============================================================================

def evaluate_all_models(apps, tfidf_data, embeddings, appid_to_emb, svd_data, reviews, K=10):
    print("\n" + "="*60)
    print(f"STEP 4 — Оценка моделей (K={K})")
    print("="*60)

    vectorizer, tfidf_matrix, tfidf_appids = tfidf_data
    tfidf_idx = {appid: i for i, appid in enumerate(tfidf_appids)}

    user_factors  = svd_data["user_factors"]
    item_factors  = svd_data["item_factors"]
    user_to_idx   = {str(u): i for i, u in enumerate(svd_data["user_categories"])}
    item_to_appid = {i: int(a) for i, a in enumerate(svd_data["item_categories"])}

    test_users = prepare_test_users(reviews, n_users=300)
    print(f"  Тестовых пользователей: {len(test_users)}")

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    def tfidf_recs(uid, train_games):
        profile = np.zeros(tfidf_matrix.shape[1])
        total_w = 0.0
        for g in train_games:
            appid = int(g["appid"])
            if appid not in tfidf_idx: continue
            w = max(np.log1p(g.get("playtime_hours", 1)), 0.1)
            profile += w * tfidf_matrix[tfidf_idx[appid]].toarray()[0]
            total_w += w
        if total_w == 0: return []
        profile /= total_w
        scores  = cosine_similarity([profile], tfidf_matrix)[0]
        owned   = {int(g["appid"]) for g in train_games}
        ranked  = sorted(
            ((tfidf_appids[i], scores[i]) for i in range(len(tfidf_appids))
             if tfidf_appids[i] not in owned),
            key=lambda x: -x[1]
        )
        return [a for a, _ in ranked[:K]]

    # ── SVD ───────────────────────────────────────────────────────────────────
    def svd_recs(uid, train_games):
        uid_str = str(uid)
        if uid_str not in user_to_idx: return []
        owned      = {int(g["appid"]) for g in train_games}
        u_vec      = user_factors[user_to_idx[uid_str]]  # (100,)
        scores_all = item_factors @ u_vec                 # (n_items,) — dot product
        ranked     = sorted(
            ((item_to_appid[i], float(scores_all[i]))
             for i in range(len(item_to_appid))
             if item_to_appid[i] not in owned),
            key=lambda x: -x[1]
        )
        return [a for a, _ in ranked[:K]]

    # ── Embeddings ────────────────────────────────────────────────────────────
    def emb_recs(uid, train_games):
        dim     = embeddings.shape[1]
        profile = np.zeros(dim, dtype=np.float32)
        total_w = 0.0
        for g in train_games:
            appid = int(g["appid"])
            if appid not in appid_to_emb: continue
            w = max(np.log1p(g.get("playtime_hours", 1)), 0.1)
            profile += w * embeddings[appid_to_emb[appid]]
            total_w += w
        if total_w == 0: return []
        profile /= total_w
        scores  = cosine_similarity([profile], embeddings)[0]
        owned   = {int(g["appid"]) for g in train_games}
        emb_ids = list(appid_to_emb.keys())
        ranked  = sorted(
            ((emb_ids[i], float(scores[appid_to_emb[emb_ids[i]]]))
             for i in range(len(emb_ids))
             if emb_ids[i] not in owned),
            key=lambda x: -x[1]
        )
        return [a for a, _ in ranked[:K]]

    print("\n  Модель 1: TF-IDF...")
    m_tfidf = evaluate_model(tfidf_recs, test_users, K)
    print(f"    {m_tfidf}")

    print("  Модель 2: SVD...")
    m_svd = evaluate_model(svd_recs, test_users, K)
    print(f"    {m_svd}")

    print("  Модель 3: Embeddings...")
    m_emb = evaluate_model(emb_recs, test_users, K)
    print(f"    {m_emb}")

    return m_tfidf, m_svd, m_emb, test_users, tfidf_recs, svd_recs, emb_recs


# =============================================================================
# STEP 5 — ПОДБОР ВЕСОВ ГИБРИДА (Nelder-Mead)
# =============================================================================

def find_optimal_weights(test_users, tfidf_recs, svd_recs, emb_recs, K=10):
    print("\n" + "="*60)
    print("STEP 5 — Подбор оптимальных весов (Nelder-Mead)")
    print("="*60)

    # Предвычисляем топ-200 от каждой модели для каждого пользователя — один раз
    # Это на порядок быстрее чем считать внутри objective каждый раз
    print("  Предвычисление рекомендаций...")
    cache = []
    for uid, liked, train in tqdm(test_users[:50], desc="  Кэш", leave=False):
        cb  = tfidf_recs(uid, train)
        sv  = svd_recs(uid, train)
        em  = emb_recs(uid, train)
        cache.append((liked, train, cb, sv, em))

    def hybrid_ndcg(raw_w):
        w = np.abs(raw_w) / (np.sum(np.abs(raw_w)) + 1e-9)
        a, b, g = w
        ndcgs = []
        for liked, train, cb, sv, em in cache:
            owned  = {int(x["appid"]) for x in train}
            scores = {}
            for rank, appid in enumerate(cb):
                scores[appid] = scores.get(appid, 0) + a / (rank + 1)
            for rank, appid in enumerate(sv):
                scores[appid] = scores.get(appid, 0) + b / (rank + 1)
            for rank, appid in enumerate(em):
                scores[appid] = scores.get(appid, 0) + g / (rank + 1)
            ranked = sorted(
                ((ap, sc) for ap, sc in scores.items() if ap not in owned),
                key=lambda x: -x[1]
            )
            recs = [ap for ap, _ in ranked[:K]]
            ndcgs.append(ndcg_at_k(recs, set(liked), K))
        return -np.mean(ndcgs)

    print("  Оптимизация весов...")
    result  = minimize(hybrid_ndcg, [1/3, 1/3, 1/3], method="Nelder-Mead",
                       options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-3})
    raw     = np.abs(result.x)
    weights = raw / raw.sum()
    a, b, g = weights

    print(f"\n  Оптимальные веса:")
    print(f"    α = {a:.3f}  (TF-IDF)")
    print(f"    β = {b:.3f}  (SVD)")
    print(f"    γ = {g:.3f}  (Embeddings)")
    return float(a), float(b), float(g)


# =============================================================================
# STEP 6 — СОХРАНЕНИЕ
# =============================================================================

def save_all(tfidf_data, svd_data, embeddings, appid_to_emb,
             alpha, beta, gamma, apps):
    print("\n" + "="*60)
    print("STEP 6 — Сохранение моделей")
    print("="*60)

    vectorizer, tfidf_matrix, tfidf_appids = tfidf_data

    joblib.dump({
        "vectorizer":   vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "appids":       tfidf_appids,
    }, MODELS_DIR / "tfidf.pkl", compress=3)
    print("  models/tfidf.pkl")

    joblib.dump({
        "svd":             svd_data["svd"],
        "user_factors":    svd_data["user_factors"],
        "item_factors":    svd_data["item_factors"],
        "user_categories": list(svd_data["user_categories"]),
        "item_categories": list(svd_data["item_categories"]),
    }, MODELS_DIR / "svd.pkl", compress=3)
    print("  models/svd.pkl")

    joblib.dump({
        "embeddings":   embeddings,
        "appid_to_idx": appid_to_emb,
    }, MODELS_DIR / "embeddings.pkl", compress=3)
    print("  models/embeddings.pkl")

    joblib.dump(
        {"alpha": alpha, "beta": beta, "gamma": gamma},
        MODELS_DIR / "hybrid_weights.pkl"
    )
    print("  models/hybrid_weights.pkl")

    cols = [c for c in ["name", "header_image", "mat_final_price", "is_free",
                        "recommendations_total", "metacritic_score",
                        "short_description"] if c in apps.columns]
    joblib.dump(apps[cols].copy(), MODELS_DIR / "apps_meta.pkl", compress=3)
    print("  models/apps_meta.pkl")
    print("\n  Все модели сохранены!")


# =============================================================================
# ИТОГОВЫЙ ОТЧЁТ
# =============================================================================

def print_report(m_tfidf, m_svd, m_emb, alpha, beta, gamma):
    print("\n" + "="*60)
    print("ИТОГОВЫЙ ОТЧЁТ — вставляй в диплом")
    print("="*60)
    K = list(m_tfidf.keys())[0].split("@")[1]
    report = pd.DataFrame({
        "Модель": ["TF-IDF (Content-Based)", "SVD (Collaborative)", "Embeddings (Semantic)"],
        f"Precision@{K}": [m_tfidf[f"Precision@{K}"], m_svd[f"Precision@{K}"], m_emb[f"Precision@{K}"]],
        f"Recall@{K}":    [m_tfidf[f"Recall@{K}"],    m_svd[f"Recall@{K}"],    m_emb[f"Recall@{K}"]],
        f"NDCG@{K}":      [m_tfidf[f"NDCG@{K}"],      m_svd[f"NDCG@{K}"],      m_emb[f"NDCG@{K}"]],
    })
    print(report.to_string(index=False))
    print(f"\nОптимальные веса гибрида:")
    print(f"  α={alpha:.3f} (TF-IDF)  β={beta:.3f} (SVD)  γ={gamma:.3f} (Embeddings)")
    report.to_csv(MODELS_DIR / "metrics_report.csv", index=False)
    print(f"\n→ models/metrics_report.csv")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t_start = time.time()

    apps, embeddings, appid_to_emb, reviews = load_data()
    tfidf_data = train_tfidf(apps)
    svd_data   = train_svd(reviews)

    m_tfidf, m_svd, m_emb, test_users, tfidf_recs, svd_recs, emb_recs = evaluate_all_models(
        apps, tfidf_data, embeddings, appid_to_emb, svd_data, reviews
    )

    alpha, beta, gamma = find_optimal_weights(test_users, tfidf_recs, svd_recs, emb_recs)

    save_all(tfidf_data, svd_data, embeddings, appid_to_emb, alpha, beta, gamma, apps)
    print_report(m_tfidf, m_svd, m_emb, alpha, beta, gamma)

    print(f"\nОбщее время: {(time.time()-t_start)/60:.1f} минут")
    print("="*60)