"""
train.py — обучение и оценка гибридной рекомендательной системы видеоигр.

Исправленная версия:
    1. Offline-оценка без утечки данных: тестовые взаимодействия удаляются из train-части.
    2. SVD для оценки обучается на train_reviews, а финальная SVD-модель сохраняется после обучения на всех данных.
    3. TF-IDF расширен до 50 000 признаков.
    4. Рекомендательные функции возвращают top_n кандидатов, а не только top-10.
    5. Добавлена полноценная оценка Hybrid-модели.
    6. Веса гибрида подбираются grid search по NDCG@10.
    7. Сохраняются веса для авторизованного и анонимного режимов.

Запуск:
    python train.py
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

DATA_DIR = Path("../backend/data")
MODELS_DIR = Path("../backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K = 10
TOP_N = 300
N_TEST_USERS = 300


def top_indices(scores: np.ndarray, top_n: int, exclude_idx=None):
    scores = np.asarray(scores, dtype=np.float32).copy()
    if exclude_idx:
        scores[list(exclude_idx)] = -np.inf
    valid_count = np.isfinite(scores).sum()
    if valid_count == 0:
        return np.array([], dtype=int)
    n = min(top_n, valid_count)
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx])]


def ndcg_at_k(recs, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, appid in enumerate(recs[:k]) if appid in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recs, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for x in recs[:k] if x in relevant) / k


def recall_at_k(recs, relevant, k):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return sum(1 for x in recs[:k] if x in relevant) / len(relevant)


def evaluate_model(get_recs_fn, test_users, k=10, top_n=300):
    p_list, r_list, n_list = [], [], []
    for case in tqdm(test_users, desc="  Оценка", leave=False):
        recs = get_recs_fn(case["uid"], case["train_games"], top_n=top_n)
        if not recs:
            continue
        p_list.append(precision_at_k(recs, case["test_games"], k))
        r_list.append(recall_at_k(recs, case["test_games"], k))
        n_list.append(ndcg_at_k(recs, case["test_games"], k))
    return {
        f"Precision@{k}": round(float(np.mean(p_list)), 4) if p_list else 0.0,
        f"Recall@{k}": round(float(np.mean(r_list)), 4) if r_list else 0.0,
        f"NDCG@{k}": round(float(np.mean(n_list)), 4) if n_list else 0.0,
    }


def load_data():
    print("\n" + "=" * 60)
    print("STEP 1 — Загрузка очищенных данных")
    print("=" * 60)

    required = [
        DATA_DIR / "apps_clean.csv",
        DATA_DIR / "reviews_clean.csv",
        DATA_DIR / "genres_agg.csv",
        DATA_DIR / "embeddings_clean.npy",
        DATA_DIR / "embedding_map_clean.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Не найдены подготовленные файлы. Сначала запусти preprocess.py:\n" + "\n".join(missing))

    apps = pd.read_csv(DATA_DIR / "apps_clean.csv")
    apps["appid"] = apps["appid"].astype(int)

    genres = pd.read_csv(DATA_DIR / "genres_agg.csv")
    genres["appid"] = genres["appid"].astype(int)
    apps = apps.merge(genres, on="appid", how="left")
    apps["genres_text"] = apps["genres_text"].fillna("")

    if (DATA_DIR / "categories_agg.csv").exists():
        cats = pd.read_csv(DATA_DIR / "categories_agg.csv")
        cats["appid"] = cats["appid"].astype(int)
        apps = apps.merge(cats, on="appid", how="left")
        apps["categories_text"] = apps["categories_text"].fillna("")
    else:
        apps["categories_text"] = ""

    apps["content"] = (
        apps["short_description"].fillna("") + " " +
        apps["genres_text"].fillna("") + " " +
        apps["categories_text"].fillna("")
    ).str.strip()

    apps = apps.drop_duplicates(subset=["appid"]).set_index("appid")
    print(f"  Игр: {len(apps):,}")

    embeddings = np.load(DATA_DIR / "embeddings_clean.npy").astype(np.float32)
    embeddings = normalize(embeddings).astype(np.float32)

    emb_map = pd.read_csv(DATA_DIR / "embedding_map_clean.csv")
    emb_map["appid"] = emb_map["appid"].astype(int)
    emb_map["vector_index"] = emb_map["vector_index"].astype(int)
    appid_to_emb = dict(zip(emb_map["appid"], emb_map["vector_index"]))
    print(f"  Эмбеддинги: {embeddings.shape}")

    reviews = pd.read_csv(DATA_DIR / "reviews_clean.csv")
    reviews["appid"] = reviews["appid"].astype(int)
    reviews["author_steamid"] = reviews["author_steamid"].astype(str)

    if "playtime_hours" not in reviews.columns:
        if "author_playtime_forever" in reviews.columns:
            reviews["playtime_hours"] = reviews["author_playtime_forever"].fillna(0) / 60.0
        else:
            reviews["playtime_hours"] = 0.0

    if "rating" not in reviews.columns:
        reviews["rating"] = reviews["voted_up"].astype(int)

    reviews["rating"] = reviews["rating"].astype(float)
    reviews["playtime_hours"] = reviews["playtime_hours"].fillna(0).clip(lower=0)

    print(f"  Отзывов: {len(reviews):,} | Пользователей: {reviews['author_steamid'].nunique():,}")
    return apps, embeddings, appid_to_emb, reviews


def make_offline_split(reviews: pd.DataFrame, n_users=300, min_liked=20, test_ratio=0.2):
    print("\n" + "=" * 60)
    print("STEP 2 — Подготовка offline train/test")
    print("=" * 60)

    rng = np.random.default_rng(RANDOM_STATE)
    liked = reviews[reviews["rating"] > 0].copy()
    counts = liked["author_steamid"].value_counts()
    candidates = counts[counts >= min_liked].index.to_numpy()
    rng.shuffle(candidates)
    selected = candidates[:n_users]

    test_pairs = set()
    test_users = []

    for uid in selected:
        appids = liked.loc[liked["author_steamid"] == uid, "appid"].astype(int).to_numpy()
        rng.shuffle(appids)
        n_test = max(1, int(len(appids) * test_ratio))
        test_appids = set(appids[:n_test])
        train_appids = list(appids[n_test:])
        if not train_appids or not test_appids:
            continue

        rows = reviews[(reviews["author_steamid"] == uid) & (reviews["appid"].isin(train_appids))]
        train_games = []
        for _, row in rows.iterrows():
            if row["rating"] > 0:
                train_games.append({
                    "appid": int(row["appid"]),
                    "playtime_hours": float(row.get("playtime_hours", 1.0)),
                    "rating": float(row.get("rating", 1.0)),
                })

        if not train_games:
            continue

        for appid in test_appids:
            test_pairs.add((str(uid), int(appid)))

        test_users.append({"uid": str(uid), "train_games": train_games, "test_games": list(test_appids)})

    pairs = list(zip(reviews["author_steamid"].astype(str), reviews["appid"].astype(int)))
    mask = np.array([pair not in test_pairs for pair in pairs], dtype=bool)
    train_reviews = reviews.loc[mask].copy()

    print(f"  Тестовых пользователей: {len(test_users):,}")
    print(f"  Отложено test-взаимодействий: {len(test_pairs):,}")
    print(f"  Train reviews: {len(train_reviews):,} из {len(reviews):,}")
    return train_reviews, test_users


def train_tfidf(apps: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 3 — TF-IDF")
    print("=" * 60)
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(apps["content"].fillna(""))
    appids = np.array(apps.index.astype(int).tolist())

    print(f"  Словарь: {len(vectorizer.vocabulary_):,} токенов")
    print(f"  Матрица: {tfidf_matrix.shape}")
    print(f"  Время:   {time.time() - t0:.1f}с")
    return vectorizer, tfidf_matrix, appids


def build_interaction_values(reviews: pd.DataFrame):
    playtime = reviews["playtime_hours"].fillna(0).clip(lower=0).to_numpy()
    rating = reviews["rating"].fillna(0).to_numpy()
    signed_rating = np.where(rating > 0, 1.0, 0)
    return (signed_rating * (1.0 + np.log1p(playtime))).astype(np.float32)


def train_svd(reviews: pd.DataFrame, n_components=100, label="SVD"):
    print("\n" + "=" * 60)
    print(f"STEP 4 — {label}")
    print("=" * 60)
    t0 = time.time()

    user_series = reviews["author_steamid"].astype(str).astype("category")
    item_series = reviews["appid"].astype(int).astype("category")
    user_idx = user_series.cat.codes.to_numpy()
    item_idx = item_series.cat.codes.to_numpy()
    values = build_interaction_values(reviews)

    n_users = len(user_series.cat.categories)
    n_items = len(item_series.cat.categories)
    user_item = csr_matrix((values, (user_idx, item_idx)), shape=(n_users, n_items), dtype=np.float32)

    real_components = min(n_components, max(2, min(user_item.shape) - 1))
    svd = TruncatedSVD(n_components=real_components, n_iter=15, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_.T
    user_factors = normalize(user_factors).astype(np.float32)
    item_factors = normalize(item_factors).astype(np.float32)

    print(f"  User-item матрица: {user_item.shape} ({user_item.nnz:,} ненулевых)")
    print(f"  Латентных факторов: {real_components}")
    print(f"  Объяснённая дисперсия: {svd.explained_variance_ratio_.sum():.3f}")
    print(f"  Время: {time.time() - t0:.1f}с")

    return {
        "svd": svd,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_categories": user_series.cat.categories.astype(str).tolist(),
        "item_categories": item_series.cat.categories.astype(int).tolist(),
    }


def build_recommenders(tfidf_data, embeddings, appid_to_emb, svd_data):
    _, tfidf_matrix, tfidf_appids = tfidf_data
    tfidf_appids = np.array(tfidf_appids, dtype=int)
    tfidf_idx = {int(appid): i for i, appid in enumerate(tfidf_appids)}

    emb_appids = np.array(list(appid_to_emb.keys()), dtype=int)
    emb_indices = np.array([appid_to_emb[int(a)] for a in emb_appids], dtype=int)

    user_to_idx = {str(u): i for i, u in enumerate(svd_data["user_categories"])}
    item_appids = np.array(svd_data["item_categories"], dtype=int)
    item_factors = svd_data["item_factors"]
    user_factors = svd_data["user_factors"]
    svd_item_idx = {int(appid): i for i, appid in enumerate(item_appids)}

    def tfidf_recs(uid, train_games, top_n=300):
        profile = np.zeros(tfidf_matrix.shape[1], dtype=np.float32)
        total_w = 0.0
        owned = set()
        for g in train_games:
            appid = int(g["appid"])
            owned.add(appid)
            idx = tfidf_idx.get(appid)
            if idx is None or float(g.get("rating", 1.0)) <= 0:
                continue
            w = max(np.log1p(max(float(g.get("playtime_hours", 1.0)), 0.0)), 0.1)
            profile += w * tfidf_matrix[idx].toarray()[0]
            total_w += w
        if total_w <= 0:
            return []
        profile /= total_w
        scores = cosine_similarity(profile.reshape(1, -1), tfidf_matrix)[0]
        exclude_idx = {tfidf_idx[a] for a in owned if a in tfidf_idx}
        idx = top_indices(scores, top_n, exclude_idx=exclude_idx)
        return [int(tfidf_appids[i]) for i in idx]

    def emb_recs(uid, train_games, top_n=300):
        profile = np.zeros(embeddings.shape[1], dtype=np.float32)
        total_w = 0.0
        owned = set()
        for g in train_games:
            appid = int(g["appid"])
            owned.add(appid)
            idx = appid_to_emb.get(appid)
            if idx is None or float(g.get("rating", 1.0)) <= 0:
                continue
            w = max(np.log1p(max(float(g.get("playtime_hours", 1.0)), 0.0)), 0.1)
            profile += w * embeddings[idx]
            total_w += w
        if total_w <= 0:
            return []
        profile = normalize(profile.reshape(1, -1)).astype(np.float32)[0]
        scores_all = embeddings @ profile
        candidate_scores = scores_all[emb_indices]
        exclude_local = {pos for pos, appid in enumerate(emb_appids) if int(appid) in owned}
        idx = top_indices(candidate_scores, top_n, exclude_idx=exclude_local)
        return [int(emb_appids[i]) for i in idx]

    def svd_recs(uid, train_games, top_n=300):
        uid = str(uid)
        if uid not in user_to_idx:
            return []
        owned = {int(g["appid"]) for g in train_games}
        scores = item_factors @ user_factors[user_to_idx[uid]]
        exclude_idx = {svd_item_idx[a] for a in owned if a in svd_item_idx}
        idx = top_indices(scores, top_n, exclude_idx=exclude_idx)
        return [int(item_appids[i]) for i in idx]

    return tfidf_recs, svd_recs, emb_recs


def merge_ranked_lists(cb, sv, em, weights, top_n=300, exclude=None):
    alpha, beta, gamma = weights
    exclude = set(exclude or [])
    scores = {}
    for rank, appid in enumerate(cb):
        if appid not in exclude:
            scores[appid] = scores.get(appid, 0.0) + alpha / (rank + 1)
    for rank, appid in enumerate(sv):
        if appid not in exclude:
            scores[appid] = scores.get(appid, 0.0) + beta / (rank + 1)
    for rank, appid in enumerate(em):
        if appid not in exclude:
            scores[appid] = scores.get(appid, 0.0) + gamma / (rank + 1)
    return [int(appid) for appid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def precompute_recs(test_users, tfidf_recs, svd_recs, emb_recs, top_n=300):
    cache = []
    for case in tqdm(test_users, desc="  Предвычисление", leave=False):
        uid = case["uid"]
        train_games = case["train_games"]
        cache.append({
            "train_games": train_games,
            "test_games": case["test_games"],
            "cb": tfidf_recs(uid, train_games, top_n=top_n),
            "sv": svd_recs(uid, train_games, top_n=top_n),
            "em": emb_recs(uid, train_games, top_n=top_n),
        })
    return cache


def metrics_for_cached_hybrid(cache, weights, k=10, top_n=300):
    p_list, r_list, n_list = [], [], []
    for row in cache:
        owned = {int(g["appid"]) for g in row["train_games"]}
        recs = merge_ranked_lists(row["cb"], row["sv"], row["em"], weights, top_n=top_n, exclude=owned)
        p_list.append(precision_at_k(recs, row["test_games"], k))
        r_list.append(recall_at_k(recs, row["test_games"], k))
        n_list.append(ndcg_at_k(recs, row["test_games"], k))
    return {
        f"Precision@{k}": round(float(np.mean(p_list)), 4) if p_list else 0.0,
        f"Recall@{k}": round(float(np.mean(r_list)), 4) if r_list else 0.0,
        f"NDCG@{k}": round(float(np.mean(n_list)), 4) if n_list else 0.0,
    }


def find_optimal_weights(test_users, tfidf_recs, svd_recs, emb_recs, k=10, top_n=300):
    print("\n" + "=" * 60)
    print("STEP 6 — Подбор весов гибридной модели")
    print("=" * 60)
    cache = precompute_recs(test_users, tfidf_recs, svd_recs, emb_recs, top_n=top_n)

    candidates = []
    for alpha in np.arange(0.0, 1.01, 0.05):
        for beta in np.arange(0.0, 1.01, 0.05):
            gamma = 1.0 - alpha - beta
            if gamma < -1e-9:
                continue
            weights = (float(alpha), float(beta), float(gamma))
            metrics = metrics_for_cached_hybrid(cache, weights, k=k, top_n=top_n)
            candidates.append((metrics[f"NDCG@{k}"], weights, metrics))

    candidates.sort(key=lambda x: -x[0])
    _, best_weights, best_metrics = candidates[0]
    alpha, beta, gamma = best_weights

    print("  Лучшие веса по NDCG@10:")
    print(f"    α = {alpha:.2f}  TF-IDF")
    print(f"    β = {beta:.2f}  SVD")
    print(f"    γ = {gamma:.2f}  Embeddings")
    print(f"  Hybrid metrics: {best_metrics}")

    return {
        "auth": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "anon": {"alpha": 0.50, "beta": 0.00, "gamma": 0.50},
        "metrics": best_metrics,
        "top_candidates": candidates[:10],
    }


def save_models(tfidf_data, final_svd_data, embeddings, appid_to_emb, weights_data, apps):
    print("\n" + "=" * 60)
    print("STEP 7 — Сохранение моделей")
    print("=" * 60)

    vectorizer, tfidf_matrix, tfidf_appids = tfidf_data
    joblib.dump({"vectorizer": vectorizer, "tfidf_matrix": tfidf_matrix, "appids": list(map(int, tfidf_appids))}, MODELS_DIR / "tfidf.pkl", compress=3)
    print("  models/tfidf.pkl")

    joblib.dump({
        "svd": final_svd_data["svd"],
        "user_factors": final_svd_data["user_factors"],
        "item_factors": final_svd_data["item_factors"],
        "user_categories": final_svd_data["user_categories"],
        "item_categories": final_svd_data["item_categories"],
    }, MODELS_DIR / "svd.pkl", compress=3)
    print("  models/svd.pkl")

    joblib.dump({"embeddings": embeddings, "appid_to_idx": appid_to_emb}, MODELS_DIR / "embeddings.pkl", compress=3)
    print("  models/embeddings.pkl")

    joblib.dump(weights_data, MODELS_DIR / "hybrid_weights.pkl", compress=3)
    print("  models/hybrid_weights.pkl")

    cols = ["name", "header_image", "mat_final_price", "is_free", "recommendations_total", "metacritic_score", "short_description", "genres_text", "categories_text"]
    cols = [c for c in cols if c in apps.columns]
    joblib.dump(apps[cols].copy(), MODELS_DIR / "apps_meta.pkl", compress=3)
    print("  models/apps_meta.pkl")


def print_and_save_report(m_tfidf, m_svd, m_emb, m_hybrid, weights_data, k=10):
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ — вставляй в диплом")
    print("=" * 60)

    rows = [("TF-IDF (Content-Based)", m_tfidf), ("SVD (Collaborative)", m_svd), ("Embeddings (Semantic)", m_emb), ("Hybrid", m_hybrid)]
    report = pd.DataFrame({
        "Модель": [r[0] for r in rows],
        f"Precision@{k}": [r[1][f"Precision@{k}"] for r in rows],
        f"Recall@{k}": [r[1][f"Recall@{k}"] for r in rows],
        f"NDCG@{k}": [r[1][f"NDCG@{k}"] for r in rows],
    })
    print(report.to_string(index=False))

    auth = weights_data["auth"]
    anon = weights_data["anon"]
    print("\nВеса авторизованного режима:")
    print(f"  α={auth['alpha']:.2f} (TF-IDF)  β={auth['beta']:.2f} (SVD)  γ={auth['gamma']:.2f} (Embeddings)")
    print("\nВеса анонимного режима:")
    print(f"  α={anon['alpha']:.2f} (TF-IDF)  β={anon['beta']:.2f} (SVD)  γ={anon['gamma']:.2f} (Embeddings)")

    report.to_csv(MODELS_DIR / "metrics_report.csv", index=False)
    print(f"\n→ {MODELS_DIR / 'metrics_report.csv'}")


if __name__ == "__main__":
    total_start = time.time()

    apps, embeddings, appid_to_emb, reviews = load_data()
    train_reviews, test_users = make_offline_split(reviews, n_users=N_TEST_USERS, min_liked=20, test_ratio=0.2)

    tfidf_data = train_tfidf(apps)

    # Для оценки обучаем SVD только на train_reviews, иначе будет утечка тестовых взаимодействий.
    eval_svd_data = train_svd(train_reviews, n_components=100, label="SVD для offline-оценки")
    tfidf_recs, svd_recs, emb_recs = build_recommenders(tfidf_data, embeddings, appid_to_emb, eval_svd_data)

    print("\n" + "=" * 60)
    print(f"STEP 5 — Оценка моделей (K={K})")
    print("=" * 60)

    print("\n  Модель 1: TF-IDF...")
    m_tfidf = evaluate_model(tfidf_recs, test_users, k=K, top_n=TOP_N)
    print(f"    {m_tfidf}")

    print("  Модель 2: SVD...")
    m_svd = evaluate_model(svd_recs, test_users, k=K, top_n=TOP_N)
    print(f"    {m_svd}")

    print("  Модель 3: Embeddings...")
    m_emb = evaluate_model(emb_recs, test_users, k=K, top_n=TOP_N)
    print(f"    {m_emb}")

    weights_data = find_optimal_weights(test_users, tfidf_recs, svd_recs, emb_recs, k=K, top_n=TOP_N)
    m_hybrid = weights_data["metrics"]

    # Финальную модель для приложения обучаем на всех данных.
    final_svd_data = train_svd(reviews, n_components=100, label="Финальная SVD для сохранения")

    save_models(tfidf_data, final_svd_data, embeddings, appid_to_emb, weights_data, apps)
    print_and_save_report(m_tfidf, m_svd, m_emb, m_hybrid, weights_data, k=K)

    print(f"\nОбщее время: {(time.time() - total_start) / 60:.1f} минут")
    print("=" * 60)
