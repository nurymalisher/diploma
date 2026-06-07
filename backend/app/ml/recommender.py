import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Путь к моделям относительно корня проекта
MODELS_DIR = Path("models")


class HybridRecommender:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_appids = []
        self.tfidf_idx = {}
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_to_idx = {}
        self.item_to_appid = {}
        self.embeddings = None
        self.appid_to_emb = {}
        self.auth_alpha = 0.33
        self.auth_beta = 0.33
        self.auth_gamma = 0.34

        self.anon_alpha = 0.50
        self.anon_beta = 0.00
        self.anon_gamma = 0.50
        self.apps_meta = None

    def load(self):
        d = joblib.load(MODELS_DIR / "tfidf.pkl")
        self.vectorizer = d["vectorizer"]
        self.tfidf_matrix = d["tfidf_matrix"]
        self.tfidf_appids = d["appids"]
        self.tfidf_idx = {a: i for i, a in enumerate(self.tfidf_appids)}

        d = joblib.load(MODELS_DIR / "svd.pkl")
        self.svd = d["svd"]
        self.user_factors = d["user_factors"]
        self.item_factors = d["item_factors"]
        self.user_to_idx = {str(u): i for i, u in enumerate(d["user_categories"])}
        self.item_to_appid = {i: int(a) for i, a in enumerate(d["item_categories"])}

        d = joblib.load(MODELS_DIR / "embeddings.pkl")
        raw = d["embeddings"]
        self.embeddings = raw.astype(np.float32) if hasattr(raw, "astype") else np.array(raw, dtype=np.float32)
        self.appid_to_emb = d["appid_to_idx"]

        w = joblib.load(MODELS_DIR / "hybrid_weights.pkl")

        # Новый формат весов:
        # {
        #   "auth": {"alpha": ..., "beta": ..., "gamma": ...},
        #   "anon": {"alpha": ..., "beta": ..., "gamma": ...}
        # }
        if "auth" in w and "anon" in w:
            self.auth_alpha = float(w["auth"]["alpha"])
            self.auth_beta = float(w["auth"]["beta"])
            self.auth_gamma = float(w["auth"]["gamma"])

            self.anon_alpha = float(w["anon"]["alpha"])
            self.anon_beta = float(w["anon"]["beta"])
            self.anon_gamma = float(w["anon"]["gamma"])

        # Старый формат оставляем для совместимости:
        # {"alpha": ..., "beta": ..., "gamma": ...}
        else:
            self.auth_alpha = float(w["alpha"])
            self.auth_beta = float(w["beta"])
            self.auth_gamma = float(w["gamma"])

            self.anon_alpha = float(w.get("anon_alpha", 0.50))
            self.anon_beta = float(w.get("anon_beta", 0.00))
            self.anon_gamma = float(w.get("anon_gamma", 0.50))

        self.apps_meta = joblib.load(MODELS_DIR / "apps_meta.pkl")

    def get_recommendations(self, owned_games, steamid=None, top_n=20, min_reviews=50, exclude_free=False,
                            db_weights=None):
        owned_games = self._enrich_with_db(owned_games, db_weights)

        owned = {int(g["appid"]) for g in owned_games}
        scores = {}

        use_svd = steamid is not None and str(steamid) in self.user_to_idx

        if use_svd:
            alpha = self.auth_alpha
            beta = self.auth_beta
            gamma = self.auth_gamma
        else:
            alpha = self.anon_alpha
            beta = self.anon_beta
            gamma = self.anon_gamma

        for rank, appid in enumerate(self._tfidf_top(owned_games, owned)):
            scores[appid] = scores.get(appid, 0) + alpha / (rank + 1)

        if use_svd:
            for rank, appid in enumerate(self._svd_top(steamid, owned)):
                scores[appid] = scores.get(appid, 0) + beta / (rank + 1)

        for rank, appid in enumerate(self._emb_top(owned_games, owned)):
            scores[appid] = scores.get(appid, 0) + gamma / (rank + 1)

        results = []
        for appid, score in scores.items():
            if appid not in self.apps_meta.index:
                continue

            game = self.apps_meta.loc[appid]
            rec_total = game.get("recommendations_total") or 0

            if rec_total < min_reviews:
                continue

            if exclude_free and bool(game.get("is_free", False)):
                continue

            results.append({
                "appid": int(appid),
                "name": game.get("name", "Unknown"),
                "score": round(float(score), 4),
                "price_usd": self._cents_to_usd(game.get("mat_final_price")),
                "is_free": bool(game.get("is_free", False)),
                "recommendations": int(rec_total),
                "metacritic": game.get("metacritic_score"),
                "header_image": game.get("header_image"),
                "store_url": f"https://store.steampowered.com/app/{appid}",
            })

        results.sort(key=lambda x: -x["score"])
        return results[:top_n]

    def get_similar_games(self, appid, top_n=10):
        if appid not in self.appid_to_emb: return []
        idx = self.appid_to_emb[appid]
        vector = self.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(vector, self.embeddings)[0]
        results = []
        for other, other_idx in self.appid_to_emb.items():
            if other == appid or other not in self.apps_meta.index: continue
            game = self.apps_meta.loc[other]
            results.append({
                "appid": int(other),
                "name": game.get("name", "Unknown"),
                "score": round(float(sims[other_idx]), 4),
                "header_image": game.get("header_image"),
                "store_url": f"https://store.steampowered.com/app/{other}",
            })
        results.sort(key=lambda x: -x["score"])
        return results[:top_n]

    def _enrich_with_db(self, owned_games: list[dict], db_weights: dict | None) -> list[dict]:
        if not db_weights: return owned_games
        steam_map = {int(g["appid"]): g.copy() for g in owned_games}
        enriched = []
        for g in owned_games:
            appid = int(g["appid"])
            game = g.copy()
            if appid in db_weights and db_weights[appid] > 0:
                extra_hours = max(db_weights[appid] * 7, 0)
                game["playtime_hours"] = game.get("playtime_hours", 0) + extra_hours
            enriched.append(game)
        for appid, weight in db_weights.items():
            if appid not in steam_map and weight > 0:
                enriched.append({"appid": appid, "playtime_hours": weight * 7})
        return enriched

    def _tfidf_top(self, owned_games, owned_set, n=200):
        profile = np.zeros(self.tfidf_matrix.shape[1])
        total_w = 0.0
        for g in owned_games:
            appid = int(g["appid"])
            if appid not in self.tfidf_idx: continue
            w = max(np.log1p(g.get("playtime_hours", 0)), 0.1)
            profile += w * self.tfidf_matrix[self.tfidf_idx[appid]].toarray()[0]
            total_w += w
        if total_w == 0: return []
        profile /= total_w
        raw = cosine_similarity([profile], self.tfidf_matrix)[0]
        ranked = sorted(((self.tfidf_appids[i], raw[i]) for i in range(len(self.tfidf_appids)) if
                         self.tfidf_appids[i] not in owned_set), key=lambda x: -x[1])
        return [a for a, _ in ranked[:n]]

    def _svd_top(self, steamid, owned_set, n=200):
        u_vec = self.user_factors[self.user_to_idx[str(steamid)]]
        scores_all = self.item_factors @ u_vec
        ranked = sorted(((self.item_to_appid[i], float(scores_all[i])) for i in range(len(self.item_to_appid)) if
                         self.item_to_appid[i] not in owned_set), key=lambda x: -x[1])
        return [a for a, _ in ranked[:n]]

    def _emb_top(self, owned_games, owned_set, n=200):
        dim = self.embeddings.shape[1]
        profile = np.zeros(dim, dtype=np.float32)
        total_w = 0.0
        for g in owned_games:
            appid = int(g["appid"])
            if appid not in self.appid_to_emb: continue
            w = max(np.log1p(g.get("playtime_hours", 0)), 0.1)
            profile += w * self.embeddings[self.appid_to_emb[appid]]
            total_w += w
        if total_w == 0: return []
        profile /= total_w
        raw = cosine_similarity([profile], self.embeddings)[0]
        emb_ids = list(self.appid_to_emb.keys())
        ranked = sorted(((emb_ids[i], float(raw[self.appid_to_emb[emb_ids[i]]])) for i in range(len(emb_ids)) if
                         emb_ids[i] not in owned_set), key=lambda x: -x[1])
        return [a for a, _ in ranked[:n]]

    @staticmethod
    def _cents_to_usd(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return round(float(value) / 100, 2)


recommender = HybridRecommender()