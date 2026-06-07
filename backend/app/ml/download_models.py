from pathlib import Path
from urllib.request import urlopen, Request
import shutil
import time


MODELS_DIR = Path("models")

MODEL_URLS = {
    "tfidf.pkl": "https://huggingface.co/syrymalisher/steam-game-recommender-models/resolve/main/tfidf.pkl",
    "svd.pkl": "https://huggingface.co/syrymalisher/steam-game-recommender-models/resolve/main/svd.pkl",
    "embeddings.pkl": "https://huggingface.co/syrymalisher/steam-game-recommender-models/resolve/main/embeddings.pkl",
    "hybrid_weights.pkl": "https://huggingface.co/syrymalisher/steam-game-recommender-models/resolve/main/hybrid_weights.pkl",
    "apps_meta.pkl": "https://huggingface.co/syrymalisher/steam-game-recommender-models/resolve/main/apps_meta.pkl",
}


def download_file(url: str, path: Path, retries: int = 5) -> None:
    temp_path = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            if temp_path.exists():
                temp_path.unlink()

            print(f"⬇️ Downloading {path.name}... attempt {attempt}/{retries}")

            request = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urlopen(request, timeout=120) as response:
                expected_size = response.headers.get("Content-Length")

                with open(temp_path, "wb") as file:
                    shutil.copyfileobj(response, file)

            if expected_size is not None:
                expected_size = int(expected_size)
                actual_size = temp_path.stat().st_size

                if actual_size != expected_size:
                    raise RuntimeError(
                        f"Incomplete download for {path.name}: "
                        f"got {actual_size} bytes, expected {expected_size} bytes"
                    )

            temp_path.replace(path)
            print(f"✅ Downloaded: {path.name}")
            return

        except Exception as e:
            print(f"⚠️ Failed to download {path.name}: {e}")

            if temp_path.exists():
                temp_path.unlink()

            if attempt == retries:
                raise

            time.sleep(5)


def ensure_models_downloaded() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        path = MODELS_DIR / filename

        if path.exists() and path.stat().st_size > 0:
            print(f"✅ Model exists: {filename}")
            continue

        download_file(url, path)