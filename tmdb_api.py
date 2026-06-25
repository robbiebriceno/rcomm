"""Cliente de la TMDB API.

Dos responsabilidades:
    1. Construir el dataset (``data/movies.csv``) descargando películas populares
       con sus detalles, créditos y keywords.
    2. Servir datos "en vivo" a la interfaz: posters, backdrops, trailers y
       sinopsis actualizada, con caché en disco para no repetir peticiones.

El módulo degrada de forma elegante: si no hay API key o falla la red, los
métodos devuelven ``None`` / placeholders en lugar de lanzar excepciones.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 viaja con requests; el import explícito facilita el Retry
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    Retry = None  # type: ignore

import config

# Caché opcional de peticiones (transparente). Si no está instalado, seguimos.
try:
    import requests_cache

    requests_cache.install_cache(
        str(config.CACHE_DIR / "tmdb_http_cache"),
        expire_after=60 * 60 * 24 * 7,  # 1 semana
        allowable_methods=("GET",),
    )
    _HAS_REQUESTS_CACHE = True
except Exception:  # pragma: no cover
    _HAS_REQUESTS_CACHE = False


class TMDBClient:
    """Cliente ligero sobre la API v3 de TMDB."""

    def __init__(self, api_key: str | None = None, language: str = "en-US") -> None:
        self.api_key = api_key or config.get_tmdb_api_key()
        self.language = language
        self.base_url = config.TMDB_BASE_URL
        self.session = self._build_session()

    # ------------------------------------------------------------------ #
    # Infraestructura
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        if Retry is not None:
            retry = Retry(
                total=4,
                backoff_factor=0.6,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET",),
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        return session

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, **params: Any) -> dict | None:
        """GET genérico contra la API. Devuelve dict o None ante cualquier fallo."""
        if not self.api_key:
            return None
        params.setdefault("api_key", self.api_key)
        params.setdefault("language", self.language)
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # red, rate limit agotado, JSON inválido…
            print(f"[TMDB] error en GET {path}: {exc}")
            return None

    # ------------------------------------------------------------------ #
    # Construcción del dataset
    # ------------------------------------------------------------------ #
    def fetch_movie_ids(self, pages: int) -> list[int]:
        """Recoge IDs de películas ordenadas por popularidad."""
        ids: list[int] = []
        seen: set[int] = set()
        for page in range(1, pages + 1):
            data = self._get(
                "discover/movie",
                sort_by="popularity.desc",
                include_adult="false",
                include_video="false",
                page=page,
                vote_count_gte=50,  # filtra ruido de poca señal
            )
            if not data or not data.get("results"):
                break
            for item in data["results"]:
                mid = item.get("id")
                if mid and mid not in seen:
                    seen.add(mid)
                    ids.append(mid)
            # respeta el límite de páginas que devuelve la API (máx 500)
            if page >= data.get("total_pages", page):
                break
            if not _HAS_REQUESTS_CACHE:
                time.sleep(0.05)  # throttling suave si no hay caché
        return ids

    def get_movie_details(self, movie_id: int) -> dict | None:
        """Detalles + keywords + créditos + vídeos en una sola llamada."""
        return self._get(
            f"movie/{movie_id}",
            append_to_response="keywords,credits,videos",
        )

    @staticmethod
    def _parse_details(d: dict) -> dict | None:
        """Aplana un payload de detalles al esquema mínimo del dataset."""
        title = d.get("title") or d.get("original_title")
        overview = d.get("overview")
        if not title or not overview:
            return None

        genres = [g["name"] for g in d.get("genres", []) if g.get("name")]
        kw = d.get("keywords", {}).get("keywords", [])
        keywords = [k["name"] for k in kw if k.get("name")]

        credits = d.get("credits", {})
        cast = [c["name"] for c in credits.get("cast", [])[:10] if c.get("name")]
        director = next(
            (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
            "",
        )

        videos = d.get("videos", {}).get("results", [])
        trailer_key = next(
            (
                v["key"]
                for v in videos
                if v.get("site") == "YouTube" and v.get("type") == "Trailer"
            ),
            "",
        )

        return {
            "id": d.get("id"),
            "title": title,
            "overview": overview,
            "genres": genres,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "vote_average": d.get("vote_average", 0.0),
            "vote_count": d.get("vote_count", 0),
            "release_date": d.get("release_date", ""),
            "popularity": d.get("popularity", 0.0),
            "poster_path": d.get("poster_path", "") or "",
            "backdrop_path": d.get("backdrop_path", "") or "",
            "trailer_key": trailer_key,
        }

    def build_dataset(self, pages: int = config.DEFAULT_PAGES) -> pd.DataFrame:
        """Descarga y construye el DataFrame completo del dataset.

        Las columnas de listas (genres, keywords, cast) se serializan a JSON al
        guardarse en CSV; ``data_preprocessing.parse_field`` las vuelve a leer.
        """
        if not self.is_configured:
            raise RuntimeError(
                "No hay TMDB_API_KEY configurada. Crea un archivo .env con "
                "TMDB_API_KEY=tu_clave (ver .env.example)."
            )

        print(f"[TMDB] Obteniendo IDs de hasta {pages} páginas…")
        ids = self.fetch_movie_ids(pages)
        print(f"[TMDB] {len(ids)} IDs únicos. Descargando detalles…")

        rows: list[dict] = []
        for i, mid in enumerate(ids, 1):
            details = self.get_movie_details(mid)
            if details:
                parsed = self._parse_details(details)
                if parsed:
                    rows.append(parsed)
            if i % 100 == 0:
                print(f"  …{i}/{len(ids)} procesadas ({len(rows)} válidas)")
            if not _HAS_REQUESTS_CACHE:
                time.sleep(0.03)

        df = pd.DataFrame(rows)
        print(f"[TMDB] Dataset construido: {len(df)} películas válidas.")
        return df

    def save_dataset(self, df: pd.DataFrame, path: Path = config.MOVIES_CSV) -> None:
        out = df.copy()
        for col in ("genres", "keywords", "cast"):
            if col in out.columns:
                out[col] = out[col].apply(json.dumps)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False, encoding="utf-8")
        print(f"[TMDB] Guardado en {path}")

    # ------------------------------------------------------------------ #
    # Datos en vivo para la UI
    # ------------------------------------------------------------------ #
    def search_movie(self, title: str) -> dict | None:
        """Busca una película por título y devuelve el primer resultado."""
        data = self._get("search/movie", query=title, include_adult="false")
        if data and data.get("results"):
            return data["results"][0]
        return None

    def get_movie_media(self, movie_id: int) -> dict:
        """Devuelve poster, backdrop, trailer y sinopsis frescos para un ID."""
        details = self._get(f"movie/{movie_id}", append_to_response="videos")
        if not details:
            return {}
        videos = details.get("videos", {}).get("results", [])
        trailer_key = next(
            (
                v["key"]
                for v in videos
                if v.get("site") == "YouTube" and v.get("type") == "Trailer"
            ),
            "",
        )
        return {
            "poster_url": self.get_poster_url(details.get("poster_path")),
            "backdrop_url": self.get_backdrop_url(details.get("backdrop_path")),
            "trailer_url": self.get_trailer_url(trailer_key),
            "overview": details.get("overview", ""),
        }

    @staticmethod
    def get_poster_url(poster_path: str | None) -> str:
        if poster_path:
            return f"{config.TMDB_IMAGE_BASE}/{config.POSTER_SIZE}{poster_path}"
        return config.PLACEHOLDER_POSTER

    @staticmethod
    def get_backdrop_url(backdrop_path: str | None) -> str | None:
        if backdrop_path:
            return f"{config.TMDB_IMAGE_BASE}/{config.BACKDROP_SIZE}{backdrop_path}"
        return None

    @staticmethod
    def get_trailer_url(trailer_key: str | None) -> str | None:
        if trailer_key:
            return f"https://www.youtube.com/watch?v={trailer_key}"
        return None


if __name__ == "__main__":  # smoke test rápido
    client = TMDBClient()
    if not client.is_configured:
        print("Sin TMDB_API_KEY: configura .env para probar la conexión.")
    else:
        res = client.search_movie("Inception")
        print("OK" if res else "Sin resultados", res.get("title") if res else "")
