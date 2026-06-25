"""Motor de recomendación.

Carga los artefactos entrenados y expone:
    * ``recommend_movies``  -> recomendación clásica por similitud (TF-IDF/coseno)
    * ``semantic_search``   -> búsqueda semántica por título o frase libre
    * ``explain``           -> por qué se recomienda (géneros, director, cast, temática)

La carga de artefactos es perezosa para que importar el módulo sea barato.
"""
from __future__ import annotations

import difflib
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

import config


def _split(s: object) -> set[str]:
    """Convierte un string 'a, b, c' en un set normalizado en minúsculas."""
    if not isinstance(s, str) or not s.strip():
        return set()
    return {p.strip().lower() for p in s.split(",") if p.strip()}


class Recommender:
    """Recomendador de películas basado en contenido + búsqueda semántica."""

    def __init__(self) -> None:
        self._embed_model = None  # cargado bajo demanda

    # ------------------------------------------------------------------ #
    # Carga perezosa de artefactos
    # ------------------------------------------------------------------ #
    @cached_property
    def movies(self) -> pd.DataFrame:
        if not config.MOVIES_PROCESSED.exists():
            raise FileNotFoundError(
                "No se encontró el dataset procesado. Ejecuta primero: python train.py"
            )
        df = pd.read_parquet(config.MOVIES_PROCESSED)
        return df

    @cached_property
    def similarity(self) -> np.ndarray:
        return joblib.load(config.SIMILARITY_MATRIX)

    @cached_property
    def embeddings(self) -> np.ndarray | None:
        if config.EMBEDDINGS.exists():
            return joblib.load(config.EMBEDDINGS)
        return None

    @cached_property
    def _title_index(self) -> dict[str, int]:
        return {t.lower(): i for i, t in enumerate(self.movies["title"])}

    @property
    def has_semantic(self) -> bool:
        return self.embeddings is not None

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(config.EMBED_MODEL)
        return self._embed_model

    # ------------------------------------------------------------------ #
    # Utilidades de búsqueda de título
    # ------------------------------------------------------------------ #
    @property
    def titles(self) -> list[str]:
        return self.movies["title"].tolist()

    def find_index(self, title: str) -> int | None:
        """Encuentra el índice de una película tolerando mayúsculas/typos."""
        if not title:
            return None
        key = title.strip().lower()
        if key in self._title_index:
            return self._title_index[key]
        match = difflib.get_close_matches(key, self._title_index.keys(), n=1, cutoff=0.6)
        if match:
            return self._title_index[match[0]]
        return None

    # ------------------------------------------------------------------ #
    # Salida formateada
    # ------------------------------------------------------------------ #
    def _row_to_dict(self, idx: int, score: float) -> dict:
        row = self.movies.iloc[idx]
        year = row.get("year")
        return {
            "index": int(idx),
            "title": row["title"],
            "score": round(float(score), 4),
            "genres": row.get("genres_str", ""),
            "year": int(year) if pd.notna(year) else None,
            "rating": round(float(row.get("vote_average", 0.0)), 1),
            "popularity": round(float(row.get("popularity", 0.0)), 1),
            "overview": row.get("overview", ""),
            "director": row.get("director", ""),
            "cast": row.get("cast_str", ""),
            "tmdb_id": int(row["id"]) if "id" in row and pd.notna(row.get("id")) else None,
            "poster_path": row.get("poster_path", ""),
            "cluster": int(row["cluster"]) if "cluster" in row and pd.notna(row.get("cluster")) else None,
        }

    # ------------------------------------------------------------------ #
    # 1) Recomendación clásica
    # ------------------------------------------------------------------ #
    def recommend_movies(self, movie_title: str, n_recommendations: int = 10) -> list[dict]:
        """Devuelve las películas más similares a ``movie_title``."""
        idx = self.find_index(movie_title)
        if idx is None:
            return []

        scores = self.similarity[idx]
        # ordena descendente y descarta la propia película
        order = np.argsort(scores)[::-1]
        order = [i for i in order if i != idx][:n_recommendations]

        return [self._row_to_dict(i, scores[i]) for i in order]

    # ------------------------------------------------------------------ #
    # 2) Búsqueda semántica
    # ------------------------------------------------------------------ #
    def semantic_search(self, query: str, n_recommendations: int = 10) -> list[dict]:
        """Busca por frase libre o título usando embeddings semánticos.

        Ej.: "una película de ciencia ficción con viajes en el tiempo".
        Si no hay embeddings disponibles, cae a la recomendación clásica por título.
        """
        if not query or not query.strip():
            return []

        if not self.has_semantic:
            # fallback: trata la query como un título
            return self.recommend_movies(query, n_recommendations)

        model = self._get_embed_model()
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        # embeddings ya normalizados -> producto punto = coseno
        scores = self.embeddings @ q_vec
        order = np.argsort(scores)[::-1][:n_recommendations]
        return [self._row_to_dict(i, scores[i]) for i in order]

    # ------------------------------------------------------------------ #
    # 3) Explicabilidad
    # ------------------------------------------------------------------ #
    def explain(self, source_title: str, target_index: int) -> dict:
        """Explica por qué ``target_index`` se recomienda dada ``source_title``."""
        src_idx = self.find_index(source_title)
        if src_idx is None:
            return {}

        src = self.movies.iloc[src_idx]
        tgt = self.movies.iloc[target_index]

        src_genres, tgt_genres = _split(src.get("genres_str")), _split(tgt.get("genres_str"))
        src_cast, tgt_cast = _split(src.get("cast_str")), _split(tgt.get("cast_str"))
        src_kw, tgt_kw = _split(src.get("keywords_str")), _split(tgt.get("keywords_str"))

        shared_genres = src_genres & tgt_genres
        shared_cast = src_cast & tgt_cast
        shared_kw = src_kw & tgt_kw
        same_director = bool(
            str(src.get("director", "")).strip()
            and src.get("director") == tgt.get("director")
        )

        return {
            "Género": sorted(g.title() for g in shared_genres),
            "Director": [src.get("director")] if same_director else [],
            "Temática": sorted(k.title() for k in shared_kw),
            "Actores": sorted(a.title() for a in shared_cast),
            "same_cluster": bool(src.get("cluster") == tgt.get("cluster")),
        }


if __name__ == "__main__":  # prueba rápida en consola
    rec = Recommender()
    print("Películas cargadas:", len(rec.movies))
    for r in rec.recommend_movies("The Dark Knight", 5):
        print(f"  {r['score']:.3f}  {r['title']} ({r['year']})  ⭐{r['rating']}")
    if rec.has_semantic:
        print("\nSemántica: 'space adventure with aliens'")
        for r in rec.semantic_search("space adventure with aliens", 5):
            print(f"  {r['score']:.3f}  {r['title']} ({r['year']})")
