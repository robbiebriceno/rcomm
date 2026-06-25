"""Configuración central del proyecto.

Rutas, constantes de ML y carga de la API key de TMDB desde múltiples fuentes
(variable de entorno, archivo .env o st.secrets de Streamlit).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
CACHE_DIR = BASE_DIR / ".cache"

# Artefactos de datos
MOVIES_CSV = DATA_DIR / "movies.csv"

# Artefactos de modelos
TFIDF_MODEL = MODELS_DIR / "tfidf_model.pkl"
TFIDF_MATRIX = MODELS_DIR / "tfidf_matrix.pkl"
SIMILARITY_MATRIX = MODELS_DIR / "similarity_matrix.pkl"
EMBEDDINGS = MODELS_DIR / "embeddings.pkl"
KMEANS_MODEL = MODELS_DIR / "kmeans.pkl"
MOVIES_PROCESSED = MODELS_DIR / "movies_processed.parquet"

# Asegurar que las carpetas existen
for _d in (DATA_DIR, MODELS_DIR, ASSETS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Constantes de Machine Learning / NLP
# --------------------------------------------------------------------------- #
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_FEATURES = 10_000
TOP_CAST = 4                # número de actores principales a usar como features
N_CLUSTERS = 8             # clusters de KMeans para "categorías ocultas"
DEFAULT_PAGES = 125        # ~2500 películas (20 por página) por defecto

# --------------------------------------------------------------------------- #
# TMDB
# --------------------------------------------------------------------------- #
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w500"
BACKDROP_SIZE = "w780"
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750.png?text=Sin+Poster"


def get_tmdb_api_key() -> str | None:
    """Devuelve la API key de TMDB buscando en varias fuentes.

    Orden de prioridad:
        1. Variable de entorno ``TMDB_API_KEY``
        2. Archivo ``.env`` (cargado con python-dotenv si está disponible)
        3. ``st.secrets`` de Streamlit (cuando se ejecuta dentro de la app)

    Returns
    -------
    str | None
        La clave si se encuentra, en caso contrario ``None``.
    """
    # 1 + 2: entorno / .env
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
    except Exception:  # pragma: no cover - dotenv es opcional
        pass

    key = os.getenv("TMDB_API_KEY")
    if key:
        return key.strip()

    # 3: st.secrets (solo si streamlit está disponible y configurado)
    try:
        import streamlit as st

        if "TMDB_API_KEY" in st.secrets:
            return str(st.secrets["TMDB_API_KEY"]).strip()
    except Exception:  # pragma: no cover - fuera del runtime de streamlit
        pass

    return None
