"""Script de entrenamiento.

Orquesta todo el pipeline de ML:
    1. (Opcional) Descarga el dataset desde la TMDB API.
    2. Preprocesa los datos.
    3. Entrena el vectorizador TF-IDF y calcula la matriz de similitud coseno.
    4. Genera embeddings semánticos con SentenceTransformers.
    5. Agrupa con KMeans (categorías ocultas) y reduce con PCA (visualización).
    6. Persiste todos los artefactos con joblib.

Uso:
    python train.py                 # usa data/movies.csv si existe, si no lo descarga
    python train.py --refresh       # fuerza la descarga del dataset
    python train.py --pages 50      # descarga ~1000 películas
    python train.py --no-semantic   # omite embeddings/torch (más ligero)
    python train.py --clusters 10   # número de clusters de KMeans
"""
from __future__ import annotations

import argparse
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import data_preprocessing as dp


def build_dataset_if_needed(pages: int, refresh: bool) -> None:
    """Descarga el dataset o lo amplía sin volver a pedir páginas viejas."""
    from tmdb_api import TMDBClient

    client = TMDBClient()
    if not client.is_configured:
        raise SystemExit(
            "No hay TMDB_API_KEY configurada y no existe data/movies.csv.\n"
            "Crea un archivo .env con TMDB_API_KEY=tu_clave (ver .env.example)\n"
            "u obtén una clave gratis en https://www.themoviedb.org/settings/api"
        )

    existing_df = None
    start_page = 1
    if config.MOVIES_CSV.exists() and not refresh:
        existing_df = pd.read_csv(config.MOVIES_CSV)
        existing_pages = max(1, math.ceil(len(existing_df) / 20))
        if pages <= existing_pages:
            print(
                f"[train] Dataset existente en {config.MOVIES_CSV} "
                f"({len(existing_df)} filas, ~{existing_pages} páginas)."
            )
            return
        start_page = existing_pages + 1
        print(
            f"[train] Dataset existente con {len(existing_df)} filas. "
            f"Añadiendo páginas {start_page} a {pages}."
        )
    else:
        print(f"[train] Descargando dataset desde cero: páginas 1 a {pages}.")

    df_new = client.build_dataset(pages=pages, start_page=start_page)
    if df_new.empty:
        if existing_df is None:
            raise SystemExit("La descarga no devolvió películas. Revisa tu API key / conexión.")
        print("[train] No se encontraron filas nuevas para añadir.")
        return

    if existing_df is not None and not existing_df.empty:
        merged = pd.concat([existing_df, df_new], ignore_index=True)
        merged = merged.drop_duplicates(subset="id", keep="first")
        print(f"[train] CSV ampliado: {len(merged)} filas totales.")
        client.save_dataset(merged)
    else:
        client.save_dataset(df_new)


def train_tfidf(df):
    """Entrena TF-IDF y calcula la matriz de similitud coseno."""
    print("[train] Vectorizando con TF-IDF…")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=config.MAX_FEATURES)
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    print(f"[train] Matriz TF-IDF: {tfidf_matrix.shape}")

    print("[train] Calculando similitud coseno…")
    similarity = cosine_similarity(tfidf_matrix, dense_output=True).astype(np.float32)
    print(f"[train] Matriz de similitud: {similarity.shape}")
    return vectorizer, tfidf_matrix, similarity


def train_embeddings(df):
    """Genera embeddings semánticos con SentenceTransformers."""
    from sentence_transformers import SentenceTransformer

    print(f"[train] Cargando modelo de embeddings '{config.EMBED_MODEL}'…")
    model = SentenceTransformer(config.EMBED_MODEL)

    # texto rico: sinopsis + géneros + keywords legibles
    texts = (
        df["overview"].fillna("") + ". Genres: " + df["genres_str"]
        + ". Keywords: " + df["keywords_str"]
    ).tolist()

    print("[train] Generando embeddings (puede tardar la primera vez)…")
    embeddings = model.encode(
        texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    print(f"[train] Embeddings: {embeddings.shape}")
    return embeddings


def cluster_and_reduce(df, features: np.ndarray, n_clusters: int):
    """KMeans para clusters + PCA 2D para visualización."""
    n_clusters = min(n_clusters, len(df))
    print(f"[train] KMeans con {n_clusters} clusters…")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    df["cluster"] = labels

    print("[train] PCA a 2D para visualización…")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    return kmeans


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el recomendador de películas.")
    parser.add_argument("--pages", type=int, default=config.DEFAULT_PAGES,
                        help="Páginas a descargar de TMDB (20 películas/página).")
    parser.add_argument("--refresh", action="store_true",
                        help="Fuerza la descarga del dataset aunque exista.")
    parser.add_argument("--no-semantic", action="store_true",
                        help="Omite los embeddings semánticos (sin torch).")
    parser.add_argument("--clusters", type=int, default=config.N_CLUSTERS,
                        help="Número de clusters de KMeans.")
    args = parser.parse_args()

    # 1. Dataset
    build_dataset_if_needed(args.pages, args.refresh)

    # 2. Preprocesamiento
    print("[train] Preprocesando…")
    df = dp.load_and_preprocess()
    print(f"[train] {len(df)} películas tras el preprocesamiento.")

    # 3. TF-IDF + similitud
    vectorizer, tfidf_matrix, similarity = train_tfidf(df)

    # 4. Embeddings (opcional)
    embeddings = None
    if not args.no_semantic:
        try:
            embeddings = train_embeddings(df)
        except Exception as exc:
            print(f"[train] Aviso: no se pudieron generar embeddings ({exc}). "
                  "Continuo sin búsqueda semántica.")

    # 5. Clustering + reducción (usa embeddings si hay; si no, TF-IDF denso)
    cluster_features = embeddings if embeddings is not None else tfidf_matrix.toarray()
    kmeans = cluster_and_reduce(df, cluster_features, args.clusters)

    # 6. Persistencia
    print("[train] Guardando artefactos…")
    joblib.dump(vectorizer, config.TFIDF_MODEL)
    joblib.dump(tfidf_matrix, config.TFIDF_MATRIX)
    joblib.dump(similarity, config.SIMILARITY_MATRIX)
    joblib.dump(kmeans, config.KMEANS_MODEL)
    if embeddings is not None:
        joblib.dump(embeddings, config.EMBEDDINGS)

    # df procesado (sin columnas auxiliares de listas, que no van a parquet limpio)
    drop_cols = [c for c in ("genres_list", "keywords_list", "cast_list") if c in df.columns]
    df.drop(columns=drop_cols).to_parquet(config.MOVIES_PROCESSED, index=False)

    print("\n[train] ✅ Listo. Artefactos en", config.MODELS_DIR)
    print("        Ejecuta:  streamlit run app.py")


if __name__ == "__main__":
    main()
