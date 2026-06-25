"""Visualizaciones del dashboard (Matplotlib + Seaborn).

Estética monocromática: fondo blanco, ejes y bordes negros, escala de grises.
Cada función devuelve una figura de Matplotlib lista para ``st.pyplot``.
"""
from __future__ import annotations

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Estilo global monocromático ------------------------------------------- #
sns.set_theme(style="white")
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.4,
    "axes.grid": False,
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "axes.titleweight": "bold",
    "xtick.color": "black",
    "ytick.color": "black",
})

_DARK = "#1a1a1a"     # relleno principal
_MID = "#8a8a8a"      # relleno secundario
_LINE = "#000000"     # bordes


def _style(ax) -> None:
    """Marco negro grueso y ticks negros en todos los ejes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_LINE)
        spine.set_linewidth(1.4)
    ax.tick_params(colors=_LINE)
    ax.title.set_fontsize(13)


def top_genres(df: pd.DataFrame, top_n: int = 15):
    """Barra horizontal de los géneros más frecuentes."""
    counter: Counter[str] = Counter()
    for s in df.get("genres_str", pd.Series(dtype=str)).dropna():
        counter.update(g.strip() for g in str(s).split(",") if g.strip())
    common = counter.most_common(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    if common:
        names, counts = zip(*common)
        ax.barh(list(names), list(counts), color=_DARK, edgecolor=_LINE, linewidth=0.8)
        ax.invert_yaxis()
    ax.set_title("GÉNEROS MÁS FRECUENTES")
    ax.set_xlabel("Nº de películas")
    _style(ax)
    fig.tight_layout()
    return fig


def ratings_distribution(df: pd.DataFrame):
    """Histograma de la distribución de ratings (vote_average)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ratings = pd.to_numeric(df.get("vote_average"), errors="coerce").dropna()
    ratings = ratings[ratings > 0]
    ax.hist(ratings, bins=20, color=_MID, edgecolor=_LINE, linewidth=1.0)
    ax.set_title("DISTRIBUCIÓN DE RATINGS")
    ax.set_xlabel("Rating (vote_average)")
    ax.set_ylabel("Frecuencia")
    _style(ax)
    fig.tight_layout()
    return fig


def top_popular(df: pd.DataFrame, top_n: int = 10):
    """Barra horizontal de las películas más populares."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if {"title", "popularity"}.issubset(df.columns):
        top = df.nlargest(top_n, "popularity")
        ax.barh(top["title"], top["popularity"], color=_DARK,
                edgecolor=_LINE, linewidth=0.8)
        ax.invert_yaxis()
    ax.set_title(f"TOP {top_n} POR POPULARIDAD")
    ax.set_xlabel("Popularidad")
    _style(ax)
    fig.tight_layout()
    return fig


def similarity_heatmap(df: pd.DataFrame, similarity: np.ndarray, sample: int = 15):
    """Heatmap (escala de grises) de una submuestra de la matriz de similitud."""
    n = min(sample, len(df))
    idx = df.nlargest(n, "popularity").index.to_numpy()
    sub = similarity[np.ix_(idx, idx)]
    labels = [t[:22] for t in df.loc[idx, "title"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        sub, xticklabels=labels, yticklabels=labels, cmap="Greys",
        ax=ax, square=True, linewidths=0.5, linecolor=_LINE,
        cbar_kws={"label": "Similitud"},
    )
    ax.set_title("MATRIZ DE SIMILITUD")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


def cluster_scatter(df: pd.DataFrame):
    """Dispersión 2D (PCA) con clusters diferenciados en escala de grises."""
    fig, ax = plt.subplots(figsize=(9, 6))
    if {"pca_x", "pca_y", "cluster"}.issubset(df.columns):
        clusters = sorted(df["cluster"].unique())
        greys = plt.cm.Greys(np.linspace(0.35, 0.92, len(clusters)))
        for c, color in zip(clusters, greys):
            sub = df[df["cluster"] == c]
            ax.scatter(sub["pca_x"], sub["pca_y"], s=34, color=color,
                       edgecolor=_LINE, linewidth=0.6, label=f"Cluster {c}")
        leg = ax.legend(title="CLUSTER", bbox_to_anchor=(1.02, 1), loc="upper left",
                        frameon=True, edgecolor=_LINE)
        leg.get_frame().set_linewidth(1.2)
    else:
        ax.text(0.5, 0.5, "Sin datos de clustering", ha="center", va="center")
    ax.set_title("CLUSTERS DE PELÍCULAS (PCA 2D)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    _style(ax)
    fig.tight_layout()
    return fig
