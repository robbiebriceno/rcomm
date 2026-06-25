"""🎬 Recomendador Inteligente de Películas — interfaz Streamlit.

Tres modos:
    * Recomendar por película (TF-IDF + similitud coseno)
    * Búsqueda semántica por frase libre (SentenceTransformers)
    * Dashboard de visualizaciones (Matplotlib + Seaborn)

Sin usuarios, autenticación ni historial.
"""
from __future__ import annotations

import streamlit as st

import config
import visualizations as viz
from recommender import Recommender
from tmdb_api import TMDBClient

# --------------------------------------------------------------------------- #
# Configuración de página + estilos
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Recomendador Inteligente de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = config.ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


load_css()


# --------------------------------------------------------------------------- #
# Carga cacheada de recursos
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Cargando modelos…")
def get_recommender() -> Recommender:
    rec = Recommender()
    _ = rec.movies  # fuerza la carga y valida que exista el dataset
    return rec


@st.cache_resource
def get_tmdb_client() -> TMDBClient:
    return TMDBClient()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_media(tmdb_id: int | None, poster_path: str) -> dict:
    """Obtiene poster/backdrop/trailer en vivo desde TMDB (con caché)."""
    client = get_tmdb_client()
    if tmdb_id and client.is_configured:
        media = client.get_movie_media(tmdb_id)
        if media:
            return media
    return {
        "poster_url": TMDBClient.get_poster_url(poster_path or None),
        "backdrop_url": None,
        "trailer_url": None,
        "overview": "",
    }


# --------------------------------------------------------------------------- #
# Componentes de UI
# --------------------------------------------------------------------------- #
GRID_COLS = 3  # tarjetas por fila en la cuadrícula de resultados


def _meta_item(label: str, value: object) -> str:
    return (f"<div class='meta-item'><span class='k'>{label}</span>"
            f"<span class='v'>{value}</span></div>")


def _why_html(rec: Recommender, source_title: str, index: int) -> str:
    """Construye el bloque compacto de 'por qué se recomienda'."""
    why = rec.explain(source_title, index)
    shared: list[str] = []
    detail: list[str] = []
    for key, label in (("Género", "Género"), ("Director", "Director"),
                       ("Temática", "Temática"), ("Actores", "Actores")):
        vals = why.get(key)
        if vals:
            shared.append(label)
            detail.append(f"{label}: {', '.join(vals[:6])}")
    if not shared:
        return ""
    tip = " | ".join(detail).replace('"', "'")
    return (f"<div class='why' title=\"{tip}\"><span class='why-k'>Comparte</span>"
            f"{' · '.join(shared)}</div>")


def render_card(movie: dict, rec: Recommender, source_title: str | None = None) -> None:
    """Renderiza una tarjeta de película como un único bloque HTML."""
    media = fetch_media(movie.get("tmdb_id"), movie.get("poster_path", ""))
    year = movie.get("year")
    year_txt = f"({year})" if year else ""
    pct = int(round(movie["score"] * 100))

    meta = (
        _meta_item("Género", movie["genres"] or "—")
        + _meta_item("Rating", movie["rating"])
        + _meta_item("Año", year if year else "—")
        + _meta_item("Popularidad", movie["popularity"])
    )

    why_html = _why_html(rec, source_title, movie["index"]) if source_title else ""

    trailer_html = ""
    if media.get("trailer_url"):
        trailer_html = (f"<a class='trailer-link' href='{media['trailer_url']}' "
                        f"target='_blank'>Ver tráiler</a>")

    st.markdown(
        f"<div class='movie-card'>"
        f"<div class='poster'><img src='{media['poster_url']}' alt=''></div>"
        f"<div class='card-body'>"
        f"<div class='sim-badge'>Similitud {pct}%</div>"
        f"<h3 class='movie-title'>{movie['title']} <span class='year'>{year_txt}</span></h3>"
        f"<div class='meta'>{meta}</div>"
        f"{why_html}{trailer_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def render_results(results: list[dict], rec: Recommender, source_title: str | None) -> None:
    if not results:
        st.warning("No se encontraron recomendaciones. Prueba con otro título o consulta.")
        return
    for start in range(0, len(results), GRID_COLS):
        cols = st.columns(GRID_COLS, gap="large")
        for col, movie in zip(cols, results[start:start + GRID_COLS]):
            with col:
                render_card(movie, rec, source_title)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(
        "<div class='app-header'>"
        "<h1 class='app-title'>Recomendador Inteligente de Películas</h1>"
        "<p class='subtitle'>Machine Learning · NLP · Búsqueda semántica</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        rec = get_recommender()
    except FileNotFoundError:
        st.error(
            "No hay modelos entrenados todavía.\n\n"
            "Ejecuta en tu terminal:\n\n``python train.py``"
        )
        st.stop()

    client = get_tmdb_client()
    with st.sidebar:
        st.header("Estado")
        st.metric("Películas en catálogo", len(rec.movies))
        st.write("**Búsqueda semántica:**", "Activa" if rec.has_semantic else "No entrenada")
        st.write("**TMDB API:**", "Conectada" if client.is_configured else "Sin clave (placeholders)")
        st.caption("Ajusta el número de recomendaciones en cada pestaña.")

    tab_rec, tab_sem, tab_dash = st.tabs(
        ["Recomendar", "Búsqueda semántica", "Dashboard"]
    )

    # ------------------------------- Recomendar -------------------------- #
    with tab_rec:
        st.subheader("Recomendar a partir de una película")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox(
                "Busca y selecciona una película", options=rec.titles,
                index=0, placeholder="Escribe para buscar…",
            )
        with col2:
            n_rec = st.slider("Nº de recomendaciones", 1, 20, 10, key="n_rec")
        go = st.button("Recomendar", type="primary", use_container_width=True)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if go:
            with st.spinner("Calculando…"):
                results = rec.recommend_movies(selected, n_rec)
            st.markdown(
                f"<p class='results-meta'>{len(results)} películas similares a "
                f"&laquo;{selected}&raquo;</p>",
                unsafe_allow_html=True,
            )
            render_results(results, rec, source_title=selected)

    # --------------------------- Búsqueda semántica ---------------------- #
    with tab_sem:
        st.subheader("Búsqueda por descripción en lenguaje natural")
        if not rec.has_semantic:
            st.info("La búsqueda semántica no está entrenada. Ejecuta "
                    "`python train.py` (sin `--no-semantic`).")
        query = st.text_input(
            "Describe lo que buscas",
            placeholder="Ej.: una película de ciencia ficción con viajes en el tiempo",
        )
        n_sem = st.slider("Nº de recomendaciones", 1, 20, 10, key="n_sem")
        go_sem = st.button("Buscar", type="primary", use_container_width=True, key="sem_btn")
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if go_sem:
            if query.strip():
                with st.spinner("Buscando por significado…"):
                    results = rec.semantic_search(query, n_sem)
                st.markdown(
                    f"<p class='results-meta'>{len(results)} resultados para "
                    f"&laquo;{query}&raquo;</p>",
                    unsafe_allow_html=True,
                )
                render_results(results, rec, source_title=None)
            else:
                st.warning("Escribe una descripción para buscar.")

    # ------------------------------- Dashboard --------------------------- #
    with tab_dash:
        st.subheader("Análisis del catálogo")
        df = rec.movies
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(viz.top_genres(df))
        with c2:
            st.pyplot(viz.ratings_distribution(df))

        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(viz.top_popular(df))
        with c4:
            st.pyplot(viz.cluster_scatter(df))

        st.markdown("#### Matriz de similitud")
        st.pyplot(viz.similarity_heatmap(df, rec.similarity))


if __name__ == "__main__":
    main()
