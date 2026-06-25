<div align="center">

# 🎬 rcomm — Recomendador Inteligente de Películas

**Encuentra películas similares con Machine Learning, NLP y búsqueda semántica.**
Sin usuarios, sin autenticación, sin historial — solo recomendaciones.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/NLP-all--MiniLM--L6--v2-4B8BBE)
![TMDB](https://img.shields.io/badge/TMDB-API-01B4E4?logo=themoviedatabase&logoColor=white)

</div>

---

## ✨ Características

| | Función | Tecnología |
|---|---|---|
| 🎯 | **Recomendación por película** — las más parecidas a una que elijas | TF-IDF + similitud coseno |
| 🧠 | **Búsqueda semántica** — por frase libre, p. ej. *"ciencia ficción con viajes en el tiempo"* | Embeddings `all-MiniLM-L6-v2` |
| 💡 | **Explicabilidad** — por qué se recomienda cada película | Género · Director · Temática · Actores |
| 📊 | **Dashboard** — análisis visual del catálogo | Matplotlib + Seaborn |
| 🖼️ | **Integración TMDB** — pósters, backdrops, tráilers y sinopsis en vivo | TMDB API + caché |
| 🎨 | **Interfaz** — diseño minimalista monocromático | Estética editorial |

---

## ⚡ Inicio rápido

```bash
pip install -r requirements.txt     # 1. instalar dependencias
cp .env.example .env                # 2. añade tu TMDB_API_KEY (copy en Windows)
python train.py                     # 3. descarga el dataset y entrena
streamlit run app.py                # 4. ¡a recomendar!
```

> Orden obligatorio: **instalar → configurar clave → entrenar → ejecutar**.

---

## 🧠 Cómo funciona

```
                                ┌─ TF-IDF + similitud coseno ─┐
  TMDB API ─→ movies.csv ─→ preprocesamiento ─→ embeddings ──┼─→ modelos (.pkl) ─→ app
   (fetch)     (dataset)     (combined_features) KMeans + PCA ┘    (joblib)        (Streamlit)
```

1. **Datos** — `train.py` descarga películas populares de TMDB (géneros, sinopsis, keywords, reparto, director, rating, popularidad).
2. **Preprocesamiento** — limpia el texto y combina todos los campos en una columna `combined_features`.
3. **Vectorización** — `TfidfVectorizer` (stop words en inglés, hasta 10.000 features) + **similitud coseno**.
4. **Semántica** — `SentenceTransformer("all-MiniLM-L6-v2")` genera embeddings para búsqueda por significado.
5. **Extras** — **KMeans** descubre grupos ocultos y **PCA** los proyecta a 2D para el dashboard.
6. **Persistencia** — todo se guarda con **Joblib** para que la app cargue al instante.

---

## 🗂️ Estructura del proyecto

```
rcomm/
├── app.py                 # Interfaz Streamlit (recomendar · semántica · dashboard)
├── train.py               # Pipeline de entrenamiento (CLI)
├── recommender.py         # Motor: recommend_movies / semantic_search / explain
├── data_preprocessing.py  # Limpieza, parseo y combined_features
├── tmdb_api.py            # Cliente TMDB (dataset + media en vivo, con caché)
├── visualizations.py      # Gráficos del dashboard (Matplotlib/Seaborn)
├── config.py              # Rutas, constantes y carga de la API key
├── models/                # Artefactos entrenados (.pkl, .parquet) — los genera train.py
├── data/                  # movies.csv — generado por el fetch
├── assets/style.css       # Estilos de la interfaz
├── docs/                  # Manuales en PDF (usuario/sistema y guía rápida)
├── .streamlit/config.toml # Tema base de la app
├── .env.example           # Plantilla para TMDB_API_KEY
├── requirements.txt
└── README.md
```

---

## 📦 Instalación detallada

**1. Requisitos** — Python **3.11+**.

**2. Dependencias**
```bash
pip install -r requirements.txt
```
> La búsqueda semántica instala **PyTorch** (~1–2 GB). Para una versión ligera, entrena con `python train.py --no-semantic`.

**3. Clave de TMDB** (gratuita)
1. Crea una cuenta en [TMDB](https://www.themoviedb.org/) y pide una **API Key (v3 auth)** en [Ajustes → API](https://www.themoviedb.org/settings/api).
2. Copia la plantilla y pega tu clave:
   ```bash
   cp .env.example .env       # Windows: copy .env.example .env
   ```
   ```env
   TMDB_API_KEY=tu_clave_real
   ```
   También se admite la variable de entorno `TMDB_API_KEY` o `st.secrets` de Streamlit.

---

## 🎛️ Uso

La app tiene tres pestañas:

- **🎯 Recomendar** — elige una película y recibe las más parecidas en tarjetas (póster, datos, % de similitud y el motivo).
- **🧠 Búsqueda semántica** — describe lo que quieres en lenguaje natural.
- **📊 Dashboard** — géneros frecuentes, distribución de ratings, top de popularidad, matriz de similitud y clusters.

**Opciones de entrenamiento**

| Comando | Efecto |
|---|---|
| `python train.py` | Entrenamiento por defecto (~2.500 películas) |
| `python train.py --pages 300 --refresh` | Catálogo más grande (~6.000 películas) |
| `python train.py --no-semantic` | Versión ligera, sin PyTorch |
| `python train.py --clusters 10` | Nº de grupos de KMeans |

> Cada página de TMDB = 20 películas (máximo ~10.000). Usa `--refresh` para reconstruir el dataset.

---

## 🧪 Verificación rápida

```bash
python tmdb_api.py        # comprueba la conexión a TMDB
python recommender.py     # prueba el recomendador en consola (tras entrenar)
```

---

## ⚙️ Rendimiento y notas

- **Memoria** — la matriz de similitud es N×N: ~2.500 películas ≈ decenas de MB; ~10.000 ≈ 400 MB. Ajusta `--pages` a tus recursos.
- **Caché** — peticiones HTTP a TMDB en `.cache/` (1 semana); media y modelos cacheados en la app (`st.cache_data` / `st.cache_resource`).
- **Sin clave** — la app no se rompe: muestra placeholders. La clave solo es imprescindible para *construir el dataset* (o aporta tu propio `data/movies.csv`).

---

## 🔧 Solución de problemas

| Problema | Solución |
|---|---|
| `No hay modelos entrenados` en la app | Ejecuta `python train.py` antes de `streamlit run app.py`. |
| `No hay TMDB_API_KEY` al entrenar | Crea `.env` con tu clave o exporta `TMDB_API_KEY`. |
| Descarga lenta / errores 429 | TMDB limita el ritmo; el cliente reintenta solo. Reduce `--pages`. |
| Instalación de PyTorch muy pesada | Usa `python train.py --no-semantic`. |


---

## 📚 Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `Joblib` · `Matplotlib` · `Seaborn` · `SentenceTransformers` · `TMDB API`

---

<div align="center">

Este producto usa la **TMDB API** pero no está avalado ni certificado por TMDB.
<br>
🎬 Hecho con fines educativos y de portfolio.

</div>
