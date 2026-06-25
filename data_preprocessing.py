"""Preprocesamiento de datos para el recomendador.

Limpia el dataset crudo, parsea los campos de lista (que pueden venir como
listas reales de la TMDB API o como strings JSON desde el CSV / Kaggle),
y construye la columna ``combined_features`` que alimenta al vectorizador.
"""
from __future__ import annotations

import ast
import json
import re

import pandas as pd

import config

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_MULTISPACE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Minúsculas, sin puntuación, espacios colapsados."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower()
    s = _NON_ALNUM.sub(" ", s)
    s = _MULTISPACE.sub(" ", s)
    return s.strip()


def parse_field(value: object) -> list[str]:
    """Convierte un campo en lista de strings de nombres.

    Acepta:
        * lista de strings  -> tal cual
        * lista de dicts con clave 'name' (formato crudo TMDB/Kaggle)
        * string JSON / repr de lista de Python
        * string vacío / NaN -> []
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        items = s
        # Decodifica tolerando doble codificación JSON (string -> string -> list).
        # Esto repara CSV donde una columna ya serializada se volvió a serializar.
        for _ in range(3):
            parsed = None
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(items)
                    break
                except Exception:
                    continue
            if parsed is None:
                break
            if isinstance(parsed, str):
                items = parsed  # estaba doblemente codificado: reintenta
                continue
            items = parsed
            break
        if isinstance(items, str):
            # no era JSON: trátalo como lista separada por comas
            items = [p.strip() for p in items.split(",") if p.strip()]
    else:
        return []

    names: list[str] = []
    for it in items:
        if isinstance(it, dict):
            name = it.get("name")
            if name:
                names.append(str(name))
        elif it is not None:
            names.append(str(it))
    return names


def get_director(crew: object) -> str:
    """Extrae el director de un campo crew (lista de dicts con 'job')."""
    if isinstance(crew, str):
        try:
            crew = json.loads(crew)
        except Exception:
            try:
                crew = ast.literal_eval(crew)
            except Exception:
                return ""
    if isinstance(crew, list):
        for member in crew:
            if isinstance(member, dict) and member.get("job") == "Director":
                return str(member.get("name", ""))
    return ""


def top_cast(cast: object, n: int = config.TOP_CAST) -> list[str]:
    """Devuelve los n primeros actores."""
    return parse_field(cast)[:n]


def _tokenize_names(names: list[str]) -> str:
    """Une nombres propios en un solo token (sin espacios) para el vectorizador.

    "Christopher Nolan" -> "christophernolan" para que cuente como entidad única.
    """
    return " ".join(clean_text(name).replace(" ", "") for name in names if name)


def add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = (
        pd.to_datetime(df.get("release_date"), errors="coerce").dt.year
    )
    return df


def build_combined_features(df: pd.DataFrame) -> pd.Series:
    """Combina géneros + keywords + cast + director + sinopsis en un texto."""
    genres = df["genres_list"].apply(_tokenize_names)
    keywords = df["keywords_list"].apply(_tokenize_names)
    cast = df["cast_list"].apply(_tokenize_names)
    director = df["director"].apply(lambda d: clean_text(str(d)).replace(" ", ""))
    overview = df["overview"].apply(clean_text)

    # géneros y keywords pesan el doble (repetidos) por su valor de señal
    combined = (
        genres + " " + genres + " " +
        keywords + " " +
        cast + " " +
        director + " " + director + " " +
        overview
    )
    return combined.apply(lambda s: _MULTISPACE.sub(" ", s).strip())


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo: limpia, parsea y genera ``combined_features``."""
    df = df.copy()

    # columnas obligatorias mínimas
    required = ["title", "overview"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria '{col}' en el dataset.")

    # elimina filas sin título o sinopsis
    df = df.dropna(subset=["title", "overview"])
    df = df[df["overview"].astype(str).str.strip().astype(bool)]

    # parsea listas a columnas auxiliares
    df["genres_list"] = (
        df["genres"].apply(parse_field)
        if "genres" in df.columns
        else [[] for _ in range(len(df))]
    )
    df["keywords_list"] = (
        df["keywords"].apply(parse_field)
        if "keywords" in df.columns
        else [[] for _ in range(len(df))]
    )
    df["cast_list"] = (
        df["cast"].apply(top_cast)
        if "cast" in df.columns
        else [[] for _ in range(len(df))]
    )

    # director: si no existe pero hay 'crew', se deriva
    if "director" not in df.columns and "crew" in df.columns:
        df["director"] = df["crew"].apply(get_director)
    df["director"] = df.get("director", "").fillna("")

    # campos numéricos
    df["vote_average"] = pd.to_numeric(
        df.get("vote_average"), errors="coerce"
    ).fillna(0.0)
    df["popularity"] = pd.to_numeric(
        df.get("popularity"), errors="coerce"
    ).fillna(0.0)

    df = add_year(df)

    # versiones legibles (para mostrar en la UI)
    df["genres_str"] = df["genres_list"].apply(lambda xs: ", ".join(xs))
    df["keywords_str"] = df["keywords_list"].apply(lambda xs: ", ".join(xs))
    df["cast_str"] = df["cast_list"].apply(lambda xs: ", ".join(xs))

    # combined features
    df["combined_features"] = build_combined_features(df)

    # dedup por título (conserva la más popular) y reindexa
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset="title", keep="first")
    df = df[df["combined_features"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)

    return df


def load_and_preprocess(path=config.MOVIES_CSV) -> pd.DataFrame:
    """Carga el CSV y aplica el preprocesamiento."""
    df = pd.read_csv(path)
    return preprocess(df)
