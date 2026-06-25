"""Genera los manuales en PDF con estética monocromática (diseño suizo / brutalismo).

Salida:
    docs/Manual_Usuario_y_Sistema.pdf   (completo)
    docs/Guia_Rapida.pdf                (resumido)

Solo usa fuentes estándar (Helvetica / Courier) y caracteres WinAnsi seguros.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

DOCS = Path(__file__).resolve().parent
INK = colors.black
PAPER = colors.white
MUTED = colors.HexColor("#5f5f5f")
HAIR = colors.HexColor("#cfcfcf")
CODEBG = colors.HexColor("#f2f2f2")

DATE = "25 de junio de 2026"
VERSION = "1.0"

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    s["cover_kicker"] = ParagraphStyle(
        "cover_kicker", parent=base["Normal"], fontName="Helvetica-Bold",
        fontSize=9, textColor=MUTED, leading=14, alignment=TA_LEFT,
        spaceAfter=6, tracking=2,
    )
    s["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=34, textColor=INK, leading=36, alignment=TA_LEFT, spaceAfter=10,
    )
    s["cover_sub"] = ParagraphStyle(
        "cover_sub", parent=base["Normal"], fontName="Helvetica",
        fontSize=11, textColor=MUTED, leading=16, alignment=TA_LEFT,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=16, textColor=INK, leading=20, spaceBefore=18, spaceAfter=2,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.5, textColor=INK, leading=15, spaceBefore=12, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Helvetica",
        fontSize=10, textColor=INK, leading=15, spaceAfter=6, alignment=TA_LEFT,
    )
    s["muted"] = ParagraphStyle(
        "muted", parent=s["body"], textColor=MUTED, fontSize=9, leading=13,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=14, bulletIndent=2, spaceAfter=3,
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontName="Courier", fontSize=9,
        textColor=INK, leading=13,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=base["Normal"], fontName="Helvetica", fontSize=9,
        textColor=INK, leading=13,
    )
    s["cellb"] = ParagraphStyle(
        "cellb", parent=s["cell"], fontName="Helvetica-Bold",
    )
    s["toc"] = ParagraphStyle(
        "toc", parent=s["body"], fontSize=10.5, leading=20,
    )
    return s


ST = make_styles()


# --------------------------------------------------------------------------- #
# Bloques reutilizables
# --------------------------------------------------------------------------- #
def heading(text: str):
    """Título de sección con regla negra gruesa debajo."""
    tbl = Table([[Paragraph(text.upper(), ST["h1"])]], colWidths=[16.6 * cm])
    tbl.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 2, INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def sub(text: str):
    return Paragraph(text, ST["h2"])


def para(text: str):
    return Paragraph(text, ST["body"])


def bullets(items: list[str]):
    return [Paragraph(it, ST["bullet"], bulletText="•") for it in items]


def code_block(lines: list[str]):
    """Bloque de código en caja gris con borde negro."""
    text = "<br/>".join(
        ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for ln in lines
    )
    p = Paragraph(text or "&nbsp;", ST["code"])
    tbl = Table([[p]], colWidths=[16.6 * cm])
    tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.4, INK),
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def data_table(header: list[str], rows: list[list[str]], col_widths=None):
    """Tabla con rejilla negra y cabecera en negro sólido."""
    head = [Paragraph(h, ParagraphStyle("hh", parent=ST["cellb"],
                                        textColor=PAPER)) for h in header]
    body = [[Paragraph(c, ST["cell"]) for c in r] for r in rows]
    data = [head] + body
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, INK),
        ("BOX", (0, 0), (-1, -1), 1.6, INK),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


# --------------------------------------------------------------------------- #
# Plantilla: marco negro + pie de página en cada hoja
# --------------------------------------------------------------------------- #
def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    m = 1.2 * cm
    # Marco exterior grueso
    canvas.setLineWidth(2)
    canvas.setStrokeColor(INK)
    canvas.rect(m, m, w - 2 * m, h - 2 * m)
    # Pie: regla + textos
    canvas.setLineWidth(0.8)
    canvas.line(2.0 * cm, 1.7 * cm, w - 2.0 * cm, 1.7 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2.0 * cm, 1.45 * cm,
                      "RECOMENDADOR INTELIGENTE DE PELICULAS")
    canvas.drawRightString(w - 2.0 * cm, 1.45 * cm,
                           f"Pagina {doc.page}")
    canvas.restoreState()


def _decorate_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    m = 1.2 * cm
    canvas.setLineWidth(2.5)
    canvas.setStrokeColor(INK)
    canvas.rect(m, m, w - 2 * m, h - 2 * m)
    canvas.restoreState()


def build_doc(path: Path, story: list):
    doc = BaseDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
        title="Manual - Recomendador Inteligente de Peliculas",
        author="Equipo de Machine Learning",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=_decorate_cover),
        PageTemplate(id="body", frames=[frame], onPage=_decorate),
    ])
    doc.build(story)


# --------------------------------------------------------------------------- #
# Portada
# --------------------------------------------------------------------------- #
def cover(title: str, subtitle: str):
    el = [
        Spacer(1, 5.5 * cm),
        Paragraph("MANUAL", ST["cover_kicker"]),
        Paragraph(title, ST["cover_title"]),
        Spacer(1, 0.2 * cm),
        Table([[None]], colWidths=[16.6 * cm], rowHeights=[2],
              style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 2.5, INK)])),
        Spacer(1, 0.5 * cm),
        Paragraph(subtitle, ST["cover_sub"]),
        Spacer(1, 9 * cm),
        Paragraph(
            f"Version {VERSION} &nbsp;&nbsp;|&nbsp;&nbsp; {DATE}<br/>"
            "Python 3.11+ &nbsp;&nbsp;|&nbsp;&nbsp; Machine Learning, NLP y "
            "busqueda semantica", ST["muted"]),
    ]
    return el


# =========================================================================== #
# MANUAL COMPLETO (Usuario y Sistema)
# =========================================================================== #
def build_full():
    s: list = []
    s += cover(
        "Recomendador Inteligente<br/>de Peliculas",
        "Manual de Usuario y Sistema. Guia completa de funciones, instalacion, "
        "operacion y arquitectura tecnica.")
    s += [NextPageTemplate("body"), PageBreak()]

    # --- Indice ---
    s.append(heading("Contenido"))
    toc = [
        "1.  Introduccion",
        "2.  Requisitos del sistema",
        "3.  Instalacion y configuracion",
        "4.  Funciones principales (usuario)",
        "5.  Funcionamiento interno (sistema / ML)",
        "6.  Arquitectura y modulos",
        "7.  Operacion y mantenimiento",
        "8.  Solucion de problemas",
        "9.  Glosario y stack tecnologico",
    ]
    for t in toc:
        s.append(Paragraph(t, ST["toc"]))

    # --- 1. Introduccion ---
    s.append(heading("1. Introduccion"))
    s.append(para(
        "El <b>Recomendador Inteligente de Peliculas</b> es una aplicacion que sugiere "
        "peliculas similares a partir de una pelicula seleccionada o de una descripcion "
        "escrita en lenguaje natural. Combina un recomendador clasico basado en contenido "
        "(TF-IDF y similitud coseno) con busqueda semantica mediante embeddings."))
    s.append(para(
        "No requiere registro de usuarios, autenticacion ni historial: es una herramienta "
        "directa donde el contenido es el protagonista. El analisis de similitud considera "
        "generos, sinopsis, palabras clave, reparto, director, popularidad y calificaciones."))
    s.append(sub("Para quien es este manual"))
    s.extend(bullets([
        "<b>Usuario final:</b> consulta las secciones 1, 3 (instalacion) y 4 (funciones).",
        "<b>Operador / tecnico:</b> consulta ademas las secciones 5, 6, 7 y 8.",
    ]))

    # --- 2. Requisitos ---
    s.append(heading("2. Requisitos del sistema"))
    s.append(data_table(
        ["Componente", "Requisito"],
        [
            ["Python", "Version 3.11 o superior"],
            ["Memoria RAM", "4 GB minimo (8 GB recomendado con busqueda semantica)"],
            ["Disco", "~2 GB libres (PyTorch + modelo de embeddings)"],
            ["Conexion", "Internet para descargar el dataset y los posters de TMDB"],
            ["Clave TMDB", "API Key (v3 auth) gratuita de The Movie Database"],
        ],
        col_widths=[4.5 * cm, 12.1 * cm]))
    s.append(Spacer(1, 4))
    s.append(Paragraph(
        "La busqueda semantica instala PyTorch (~1-2 GB). Si no se necesita, puede "
        "omitirse y la aplicacion funciona solo con el recomendador clasico.", ST["muted"]))

    # --- 3. Instalacion ---
    s.append(heading("3. Instalacion y configuracion"))
    s.append(sub("3.1  Instalar dependencias"))
    s.append(code_block(["pip install -r requirements.txt"]))
    s.append(Spacer(1, 6))
    s.append(sub("3.2  Configurar la clave de TMDB"))
    s.append(para(
        "Obtenga una API Key (v3 auth) gratuita en themoviedb.org > Ajustes > API. "
        "Copie el archivo de ejemplo y pegue su clave:"))
    s.append(code_block([
        "copy .env.example .env        (Windows)",
        "cp   .env.example .env        (macOS / Linux)",
        "",
        "# dentro de .env:",
        "TMDB_API_KEY=su_clave_real",
    ]))
    s.append(Spacer(1, 6))
    s.append(para(
        "La clave tambien se admite como variable de entorno <font face='Courier'>"
        "TMDB_API_KEY</font> o mediante <font face='Courier'>st.secrets</font> de Streamlit."))
    s.append(sub("3.3  Entrenar los modelos"))
    s.append(para(
        "Descarga el dataset desde TMDB y genera todos los modelos (una sola vez):"))
    s.append(code_block(["python train.py"]))
    s.append(Spacer(1, 4))
    s.append(para("Opciones utiles del entrenamiento:"))
    s.append(data_table(
        ["Comando", "Efecto"],
        [
            ["python train.py --pages 50", "Descarga mas rapida (~1000 peliculas)"],
            ["python train.py --refresh", "Vuelve a descargar el dataset"],
            ["python train.py --no-semantic", "Omite embeddings / PyTorch (mas ligero)"],
            ["python train.py --clusters 10", "Numero de grupos de KMeans"],
        ],
        col_widths=[6.6 * cm, 10.0 * cm]))
    s.append(Spacer(1, 6))
    s.append(sub("3.4  Iniciar la aplicacion"))
    s.append(code_block(["streamlit run app.py"]))
    s.append(Spacer(1, 4))
    s.append(para(
        "La aplicacion abre en el navegador. El orden correcto siempre es: "
        "<b>instalar &gt; configurar clave &gt; entrenar &gt; ejecutar</b>."))

    # --- 4. Funciones principales ---
    s.append(heading("4. Funciones principales"))
    s.append(para(
        "La interfaz se organiza en tres pestanas, con una barra lateral que muestra el "
        "estado del sistema (peliculas en catalogo, disponibilidad de busqueda semantica "
        "y conexion con TMDB)."))

    s.append(sub("4.1  Recomendar por pelicula"))
    s.append(para(
        "Seleccione una pelicula en el buscador, elija cuantas recomendaciones desea "
        "(1 a 20) y pulse <b>Recomendar</b>. El sistema devuelve las peliculas mas "
        "parecidas ordenadas por porcentaje de similitud."))
    s.extend(bullets([
        "Busqueda tolerante a errores tipograficos en el titulo.",
        "Cada resultado se muestra como una ficha con poster, titulo, ano, generos, "
        "rating, popularidad y porcentaje de similitud.",
        "Enlace directo al trailer cuando esta disponible.",
    ]))

    s.append(sub("4.2  Busqueda semantica"))
    s.append(para(
        "Permite buscar por una descripcion en lenguaje natural en lugar de un titulo. "
        "Por ejemplo:"))
    s.append(code_block([
        '"una pelicula de ciencia ficcion con viajes en el tiempo"']))
    s.append(Spacer(1, 4))
    s.append(para(
        "El sistema interpreta el significado de la frase (no solo las palabras exactas) "
        "y devuelve las peliculas mas afines. Si los embeddings no estan entrenados, "
        "la busqueda recurre automaticamente a la coincidencia por titulo."))

    s.append(sub("4.3  Explicabilidad"))
    s.append(para(
        "En cada recomendacion por pelicula, el desplegable <b>Por que te la recomendamos</b> "
        "indica los atributos compartidos con la pelicula de origen:"))
    s.extend(bullets([
        "<b>Genero</b> en comun.",
        "<b>Director</b> coincidente.",
        "<b>Tematica</b> (palabras clave compartidas).",
        "<b>Actores</b> en comun.",
        "Pertenencia al mismo grupo tematico (cluster).",
    ]))

    s.append(sub("4.4  Dashboard de analisis"))
    s.append(para("Visualizaciones del catalogo completo:"))
    s.extend(bullets([
        "Generos mas frecuentes.",
        "Distribucion de calificaciones (ratings).",
        "Top de peliculas por popularidad.",
        "Mapa de calor de la matriz de similitud.",
        "Mapa de clusters (KMeans) proyectado en 2D mediante PCA.",
    ]))

    # --- 5. Funcionamiento interno ---
    s.append(heading("5. Funcionamiento interno (sistema / ML)"))
    s.append(para(
        "El motor de recomendacion se apoya en una tuberia de procesamiento de lenguaje "
        "natural y aprendizaje automatico que se ejecuta durante el entrenamiento."))
    s.append(sub("5.1  Preprocesamiento"))
    s.extend(bullets([
        "Eliminacion de registros sin titulo o sin sinopsis.",
        "Limpieza de texto (minusculas, sin puntuacion, espacios normalizados).",
        "Conversion de generos, keywords y reparto a texto homogeneo.",
        "Combinacion de generos + keywords + reparto + director + sinopsis en una "
        "unica columna <font face='Courier'>combined_features</font>.",
    ]))
    s.append(sub("5.2  Vectorizacion y similitud"))
    s.append(para(
        "El texto combinado se vectoriza con <font face='Courier'>TfidfVectorizer</font> "
        "(stop words en ingles, hasta 10.000 caracteristicas). La afinidad entre peliculas "
        "se calcula con <b>similitud coseno</b>, produciendo una matriz N x N que se "
        "persiste para respuestas instantaneas."))
    s.append(sub("5.3  Busqueda semantica (embeddings)"))
    s.append(para(
        "El modelo <font face='Courier'>all-MiniLM-L6-v2</font> (SentenceTransformers) "
        "convierte cada pelicula y cada consulta en un vector denso de significado. La "
        "consulta se compara con todo el catalogo por similitud coseno."))
    s.append(sub("5.4  Clustering y reduccion de dimensionalidad"))
    s.append(para(
        "<b>KMeans</b> agrupa las peliculas en categorias ocultas y <b>PCA</b> proyecta "
        "esos grupos a dos dimensiones para visualizarlos en el dashboard."))

    # --- 6. Arquitectura ---
    s.append(heading("6. Arquitectura y modulos"))
    s.append(data_table(
        ["Archivo", "Responsabilidad"],
        [
            ["app.py", "Interfaz Streamlit (3 pestanas) y renderizado de fichas"],
            ["train.py", "Pipeline de entrenamiento y persistencia de modelos"],
            ["recommender.py", "Motor: recommend_movies, semantic_search, explain"],
            ["data_preprocessing.py", "Limpieza, parseo y combined_features"],
            ["tmdb_api.py", "Cliente TMDB: dataset y media en vivo, con cache"],
            ["visualizations.py", "Graficos del dashboard (monocromos)"],
            ["config.py", "Rutas, constantes y carga de la clave de API"],
            ["models/", "Artefactos entrenados (.pkl, .parquet)"],
            ["data/movies.csv", "Dataset generado desde TMDB"],
        ],
        col_widths=[5.0 * cm, 11.6 * cm]))
    s.append(Spacer(1, 6))
    s.append(sub("Artefactos generados por el entrenamiento"))
    s.extend(bullets([
        "<font face='Courier'>tfidf_model.pkl</font> &#8212; vectorizador entrenado.",
        "<font face='Courier'>similarity_matrix.pkl</font> &#8212; matriz de similitud N x N.",
        "<font face='Courier'>embeddings.pkl</font> &#8212; vectores semanticos.",
        "<font face='Courier'>kmeans.pkl</font> &#8212; modelo de clustering.",
        "<font face='Courier'>movies_processed.parquet</font> &#8212; catalogo procesado.",
    ]))

    # --- 7. Operacion ---
    s.append(heading("7. Operacion y mantenimiento"))
    s.extend(bullets([
        "<b>Actualizar el catalogo:</b> ejecute <font face='Courier'>python train.py "
        "--refresh</font> periodicamente para reflejar estrenos recientes.",
        "<b>Cache:</b> las peticiones a TMDB se almacenan en <font face='Courier'>.cache/"
        "</font> (una semana); la app cachea media y modelos en memoria.",
        "<b>Rendimiento:</b> la matriz de similitud crece con el cuadrado del numero de "
        "peliculas (~100 MB a 5.000 titulos). Ajuste <font face='Courier'>--pages</font> "
        "segun los recursos disponibles.",
        "<b>Modo ligero:</b> entrene con <font face='Courier'>--no-semantic</font> para "
        "prescindir de PyTorch.",
    ]))

    # --- 8. Problemas ---
    s.append(heading("8. Solucion de problemas"))
    s.append(data_table(
        ["Sintoma", "Solucion"],
        [
            ["\"No hay modelos entrenados\" en la app",
             "Ejecute python train.py antes de streamlit run app.py."],
            ["\"No hay TMDB_API_KEY\" al entrenar",
             "Cree el archivo .env con su clave o defina la variable de entorno."],
            ["Descarga lenta o errores 429",
             "TMDB limita el ritmo; reduzca --pages. El cliente reintenta solo."],
            ["Posters no aparecen",
             "Compruebe la clave y la conexion; sin clave se muestran marcadores."],
            ["Instalacion de PyTorch muy pesada",
             "Use python train.py --no-semantic para una version ligera."],
        ],
        col_widths=[6.2 * cm, 10.4 * cm]))

    # --- 9. Glosario ---
    s.append(heading("9. Glosario y stack tecnologico"))
    s.append(data_table(
        ["Termino", "Definicion"],
        [
            ["TF-IDF", "Tecnica que pondera la importancia de cada palabra en un texto."],
            ["Similitud coseno", "Medida de parecido entre dos vectores (0 a 1)."],
            ["Embedding", "Vector numerico que captura el significado de un texto."],
            ["KMeans", "Algoritmo que agrupa elementos en k categorias."],
            ["PCA", "Reduccion de dimensiones para visualizar datos complejos."],
            ["TMDB", "The Movie Database: fuente de datos y posters."],
        ],
        col_widths=[4.2 * cm, 12.4 * cm]))
    s.append(Spacer(1, 8))
    s.append(Paragraph(
        "<b>Stack:</b> Python &#183; Pandas &#183; NumPy &#183; scikit-learn &#183; "
        "Streamlit &#183; Joblib &#183; Matplotlib &#183; Seaborn &#183; "
        "SentenceTransformers &#183; TMDB API", ST["muted"]))

    build_doc(DOCS / "Manual_Usuario_y_Sistema.pdf", s)
    print("OK -> Manual_Usuario_y_Sistema.pdf")


# =========================================================================== #
# GUIA RAPIDA (resumida)
# =========================================================================== #
def build_quick():
    s: list = []
    s += cover(
        "Guia Rapida",
        "Recomendador Inteligente de Peliculas. Lo esencial para instalar, "
        "ejecutar y usar la aplicacion en una pagina.")
    s += [NextPageTemplate("body"), PageBreak()]

    s.append(heading("Que es"))
    s.append(para(
        "Aplicacion que recomienda peliculas similares a partir de un titulo o de una "
        "descripcion en lenguaje natural, usando Machine Learning, NLP y busqueda "
        "semantica. Sin registro de usuarios."))

    s.append(heading("Puesta en marcha en 4 pasos"))
    s.append(data_table(
        ["Paso", "Comando / accion"],
        [
            ["1. Instalar", "pip install -r requirements.txt"],
            ["2. Configurar clave", "Copie .env.example a .env y ponga su TMDB_API_KEY"],
            ["3. Entrenar", "python train.py"],
            ["4. Ejecutar", "streamlit run app.py"],
        ],
        col_widths=[4.2 * cm, 12.4 * cm]))
    s.append(Spacer(1, 6))
    s.append(Paragraph(
        "Clave gratuita de TMDB en themoviedb.org &#62; Ajustes &#62; API "
        "(API Key v3 auth).", ST["muted"]))

    s.append(heading("Las tres pantallas"))
    s.append(data_table(
        ["Pestana", "Para que sirve"],
        [
            ["Recomendar",
             "Elija una pelicula y reciba las mas parecidas, con poster, datos y "
             "porcentaje de similitud. Incluye el motivo de cada recomendacion."],
            ["Busqueda semantica",
             "Escriba una frase (p. ej. \"ciencia ficcion con viajes en el tiempo\") "
             "y obtenga peliculas afines por significado."],
            ["Dashboard",
             "Graficos del catalogo: generos, ratings, popularidad, matriz de "
             "similitud y clusters."],
        ],
        col_widths=[4.6 * cm, 12.0 * cm]))

    s.append(heading("Comandos utiles"))
    s.append(data_table(
        ["Comando", "Efecto"],
        [
            ["python train.py --pages 50", "Entrenamiento rapido (~1000 peliculas)"],
            ["python train.py --refresh", "Actualizar el dataset"],
            ["python train.py --no-semantic", "Version ligera, sin PyTorch"],
        ],
        col_widths=[6.6 * cm, 10.0 * cm]))

    s.append(heading("Problemas frecuentes"))
    s.append(data_table(
        ["Sintoma", "Solucion"],
        [
            ["\"No hay modelos entrenados\"", "Ejecute python train.py primero."],
            ["\"No hay TMDB_API_KEY\"", "Cree .env con su clave."],
            ["Sin posters", "Revise la clave y la conexion a internet."],
        ],
        col_widths=[6.2 * cm, 10.4 * cm]))
    s.append(Spacer(1, 10))
    s.append(Paragraph(
        "Para detalle completo de funciones, arquitectura y operacion, consulte el "
        "<b>Manual de Usuario y Sistema</b>.", ST["muted"]))

    build_doc(DOCS / "Guia_Rapida.pdf", s)
    print("OK -> Guia_Rapida.pdf")


if __name__ == "__main__":
    build_full()
    build_quick()
    print("Manuales generados en", DOCS)
