import io
import re
import unicodedata
from difflib import SequenceMatcher
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Super Base Aspirantes",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Super Base Integral de Aspirantes")
st.caption(
    "Integra Historial de Aspirantes, EVALUATEC y CHASIDE en un solo archivo Excel."
)


# ============================================================
# CONSTANTES EVALUATEC
# ============================================================

EVAL_ETIQUETAS_AREAS = {
    "ING": "Inglés",
    "MAT": "Matemáticas",
    "COM": "Comprensión lectora",
    "RLM": "Razonamiento lógico-matemático",
    "PM": "Pensamiento matemático",
    "ARQ": "Arquitectura",
    "FIS": "Física",
    "ADMN": "Administración"
}

EVAL_ORDEN_AREAS = [
    "ING",
    "MAT",
    "COM",
    "RLM",
    "PM",
    "FIS",
    "ARQ",
    "ADMN"
]

EVAL_BLOQUES = {
    "ADM": "Administración",
    "ARQ": "Arquitectura",
    "ING": "Ingeniería"
}


# ============================================================
# CONSTANTES CHASIDE
# ============================================================

CHASIDE_AREAS = ["C", "H", "A", "S", "I", "D", "E"]

CHASIDE_AREAS_LONG = {
    "C": "Administrativo",
    "H": "Humanidades y Sociales",
    "A": "Artístico",
    "S": "Ciencias de la Salud",
    "I": "Enseñanzas Técnicas",
    "D": "Defensa y Seguridad",
    "E": "Ciencias Experimentales"
}

CHASIDE_INTERESES_ITEMS = {
    "C": [1, 12, 20, 53, 64, 71, 78, 85, 91, 98],
    "H": [9, 25, 34, 41, 56, 67, 74, 80, 89, 95],
    "A": [3, 11, 21, 28, 36, 45, 50, 57, 81, 96],
    "S": [8, 16, 23, 33, 44, 52, 62, 70, 87, 92],
    "I": [6, 19, 27, 38, 47, 54, 60, 75, 83, 97],
    "D": [5, 14, 24, 31, 37, 48, 58, 65, 73, 84],
    "E": [17, 32, 35, 42, 49, 61, 68, 77, 88, 93]
}

CHASIDE_APTITUDES_ITEMS = {
    "C": [2, 15, 46, 51],
    "H": [30, 63, 72, 86],
    "A": [22, 39, 76, 82],
    "S": [4, 29, 40, 69],
    "I": [10, 26, 59, 90],
    "D": [13, 18, 43, 66],
    "E": [7, 55, 79, 94]
}

CHASIDE_PERFILES_CARRERA = {
    "Arquitectura": ["A", "I", "C"],
    "Contador Público": ["C", "D"],
    "Licenciatura en Administración": ["C", "D"],
    "Ingeniería Ambiental": ["I", "C", "E"],
    "Ingeniería Bioquímica": ["I", "C", "E"],
    "Ingeniería en Gestión Empresarial": ["C", "D", "H"],
    "Ingeniería Industrial": ["C", "D", "H"],
    "Ingeniería en Inteligencia Artificial": ["I", "E"],
    "Ingeniería Mecatrónica": ["I", "E"],
    "Ingeniería en Sistemas Computacionales": ["I", "E"]
}

CHASIDE_COLUMNA_NOMBRE = "Ingrese su nombre completo"
CHASIDE_COLUMNA_CARRERA = "¿A qué carrera desea ingresar?"
CHASIDE_COLUMNA_EMAIL_1 = "Dirección de correo electrónico"
CHASIDE_COLUMNA_EMAIL_2 = "Escriba su correo electrónico"

# ============================================================
# LINKS PRECARGADOS EDITABLES
# ============================================================

LINK_HISTORIAL_DEFAULT = "https://docs.google.com/spreadsheets/d/1rthlRH1NiVCb7d5dr45mhQuC3We628ms/edit?usp=sharing&ouid=101744927034742701111&rtpof=true&sd=true"

LINK_CHASIDE_DEFAULT = "https://docs.google.com/spreadsheets/d/1YHMEb5hftOZfV-CMWoUsUgJh1xmsgTY3YYwAtq1dGQA/edit?resourcekey=&gid=1491376423#gid=1491376423"

LINK_EVALUATEC_ADM_DEFAULT = "https://drive.google.com/file/d/1OLECyh4lb578nJw_w00os-TdKEh7kLLN/view?usp=sharing"
LINK_EVALUATEC_ARQ_DEFAULT = "https://drive.google.com/file/d/1jE_YYsT0kk56EiGP3EwAa1w29Yd8wX2G/view?usp=share_link"
LINK_EVALUATEC_ING_DEFAULT = "https://drive.google.com/file/d/1iBUu338DgspUkSXhtuIaDs6h8F4cbIxX/view?usp=sharing"

# ============================================================
# UTILIDADES GENERALES
# ============================================================

def util_normalizar_texto(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return " ".join(texto.split())


def util_limpiar_texto(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    return re.sub(r"\s+", " ", texto)


def util_limpiar_texto_visible(valor):
    if pd.isna(valor):
        return ""

    texto = str(valor).replace("\n", " ")
    return re.sub(r"\s+", " ", texto).strip()


def util_encontrar_columna(df, posibles_nombres):
    """
    Busca una columna ignorando mayúsculas, acentos y espacios.
    """

    columnas_normalizadas = {
        util_limpiar_texto(columna): columna
        for columna in df.columns
    }

    for posible in posibles_nombres:
        posible_limpio = util_limpiar_texto(posible)

        if posible_limpio in columnas_normalizadas:
            return columnas_normalizadas[posible_limpio]

        for columna_limpia, columna_original in columnas_normalizadas.items():
            if posible_limpio in columna_limpia:
                return columna_original

    return None


def normalizar_nombre(valor):
    """
    Normaliza nombres para cruzar Historial, EVALUATEC y CHASIDE.
    """

    if pd.isna(valor):
        return ""

    texto = str(valor).upper().strip()
    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter
        for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = re.sub(r"[^A-ZÑ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def nombre_visible(valor):
    """
    Convierte nombres a formato visible.
    """

    if pd.isna(valor):
        return "Sin nombre"

    texto = str(valor).strip()
    texto = re.sub(r"\s+", " ", texto)

    return texto.title()

def normalizar_correo(valor):
    """
    Normaliza correos electrónicos.
    """

    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()

    if texto in ["", "nan", "none", "sin dato"]:
        return ""

    return texto

# ============================================================
# LECTURA DE ARCHIVOS DESDE LINKS
# ============================================================

class ArchivoDesdeURL:
    """
    Simula un archivo cargado en Streamlit.
    Sirve para que EVALUATEC pueda usar archivo.name y archivo.getvalue().
    """

    def __init__(self, contenido, nombre):
        self._contenido = contenido
        self.name = nombre

    def getvalue(self):
        return self._contenido


def extraer_id_google_drive(url):
    """
    Extrae ID de archivo desde links de Google Drive o Google Sheets.
    """

    url = str(url).strip()

    if "/d/" in url:
        return url.split("/d/")[1].split("/")[0]

    if "id=" in url:
        return url.split("id=")[1].split("&")[0]

    return None


def transformar_link_google_sheets_xlsx(url):
    """
    Convierte link editable de Google Sheets a descarga XLSX.
    """

    file_id = extraer_id_google_drive(url)

    if file_id is None:
        return url

    return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"


def transformar_link_drive_descarga(url):
    """
    Convierte link de Google Drive a descarga directa.
    """

    url = str(url).strip()

    if "drive.google.com" not in url:
        return url

    file_id = extraer_id_google_drive(url)

    if file_id is None:
        return url

    return f"https://drive.google.com/uc?export=download&id={file_id}"


def descargar_archivo_url(url):
    """
    Descarga archivo desde URL pública.
    """

    import urllib.request

    url = str(url).strip()

    if url == "":
        return None

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(req) as response:
        return response.read()


def obtener_contenido_historial_desde_link_o_upload(url_historial, archivo_historial):
    """
    Usa primero el link editable.
    Si el link está vacío, usa el archivo cargado manualmente.
    """

    if str(url_historial).strip() != "":
        url_descarga = transformar_link_google_sheets_xlsx(url_historial)
        return descargar_archivo_url(url_descarga)

    if archivo_historial is not None:
        return archivo_historial.getvalue()

    return None


def obtener_archivos_evaluatec_desde_links_o_uploads(
    url_adm,
    url_arq,
    url_ing,
    archivos_evaluatec
):
    """
    Usa links editables para los 3 CSV de EVALUATEC.
    Si los links están vacíos, usa carga manual.
    """

    archivos_desde_links = []

    links = [
        (url_adm, "EVALUATEC Administración.csv"),
        (url_arq, "EVALUATEC Arquitectura.csv"),
        (url_ing, "EVALUATEC Ingeniería.csv")
    ]

    for url, nombre_archivo in links:

        if str(url).strip() == "":
            continue

        url_descarga = transformar_link_drive_descarga(url)
        contenido = descargar_archivo_url(url_descarga)

        archivos_desde_links.append(
            ArchivoDesdeURL(
                contenido=contenido,
                nombre=nombre_archivo
            )
        )

    if len(archivos_desde_links) > 0:
        return archivos_desde_links

    if archivos_evaluatec:
        return archivos_evaluatec

    return []

def simplificar_carrera(valor):
    """
    Normaliza carreras para cruces flexibles.
    """

    if pd.isna(valor):
        return ""

    texto = util_limpiar_texto(valor)

    reemplazos = [
        "licenciatura en",
        "lic. en",
        "licenciatura",
        "lic ",
        "ingenieria en",
        "ingeniería en",
        "ing. en",
        "ing ",
        "carrera de",
        "programa de"
    ]

    for reemplazo in reemplazos:
        texto = texto.replace(reemplazo, "")

    texto = texto.replace(".", " ")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def score_nombre_tokens(nombre_a, nombre_b):
    """
    Calcula similitud flexible entre nombres.
    Sirve cuando cambia el orden o falta un apellido.
    """

    nombre_a = normalizar_nombre(nombre_a)
    nombre_b = normalizar_nombre(nombre_b)

    if nombre_a == "" or nombre_b == "":
        return 0

    tokens_a = set(nombre_a.split())
    tokens_b = set(nombre_b.split())

    tokens_a = {token for token in tokens_a if len(token) >= 3}
    tokens_b = {token for token in tokens_b if len(token) >= 3}

    if not tokens_a or not tokens_b:
        return 0

    interseccion = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))

    score_tokens = interseccion / union

    score_texto = SequenceMatcher(
        None,
        " ".join(sorted(tokens_a)),
        " ".join(sorted(tokens_b))
    ).ratio()

    return max(score_tokens, score_texto)


def formato_porcentaje(valor):
    """
    Formatea números como porcentaje.
    """

    if pd.isna(valor):
        return "Sin dato"

    try:
        return f"{float(valor):.1f}%"
    except Exception:
        return "Sin dato"

def valor_seguro(fila, columna, default="Sin dato"):
    if columna not in fila.index:
        return default

    valor = fila[columna]

    if pd.isna(valor) or str(valor).strip() == "":
        return default

    return valor


# ============================================================
# HISTORIAL DE ASPIRANTES
# ============================================================

def hist_nombres_unicos(encabezados):
    """
    Evita columnas duplicadas al leer hojas de Excel.
    """

    usados = {}
    resultado = []

    for posicion, encabezado in enumerate(encabezados, start=1):

        if pd.isna(encabezado) or str(encabezado).strip() == "":
            nombre = f"Columna_sin_nombre_{posicion}"
        else:
            nombre = str(encabezado).strip()

        if nombre in usados:
            usados[nombre] += 1
            nombre = f"{nombre}_{usados[nombre]}"
        else:
            usados[nombre] = 1

        resultado.append(nombre)

    return resultado


def hist_buscar_fila_encabezados(df_crudo):
    """
    Localiza la fila donde inician los encabezados reales.
    """

    palabras_clave = [
        "matricula/id",
        "matricula",
        "id",
        "apellido paterno",
        "apellido materno",
        "nombre (s)",
        "nombre"
    ]

    limite = min(len(df_crudo), 40)

    for indice in range(limite):

        valores = [
            util_limpiar_texto(valor)
            for valor in df_crudo.iloc[indice].tolist()
        ]

        coincidencias = sum(
            any(palabra in valor for valor in valores)
            for palabra in palabras_clave
        )

        if coincidencias >= 2:
            return indice

    return None


def hist_obtener_nombre_carrera(nombre_hoja, df_crudo):
    """
    Intenta obtener el nombre de carrera desde el contenido de la hoja.
    Si no lo encuentra, usa el nombre de la hoja.
    """

    limite = min(len(df_crudo), 15)

    for indice in range(limite):

        fila = df_crudo.iloc[indice].tolist()

        for posicion, valor in enumerate(fila):

            if util_limpiar_texto(valor) == "carrera":

                if posicion + 1 < len(fila):
                    posible_carrera = fila[posicion + 1]

                    if pd.notna(posible_carrera):
                        return str(posible_carrera).strip()

    return str(nombre_hoja).strip()


def hist_convertir_promedio(valor):
    """
    Convierte promedio de bachillerato a escala 0-100.
    """

    if pd.isna(valor) or str(valor).strip() == "":
        return np.nan, "Sin dato"

    if isinstance(valor, (datetime, date, pd.Timestamp)):
        return np.nan, "Dato dudoso: formato fecha"

    texto = str(valor).strip()
    texto = texto.replace("\xa0", " ")
    texto = texto.replace(",", ".")
    texto = texto.lstrip("'").strip()

    try:
        numero = float(texto)

    except (TypeError, ValueError):
        return np.nan, "Dato dudoso: no numérico"

    if 0 <= numero <= 10:
        return round(numero * 10, 2), "Convertido de escala 0-10"

    if 10 < numero <= 100:
        return round(numero, 2), "Válido: escala 0-100"

    return np.nan, "Dato dudoso: fuera de rango"


def hist_clasificar_rango_promedio(valor):
    """
    Clasifica promedio de bachillerato.
    """

    if pd.isna(valor):
        return "Sin dato"

    if 60 <= valor < 70:
        return "60-69"

    if 70 <= valor < 80:
        return "70-79"

    if 80 <= valor < 90:
        return "80-89"

    if 90 <= valor <= 100:
        return "90-100"

    return "Fuera de rango"


def hist_normalizar_sexo(valor):
    """
    Normaliza sexo/género cuando esté disponible.
    """

    if pd.isna(valor):
        return "Sin especificar"

    texto = util_limpiar_texto(valor)

    if texto in ["hombre", "masculino", "m", "male"]:
        return "Hombre"

    if texto in ["mujer", "femenino", "f", "female"]:
        return "Mujer"

    return "Sin especificar"


def hist_clasificar_estado_procedencia(valor):
    """
    Clasifica estado de procedencia a partir del texto de escuela.
    """

    if pd.isna(valor):
        return "Sin dato"

    texto = util_limpiar_texto(valor)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    palabras_jalisco = [
        "jalisco",
        "tuxpan",
        "cihuatlan",
        "autlan",
        "guadalajara",
        "zapopan",
        "tonala",
        "sayula",
        "zapotiltic",
        "zapotlan",
        "ciudad guzman",
        "tequila",
        "casimiro castillo",
        "el grullo",
        "union de tula",
        "tamazula",
        "teocuitatlan",
        "universidad de guadalajara",
        "udeg"
    ]

    if any(palabra in texto for palabra in palabras_jalisco):
        return "Jalisco"

    palabras_michoacan = [
        "michoacan",
        "coahuayana",
        "coalcoman",
        "morelia",
        "zamora",
        "lazaro cardenas",
        "uruapan",
        "apatzingan",
        "maravatio"
    ]

    if any(palabra in texto for palabra in palabras_michoacan):
        return "Michoacán"

    palabras_nayarit = [
        "nayarit",
        "tepic",
        "bahia de banderas",
        "santiago ixcuintla",
        "compostela"
    ]

    if any(palabra in texto for palabra in palabras_nayarit):
        return "Nayarit"

    palabras_guanajuato = [
        "guanajuato",
        "leon",
        "irapuato",
        "celaya",
        "salamanca"
    ]

    if any(palabra in texto for palabra in palabras_guanajuato):
        return "Guanajuato"

    if "nuevo leon" in texto or "monterrey" in texto:
        return "Nuevo León"

    if "sinaloa" in texto or "culiacan" in texto:
        return "Sinaloa"

    if "durango" in texto:
        return "Durango"

    if "sonora" in texto or "hermosillo" in texto:
        return "Sonora"

    if "baja california" in texto or "tijuana" in texto:
        return "Baja California"

    if "veracruz" in texto:
        return "Veracruz"

    if "ciudad de mexico" in texto or "cdmx" in texto:
        return "Ciudad de México"

    if any(
        palabra in texto
        for palabra in ["canada", "canadá", "usa", "united states"]
    ):
        return "Internacional"

    return "Colima"


def hist_obtener_numero_institucion(texto, expresiones):
    """
    Extrae número de bachillerato cuando aparece en texto.
    """

    for expresion in expresiones:

        coincidencia = re.search(expresion, texto)

        if coincidencia:
            return coincidencia.group(1)

    return None


def hist_normalizar_escuela_procedencia(valor):
    """
    Normaliza escuela de procedencia.
    """

    if pd.isna(valor):
        return "Sin dato"

    texto_visible = util_limpiar_texto_visible(valor)
    texto = util_limpiar_texto(valor)
    texto_compacto = re.sub(r"[^a-z0-9]", "", texto)

    if texto in ["", "nan", "none", "escuela de procedencia"]:
        return "Sin dato"

    if (
        "universidad de colima" in texto
        or "u de c" in texto
        or "udec" in texto
        or "bachillerato udec" in texto
        or re.search(r"\bbachillerato\s*([1-9]|[12][0-9]|30)\b", texto)
    ):
        return "Universidad de Colima (U de C)"

    if (
        "telebachillerato" in texto
        or "tele bachillerato" in texto
        or "telebach" in texto
        or "telebach" in texto_compacto
    ):
        return "Telebachillerato"

    if (
        "colegio de bachilleres" in texto
        or "colegio bachilleres" in texto
        or "colegio de bach" in texto
        or "colegio bach" in texto
        or "cobach" in texto_compacto
        or "coba" in texto_compacto
    ):
        return "Colegio de Bachilleres"

    if "cbtis" in texto_compacto or "cbti" in texto_compacto:

        numero = hist_obtener_numero_institucion(
            texto,
            [
                r"cbtis\s*#?\s*(\d+)",
                r"cbti[s]?\s*#?\s*(\d+)"
            ]
        )

        if numero:
            return f"CBTis {numero}"

        return "CBTis"

    if "cetis" in texto_compacto:

        numero = hist_obtener_numero_institucion(
            texto,
            [r"cetis\s*#?\s*(\d+)"]
        )

        if numero:
            return f"CETis {numero}"

        return "CETis"

    if "cbta" in texto_compacto:

        numero = hist_obtener_numero_institucion(
            texto,
            [r"cbta\s*#?\s*(\d+)"]
        )

        if numero:
            return f"CBTA {numero}"

        return "CBTA"

    if "emsad" in texto_compacto:

        numero = hist_obtener_numero_institucion(
            texto,
            [r"emsad\s*#?\s*(\d+)"]
        )

        if numero:
            return f"EMSAD {numero}"

        return "EMSAD"

    if "isenco" in texto_compacto:
        return "ISENCO"

    if "conalep" in texto_compacto:
        return "CONALEP"

    if "cecyte" in texto_compacto:
        return "CECyTE"

    if "icep" in texto_compacto:
        return "ICEP"

    if (
        "universidad de guadalajara" in texto
        or "udeg" in texto_compacto
        or "prepa regional tuxpan" in texto
    ):
        return "Universidad de Guadalajara (UdeG)"

    if "anahuac" in texto:
        return "Preparatoria Anáhuac"

    if "campoverde" in texto_compacto or "campo verde" in texto:
        return "Colegio Campoverde"

    if "adonai" in texto:
        return "Instituto Adonai"

    if "prepa en linea" in texto:
        return "Prepa en Línea SEP"

    if "acredita" in texto and "bach" in texto:
        return "Acredita-Bach SEP"

    return texto_visible.title()


def hist_encontrar_nombre_historial(df):
    """
    Detecta apellido paterno, apellido materno y nombre.
    """

    col_apellido_paterno = util_encontrar_columna(
        df,
        [
            "Apellido paterno",
            "Primer apellido",
            "Paterno"
        ]
    )

    col_apellido_materno = util_encontrar_columna(
        df,
        [
            "Apellido materno",
            "Segundo apellido",
            "Materno"
        ]
    )

    col_nombre = util_encontrar_columna(
        df,
        [
            "Nombre (s)",
            "Nombre(s)",
            "Nombres",
            "Nombre"
        ]
    )

    return col_apellido_paterno, col_apellido_materno, col_nombre


def hist_procesar_hoja(contenido_archivo, nombre_hoja):
    """
    Procesa una hoja del Excel de Historial.
    """

    archivo = io.BytesIO(contenido_archivo)

    df_crudo = pd.read_excel(
        archivo,
        sheet_name=nombre_hoja,
        header=None,
        dtype=object
    )

    fila_encabezados = hist_buscar_fila_encabezados(df_crudo)

    if fila_encabezados is None:
        return None, {
            "Hoja": nombre_hoja,
            "Estatus": "No procesada",
            "Detalle": "No se identificó una fila de encabezados."
        }

    carrera = hist_obtener_nombre_carrera(nombre_hoja, df_crudo)

    encabezados = hist_nombres_unicos(
        df_crudo.iloc[fila_encabezados].tolist()
    )

    df = df_crudo.iloc[fila_encabezados + 1:].copy()
    df.columns = encabezados
    df = df.dropna(how="all").copy()

    columna_id = util_encontrar_columna(
        df,
        [
            "Matrícula/ID",
            "Matricula/ID",
            "Matrícula",
            "Matricula",
            "ID"
        ]
    )

    if columna_id is not None:
        df = df[df[columna_id].notna()].copy()
        df["Matrícula/ID"] = df[columna_id]
    else:
        df["Matrícula/ID"] = "Sin dato"

    df["Carrera historial"] = carrera
    df["Hoja historial"] = nombre_hoja

    columna_promedio = util_encontrar_columna(
        df,
        [
            "Promedio Bachillerato",
            "Promedio de Bachillerato",
            "Promedio"
        ]
    )

    if columna_promedio is not None:

        df["Promedio bachillerato original"] = df[columna_promedio]

        resultado = df[columna_promedio].apply(hist_convertir_promedio)

        df["Promedio bachillerato 100"] = resultado.apply(
            lambda x: x[0]
        )

        df["Estatus promedio bachillerato"] = resultado.apply(
            lambda x: x[1]
        )

    else:
        df["Promedio bachillerato original"] = np.nan
        df["Promedio bachillerato 100"] = np.nan
        df["Estatus promedio bachillerato"] = "No se encontró columna de promedio"

    col_apellido_paterno, col_apellido_materno, col_nombre = hist_encontrar_nombre_historial(df)

    if col_apellido_paterno is not None and col_nombre is not None:

        if col_apellido_materno is None:
            df["Nombre completo historial"] = (
                df[col_apellido_paterno].fillna("").astype(str)
                + " "
                + df[col_nombre].fillna("").astype(str)
            )
        else:
            df["Nombre completo historial"] = (
                df[col_apellido_paterno].fillna("").astype(str)
                + " "
                + df[col_apellido_materno].fillna("").astype(str)
                + " "
                + df[col_nombre].fillna("").astype(str)
            )

    else:
        df["Nombre completo historial"] = "Sin nombre"

    df["Nombre visible"] = df["Nombre completo historial"].apply(nombre_visible)
    df["Nombre match"] = df["Nombre completo historial"].apply(normalizar_nombre)

    df["Carrera match historial"] = df["Carrera historial"].apply(simplificar_carrera)

    columna_sexo = util_encontrar_columna(
        df,
        ["Género", "Genero", "Sexo"]
    )

    if columna_sexo is not None:
        df["Sexo"] = df[columna_sexo].apply(hist_normalizar_sexo)
    else:
        df["Sexo"] = "Sin especificar"

    columna_escuela = util_encontrar_columna(
        df,
        [
            "Escuela de Procedencia",
            "Escuela Procedencia",
            "Procedencia",
            "Escuela"
        ]
    )

    if columna_escuela is not None:

        df["Escuela de procedencia original"] = df[
            columna_escuela
        ].fillna("Sin dato").astype(str)

        df["Escuela de procedencia normalizada"] = df[
            columna_escuela
        ].apply(hist_normalizar_escuela_procedencia)

        df["Estado de procedencia"] = df[
            columna_escuela
        ].apply(hist_clasificar_estado_procedencia)

    else:
        df["Escuela de procedencia original"] = "Sin dato"
        df["Escuela de procedencia normalizada"] = "Sin dato"
        df["Estado de procedencia"] = "Sin dato"

    df["Rango promedio bachillerato"] = df[
        "Promedio bachillerato 100"
    ].apply(hist_clasificar_rango_promedio)

    df = df[
        df["Nombre match"].notna()
        &
        (df["Nombre match"].astype(str).str.strip() != "")
        &
        (~df["Nombre match"].astype(str).str.contains("AULA", na=False))
    ].copy()

    return df, {
        "Hoja": nombre_hoja,
        "Estatus": "Procesada",
        "Detalle": f"{len(df):,} aspirantes identificados."
    }


@st.cache_data(show_spinner=False)
def hist_procesar_archivo_excel(contenido_archivo):
    """
    Procesa todo el Excel de Historial de Aspirantes.
    """

    archivo = io.BytesIO(contenido_archivo)
    excel = pd.ExcelFile(archivo)

    bases = []
    bitacora = []

    for hoja in excel.sheet_names:

        df_hoja, resultado = hist_procesar_hoja(
            contenido_archivo,
            hoja
        )

        bitacora.append(resultado)

        if df_hoja is not None and not df_hoja.empty:
            bases.append(df_hoja)

    if not bases:
        return pd.DataFrame(), pd.DataFrame(bitacora)

    df_general = pd.concat(
        bases,
        ignore_index=True,
        sort=False
    )

    return df_general, pd.DataFrame(bitacora)

# ============================================================
# EVALUATEC
# ============================================================

EVAL_ETIQUETAS_AREAS = {
    "ING": "Inglés",
    "MAT": "Matemáticas",
    "COM": "Comprensión lectora",
    "RLM": "Razonamiento lógico-matemático",
    "PM": "Pensamiento matemático",
    "ARQ": "Arquitectura",
    "FIS": "Física",
    "ADMN": "Administración"
}

EVAL_ORDEN_AREAS = [
    "ING",
    "MAT",
    "COM",
    "RLM",
    "PM",
    "FIS",
    "ARQ",
    "ADMN"
]

EVAL_BLOQUES = {
    "ADM": "Administración",
    "ARQ": "Arquitectura",
    "ING": "Ingeniería"
}


def eval_leer_csv_archivo(archivo):
    """
    Lee archivos CSV de EVALUATEC con diferentes codificaciones y separadores.
    """

    contenido = archivo.getvalue()

    codificaciones = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252"
    ]

    separadores = [
        ",",
        ";",
        "\t"
    ]

    for codificacion in codificaciones:
        for separador in separadores:
            try:
                df = pd.read_csv(
                    io.BytesIO(contenido),
                    encoding=codificacion,
                    sep=separador
                )

                if len(df.columns) > 1:
                    return df

            except Exception:
                continue

    return pd.read_csv(
        io.BytesIO(contenido),
        encoding="latin-1"
    )


def eval_identificar_bloque_archivo(nombre_archivo):
    """
    Identifica si el CSV pertenece a Administración, Arquitectura o Ingeniería.
    """

    nombre = util_normalizar_texto(nombre_archivo)

    if "administracion" in nombre:
        return "ADM"

    if "arquitectura" in nombre:
        return "ARQ"

    if "ingenieria" in nombre:
        return "ING"

    return None


def eval_limpiar_nombre_carrera(valor):
    """
    Limpia el nombre de carrera.
    """

    if pd.isna(valor):
        return "Sin carrera especificada"

    return " ".join(str(valor).strip().split())


def eval_clasificar_inicio(valor):
    """
    Determina si el aspirante inició o no inició el examen.
    """

    if pd.isna(valor):
        return "No inició"

    texto = util_normalizar_texto(valor)

    valores_no_inicio = [
        "",
        "no",
        "n",
        "false",
        "falso",
        "0",
        "no inicio",
        "no iniciado",
        "pendiente",
        "null",
        "nan",
        "none"
    ]

    if texto in valores_no_inicio:
        return "No inició"

    if "no inicio" in texto:
        return "No inició"

    return "Inició"


def eval_convertir_porcentaje(valor):
    """
    Convierte valores de porcentaje a escala 0-100.
    """

    if pd.isna(valor):
        return np.nan

    texto = str(valor).strip()

    if texto == "":
        return np.nan

    texto = texto.replace("%", "")
    texto = texto.replace(",", ".")

    try:
        numero = float(texto)

    except ValueError:
        return np.nan

    if 0 <= numero <= 1:
        return numero * 100

    if 0 <= numero <= 100:
        return numero

    return np.nan


def eval_detectar_columnas_areas(df):
    """
    Detecta columnas de porcentaje correcto por sección.
    Ejemplo esperado:
    Sección MAT PorcentajeCorrectas
    """

    areas_detectadas = {}

    for columna in df.columns:

        texto = util_normalizar_texto(columna)

        texto_compacto = re.sub(
            r"[^a-z0-9]",
            "",
            texto
        )

        if "seccion" not in texto_compacto:
            continue

        if "porcentajecorrectas" not in texto_compacto:
            continue

        coincidencia = re.search(
            r"seccion([a-z0-9]+?)porcentajecorrectas",
            texto_compacto
        )

        if coincidencia:
            codigo = coincidencia.group(1).upper()
            areas_detectadas[codigo] = columna

    areas_ordenadas = {}

    for codigo in EVAL_ORDEN_AREAS:
        if codigo in areas_detectadas:
            areas_ordenadas[codigo] = areas_detectadas[codigo]

    for codigo, columna in areas_detectadas.items():
        if codigo not in areas_ordenadas:
            areas_ordenadas[codigo] = columna

    return areas_ordenadas


def eval_encontrar_columna_nombre(df):
    """
    Detecta columna de nombre completo en EVALUATEC.
    """

    posibles_columnas = [
        "Nombre completo",
        "NombreCompleto",
        "Nombre del aspirante",
        "Aspirante",
        "Alumno",
        "Estudiante",
        "Participante",
        "Nombre",
        "Nombre(s)"
    ]

    return util_encontrar_columna(
        df,
        posibles_columnas
    )


def eval_procesar_archivo(archivo):
    """
    Procesa un archivo CSV de EVALUATEC.
    """

    df = eval_leer_csv_archivo(archivo)

    bloque = eval_identificar_bloque_archivo(archivo.name)

    if bloque is None:
        raise ValueError(
            "No se identificó el bloque académico. "
            "El archivo debe contener Administración, Arquitectura o Ingeniería en el nombre."
        )

    columna_carrera = util_encontrar_columna(
        df,
        ["Carrera"]
    )

    columna_inicio = util_encontrar_columna(
        df,
        [
            "InicioExamen",
            "Inicio Examen",
            "Inició examen",
            "Inicio"
        ]
    )

    columna_nombre = eval_encontrar_columna_nombre(df)

    if columna_carrera is None:
        raise ValueError(
            f"{archivo.name}: no se encontró la columna Carrera."
        )

    if columna_inicio is None:
        raise ValueError(
            f"{archivo.name}: no se encontró la columna InicioExamen."
        )

    if columna_nombre is None:
        raise ValueError(
            f"{archivo.name}: no se encontró la columna de nombre del aspirante."
        )

    areas_detectadas = eval_detectar_columnas_areas(df)

    if not areas_detectadas:
        raise ValueError(
            f"{archivo.name}: no se detectaron columnas de áreas evaluadas."
        )

    df["Archivo EVALUATEC"] = archivo.name
    df["Bloque EVALUATEC"] = bloque
    df["Bloque EVALUATEC nombre"] = EVAL_BLOQUES.get(bloque, bloque)

    df["Nombre completo EVALUATEC"] = df[
        columna_nombre
    ].apply(nombre_visible)

    df["Nombre match"] = df[
        columna_nombre
    ].apply(normalizar_nombre)

    df["Carrera EVALUATEC"] = df[
        columna_carrera
    ].apply(eval_limpiar_nombre_carrera)

    df["Carrera match EVALUATEC"] = df[
        "Carrera EVALUATEC"
    ].apply(simplificar_carrera)

    df["Estatus inicio EVALUATEC"] = df[
        columna_inicio
    ].apply(eval_clasificar_inicio)

    for codigo, columna in areas_detectadas.items():

        etiqueta = EVAL_ETIQUETAS_AREAS.get(
            codigo,
            codigo
        )

        df[f"EVALUATEC {codigo}"] = df[
            columna
        ].apply(eval_convertir_porcentaje)

        df[f"EVALUATEC {etiqueta}"] = df[
            f"EVALUATEC {codigo}"
        ]

    columnas_areas = [
        f"EVALUATEC {codigo}"
        for codigo in areas_detectadas.keys()
    ]

    df["Promedio global EVALUATEC"] = df[
        columnas_areas
    ].mean(axis=1)

    df["Áreas detectadas EVALUATEC"] = ", ".join(
        [
            EVAL_ETIQUETAS_AREAS.get(codigo, codigo)
            for codigo in areas_detectadas.keys()
        ]
    )

    return df, areas_detectadas


def eval_procesar_archivos_multiples(archivos_eval):
    """
    Procesa los 3 CSV de EVALUATEC y devuelve una sola base.
    """

    bases = []
    errores = []
    bitacora = []

    for archivo in archivos_eval:

        try:
            df_archivo, areas_detectadas = eval_procesar_archivo(
                archivo
            )

            bases.append(df_archivo)

            bitacora.append(
                {
                    "Archivo": archivo.name,
                    "Estatus": "Procesado",
                    "Bloque": df_archivo["Bloque EVALUATEC nombre"].iloc[0],
                    "Registros": len(df_archivo),
                    "Áreas detectadas": ", ".join(
                        [
                            EVAL_ETIQUETAS_AREAS.get(codigo, codigo)
                            for codigo in areas_detectadas.keys()
                        ]
                    )
                }
            )

        except Exception as error:
            errores.append(
                f"{archivo.name}: {error}"
            )

            bitacora.append(
                {
                    "Archivo": archivo.name,
                    "Estatus": "No procesado",
                    "Bloque": "Sin identificar",
                    "Registros": 0,
                    "Áreas detectadas": str(error)
                }
            )

    if not bases:
        return pd.DataFrame(), errores, pd.DataFrame(bitacora)

    df_evaluatec = pd.concat(
        bases,
        ignore_index=True,
        sort=False
    )

    return df_evaluatec, errores, pd.DataFrame(bitacora)

# ============================================================
# CHASIDE
# ============================================================

CHASIDE_AREAS = [
    "C",
    "H",
    "A",
    "S",
    "I",
    "D",
    "E"
]

CHASIDE_AREAS_LONG = {
    "C": "Administrativo",
    "H": "Humanidades y Sociales",
    "A": "Artístico",
    "S": "Ciencias de la Salud",
    "I": "Enseñanzas Técnicas",
    "D": "Defensa y Seguridad",
    "E": "Ciencias Experimentales"
}

CHASIDE_INTERESES_ITEMS = {
    "C": [1, 12, 20, 53, 64, 71, 78, 85, 91, 98],
    "H": [9, 25, 34, 41, 56, 67, 74, 80, 89, 95],
    "A": [3, 11, 21, 28, 36, 45, 50, 57, 81, 96],
    "S": [8, 16, 23, 33, 44, 52, 62, 70, 87, 92],
    "I": [6, 19, 27, 38, 47, 54, 60, 75, 83, 97],
    "D": [5, 14, 24, 31, 37, 48, 58, 65, 73, 84],
    "E": [17, 32, 35, 42, 49, 61, 68, 77, 88, 93]
}

CHASIDE_APTITUDES_ITEMS = {
    "C": [2, 15, 46, 51],
    "H": [30, 63, 72, 86],
    "A": [22, 39, 76, 82],
    "S": [4, 29, 40, 69],
    "I": [10, 26, 59, 90],
    "D": [13, 18, 43, 66],
    "E": [7, 55, 79, 94]
}

CHASIDE_PERFILES_CARRERA = {
    "Arquitectura": ["A", "I", "C"],
    "Contador Público": ["C", "D"],
    "Licenciatura en Administración": ["C", "D"],
    "Ingeniería Ambiental": ["I", "C", "E"],
    "Ingeniería Bioquímica": ["I", "C", "E"],
    "Ingeniería en Gestión Empresarial": ["C", "D", "H"],
    "Ingeniería Industrial": ["C", "D", "H"],
    "Ingeniería en Inteligencia Artificial": ["I", "E"],
    "Ingeniería Mecatrónica": ["I", "E"],
    "Ingeniería en Sistemas Computacionales": ["I", "E"]
}

CHASIDE_COLUMNA_NOMBRE = "Ingrese su nombre completo"
CHASIDE_COLUMNA_CARRERA = "¿A qué carrera desea ingresar?"
CHASIDE_COLUMNA_EMAIL_1 = "Dirección de correo electrónico"
CHASIDE_COLUMNA_EMAIL_2 = "Escriba su correo electrónico"


def chaside_transformar_url_google_sheets(url):
    """
    Convierte una URL editable de Google Sheets a CSV.
    """

    url = str(url).strip()

    if "export?format=csv" in url:
        return url

    if "docs.google.com/spreadsheets" not in url:
        return url

    try:
        file_id = url.split("/d/")[1].split("/")[0]

        gid = "1491376423"

        if "gid=" in url:
            gid = url.split("gid=")[-1].split("&")[0].split("#")[0]

        resourcekey = ""

        if "resourcekey=" in url:
            resourcekey = (
                url
                .split("resourcekey=")[-1]
                .split("&")[0]
                .split("#")[0]
            )

        url_csv = (
            f"https://docs.google.com/spreadsheets/d/"
            f"{file_id}/export?format=csv&gid={gid}"
        )

        if resourcekey != "":
            url_csv = f"{url_csv}&resourcekey={resourcekey}"

        return url_csv

    except Exception:
        raise ValueError(
            "No se pudo transformar el enlace de Google Sheets. "
            "Pega el enlace completo de la hoja de respuestas."
        )


@st.cache_data(show_spinner=False)
def chaside_cargar_respuestas(url):
    """
    Carga respuestas CHASIDE desde Google Sheets.
    """

    url_csv = chaside_transformar_url_google_sheets(url)

    try:
        return pd.read_csv(url_csv)

    except Exception as error:
        raise ValueError(
            "No fue posible leer la hoja de respuestas CHASIDE. "
            "Verifica que esté compartida como 'Cualquier persona con el enlace puede ver'. "
            f"Detalle técnico: {error}"
        )


def chaside_col_item(columnas_items, numero_item):
    """
    Devuelve la columna correspondiente al reactivo CHASIDE.
    """

    return columnas_items[numero_item - 1]


def chaside_procesar_respuestas(
    df_raw,
    peso_intereses=0.8,
    peso_aptitudes=0.2
):
    """
    Procesa CHASIDE y deja una base reducida para cruce.
    """

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    faltantes = [
        columna
        for columna in [
            CHASIDE_COLUMNA_NOMBRE,
            CHASIDE_COLUMNA_CARRERA
        ]
        if columna not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en CHASIDE: {faltantes}. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    columnas_items = df.columns[6:104]

    if len(columnas_items) != 98:
        raise ValueError(
            f"Se esperaban 98 reactivos CHASIDE, pero se detectaron {len(columnas_items)}."
        )

    df_items = (
        df[columnas_items]
        .astype(str)
        .apply(lambda col: col.str.strip().str.lower())
        .replace(
            {
                "sí": 1,
                "si": 1,
                "s": 1,
                "1": 1,
                "true": 1,
                "verdadero": 1,
                "x": 1,
                "no": 0,
                "n": 0,
                "0": 0,
                "false": 0,
                "falso": 0,
                "": 0,
                "nan": 0
            }
        )
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df_base = df.drop(
        columns=columnas_items,
        errors="ignore"
    ).copy()

    nuevas_columnas = pd.DataFrame(
        index=df_base.index
    )

    nuevas_columnas["Nombre completo CHASIDE"] = df_base[
        CHASIDE_COLUMNA_NOMBRE
    ].apply(nombre_visible)

    nuevas_columnas["Nombre match"] = df_base[
        CHASIDE_COLUMNA_NOMBRE
    ].apply(normalizar_nombre)

    nuevas_columnas["Carrera elegida CHASIDE"] = df_base[
        CHASIDE_COLUMNA_CARRERA
    ].astype(str).str.strip()

    nuevas_columnas["Carrera match CHASIDE"] = nuevas_columnas[
        "Carrera elegida CHASIDE"
    ].apply(simplificar_carrera)

    if CHASIDE_COLUMNA_EMAIL_1 in df_base.columns:
        nuevas_columnas["Correo Google CHASIDE"] = df_base[
            CHASIDE_COLUMNA_EMAIL_1
        ].apply(normalizar_correo)
    else:
        nuevas_columnas["Correo Google CHASIDE"] = ""

    if CHASIDE_COLUMNA_EMAIL_2 in df_base.columns:
        nuevas_columnas["Correo escrito CHASIDE"] = df_base[
            CHASIDE_COLUMNA_EMAIL_2
        ].apply(normalizar_correo)
    else:
        nuevas_columnas["Correo escrito CHASIDE"] = ""

    nuevas_columnas["Desviación respuestas CHASIDE"] = df_items.std(axis=1)

    umbral_respuesta_plana = nuevas_columnas[
        "Desviación respuestas CHASIDE"
    ].quantile(0.10)

    nuevas_columnas["Respuesta plana CHASIDE"] = (
        nuevas_columnas["Desviación respuestas CHASIDE"]
        <= umbral_respuesta_plana
    )

    for area in CHASIDE_AREAS:

        nuevas_columnas[f"CHASIDE Interés {area}"] = df_items[
            [
                chaside_col_item(columnas_items, item)
                for item in CHASIDE_INTERESES_ITEMS[area]
            ]
        ].sum(axis=1)

        nuevas_columnas[f"CHASIDE Aptitud {area}"] = df_items[
            [
                chaside_col_item(columnas_items, item)
                for item in CHASIDE_APTITUDES_ITEMS[area]
            ]
        ].sum(axis=1)

        nuevas_columnas[f"CHASIDE Puntaje {area}"] = (
            nuevas_columnas[f"CHASIDE Interés {area}"] * peso_intereses
            +
            nuevas_columnas[f"CHASIDE Aptitud {area}"] * peso_aptitudes
        )

    columnas_puntaje = [
        f"CHASIDE Puntaje {area}"
        for area in CHASIDE_AREAS
    ]

    def obtener_areas_ordenadas(fila):
        valores = []

        for area in CHASIDE_AREAS:
            valores.append(
                {
                    "Área": area,
                    "Descripción": CHASIDE_AREAS_LONG.get(area, area),
                    "Puntaje": fila[f"CHASIDE Puntaje {area}"]
                }
            )

        tabla = pd.DataFrame(valores).sort_values(
            "Puntaje",
            ascending=False
        )

        return tabla

    areas_fuertes_1 = []
    areas_fuertes_2 = []
    areas_debiles_1 = []
    areas_debiles_2 = []

    for _, fila in nuevas_columnas.iterrows():

        tabla_areas = obtener_areas_ordenadas(fila)

        fuertes = tabla_areas.head(2).reset_index(drop=True)
        debiles = tabla_areas.tail(2).sort_values(
            "Puntaje",
            ascending=True
        ).reset_index(drop=True)

        areas_fuertes_1.append(
            f"{fuertes.loc[0, 'Área']} · {fuertes.loc[0, 'Descripción']}"
        )

        areas_fuertes_2.append(
            f"{fuertes.loc[1, 'Área']} · {fuertes.loc[1, 'Descripción']}"
        )

        areas_debiles_1.append(
            f"{debiles.loc[0, 'Área']} · {debiles.loc[0, 'Descripción']}"
        )

        areas_debiles_2.append(
            f"{debiles.loc[1, 'Área']} · {debiles.loc[1, 'Descripción']}"
        )

    nuevas_columnas["Área fuerte CHASIDE 1"] = areas_fuertes_1
    nuevas_columnas["Área fuerte CHASIDE 2"] = areas_fuertes_2
    nuevas_columnas["Área débil CHASIDE 1"] = areas_debiles_1
    nuevas_columnas["Área débil CHASIDE 2"] = areas_debiles_2

    nuevas_columnas["Área fuerte principal CHASIDE"] = nuevas_columnas[
        columnas_puntaje
    ].idxmax(axis=1).str.replace(
        "CHASIDE Puntaje ",
        "",
        regex=False
    )

    nuevas_columnas["Score CHASIDE"] = nuevas_columnas[
        columnas_puntaje
    ].max(axis=1)

    def carrera_mejor_ubicada(fila):
        if fila["Respuesta plana CHASIDE"]:
            return "Requiere realizar o repetir la escala CHASIDE"

        area_fuerte = fila["Área fuerte principal CHASIDE"]
        carrera_actual = str(fila["Carrera elegida CHASIDE"]).strip()

        sugeridas = [
            carrera
            for carrera, areas in CHASIDE_PERFILES_CARRERA.items()
            if area_fuerte in areas
        ]

        if carrera_actual in sugeridas:
            return carrera_actual

        if sugeridas:
            return ", ".join(sugeridas)

        return "Sin sugerencia clara"

    nuevas_columnas["Diagnóstico CHASIDE"] = nuevas_columnas.apply(
        carrera_mejor_ubicada,
        axis=1
    )

    return nuevas_columnas.copy()

# ============================================================
# CRUCE MAESTRO
# ============================================================

def preparar_evaluatec_desde_bloques(datos_eval_global):
    """
    Integra los 3 bloques EVALUATEC en una sola base.
    """

    bases_eval = []

    for bloque, info_bloque in datos_eval_global.items():

        df_temp = info_bloque["df"].copy()

        columna_nombre = eval_encontrar_columna_nombre(df_temp)

        if columna_nombre is None:
            continue

        df_temp["Nombre completo EVALUATEC"] = df_temp[
            columna_nombre
        ].apply(nombre_visible)

        df_temp["Nombre match"] = df_temp[
            columna_nombre
        ].apply(normalizar_nombre)

        df_temp["Carrera match EVALUATEC"] = df_temp[
            "Carrera EVALUATEC"
        ].apply(simplificar_carrera)

        areas = info_bloque.get("areas", {})

        if isinstance(areas, dict):
            areas_detectadas = list(areas.keys())
        else:
            areas_detectadas = [str(area) for area in areas]

        df_temp["Áreas detectadas EVALUATEC"] = ", ".join(
            areas_detectadas
        )

        bases_eval.append(df_temp)

    if not bases_eval:
        return pd.DataFrame()

    df_evaluatec = pd.concat(
        bases_eval,
        ignore_index=True,
        sort=False
    )

    return df_evaluatec
    
def preparar_historial_para_cruce(df_historial):
    """
    Prepara historial para cruce maestro.
    """

    df = df_historial.copy()

    col_apellido_paterno, col_apellido_materno, col_nombre = hist_encontrar_nombre_historial(
        df
    )

    if col_apellido_paterno is None or col_nombre is None:
        raise ValueError(
            "No se identificaron columnas de apellido paterno y nombre en Historial."
        )

    if col_apellido_materno is None:
        df["Nombre completo Historial"] = (
            df[col_apellido_paterno].fillna("").astype(str)
            + " "
            + df[col_nombre].fillna("").astype(str)
        )
    else:
        df["Nombre completo Historial"] = (
            df[col_apellido_paterno].fillna("").astype(str)
            + " "
            + df[col_apellido_materno].fillna("").astype(str)
            + " "
            + df[col_nombre].fillna("").astype(str)
        )

    df["Nombre completo Historial"] = df[
        "Nombre completo Historial"
    ].apply(nombre_visible)

    df["Nombre match"] = df[
        "Nombre completo Historial"
    ].apply(normalizar_nombre)

    df = df[
        df["Nombre match"].notna()
        &
        (df["Nombre match"].astype(str).str.strip() != "")
        &
        (~df["Nombre match"].astype(str).str.contains("AULA", na=False))
    ].copy()

    df["Carrera match Historial"] = df[
        "Carrera historial"
    ].apply(simplificar_carrera)

    if "Matrícula/ID" not in df.columns:
        df["Matrícula/ID"] = "Sin dato"

    if "Sexo" not in df.columns:
        df["Sexo"] = "Sin especificar"

    if "Escuela de procedencia original" not in df.columns:
        df["Escuela de procedencia original"] = "Sin dato"

    if "Escuela de procedencia normalizada" not in df.columns:
        df["Escuela de procedencia normalizada"] = "Sin dato"

    if "Estado de procedencia" not in df.columns:
        df["Estado de procedencia"] = "Sin dato"

    return df



def crear_base_cruzada_maestra(df_historial, df_evaluatec):
    """
    Cruza Historial y EVALUATEC por nombre normalizado.
    """

    hist = df_historial.copy()
    eval_df = df_evaluatec.copy()

    hist = hist.add_prefix("hist_")
    eval_df = eval_df.add_prefix("eval_")

    df_cruzado = hist.merge(
        eval_df,
        left_on="hist_Nombre match",
        right_on="eval_Nombre match",
        how="outer",
        indicator=True
    )

    df_cruzado["Nombre"] = df_cruzado[
        "hist_Nombre completo Historial"
    ].combine_first(
        df_cruzado["eval_Nombre completo EVALUATEC"]
    )

    df_cruzado["Carrera Historial"] = df_cruzado[
        "hist_Carrera historial"
    ]

    df_cruzado["Carrera EVALUATEC"] = df_cruzado[
        "eval_Carrera EVALUATEC"
    ]

    df_cruzado["Carrera"] = df_cruzado[
        "Carrera Historial"
    ].combine_first(
        df_cruzado["Carrera EVALUATEC"]
    )

    df_cruzado["Carrera match"] = df_cruzado[
        "hist_Carrera match historial"
    ].combine_first(
        df_cruzado["eval_Carrera match EVALUATEC"]
    )

    df_cruzado["Estatus cruce"] = np.select(
        [
            df_cruzado["_merge"] == "both",
            df_cruzado["_merge"] == "left_only",
            df_cruzado["_merge"] == "right_only"
        ],
        [
            "Coincide en Historial y EVALUATEC",
            "Solo en Historial",
            "Solo en EVALUATEC"
        ],
        default="Sin clasificar"
    )

    df_cruzado["Carrera coincide Historial/EVALUATEC"] = (
        df_cruzado["hist_Carrera match historial"]
        ==
        df_cruzado["eval_Carrera match EVALUATEC"]
    )

    df_cruzado.loc[
        (
            df_cruzado["_merge"] == "both"
        )
        &
        (
            df_cruzado["Carrera coincide Historial/EVALUATEC"] == False
        ),
        "Estatus cruce"
    ] = "Coincide por nombre, carrera distinta"

    return df_cruzado


def extraer_correos_de_fila(fila):
    """
    Extrae correos posibles de una fila cruzada.
    """

    correos = []

    for columna in fila.index:
        columna_limpia = util_limpiar_texto(columna)

        if (
            "correo" in columna_limpia
            or "email" in columna_limpia
            or "mail" in columna_limpia
        ):
            correo = normalizar_correo(fila[columna])

            if correo != "":
                correos.append(correo)

    return list(set(correos))

def extraer_correos_historial_fila(fila):
    """
    Extrae correos únicamente desde columnas de Historial.
    Después del merge, las columnas vienen con prefijo hist_.
    """

    correos = []

    for columna in fila.index:
        columna_limpia = util_limpiar_texto(columna)

        if not columna_limpia.startswith("hist_"):
            continue

        if (
            "correo" in columna_limpia
            or "email" in columna_limpia
            or "mail" in columna_limpia
        ):
            correo = normalizar_correo(fila[columna])

            if correo != "":
                correos.append(correo)

    return list(set(correos))
    
def buscar_chaside_para_estudiante(fila, df_chaside):
    """
    Busca CHASIDE únicamente por email del Historial.
    Mucho más rápido que comparar por nombre y carrera.
    """

    resultado_base = {
        "Diagnóstico CHASIDE": "Sin respuesta CHASIDE",
        "Carrera elegida CHASIDE": "Sin dato",
        "Área fuerte CHASIDE 1": "Sin dato",
        "Área fuerte CHASIDE 2": "Sin dato",
        "Área débil CHASIDE 1": "Sin dato",
        "Área débil CHASIDE 2": "Sin dato",
        "Score CHASIDE": np.nan,
        "Estatus cruce CHASIDE": "No encontrado por email en Historial"
    }

    if df_chaside is None or df_chaside.empty:
        return resultado_base

    correos_historial = extraer_correos_historial_fila(fila)

    if not correos_historial:
        return resultado_base

    df_match = df_chaside[
        (
            df_chaside["Correo Google CHASIDE"].fillna("").isin(correos_historial)
        )
        |
        (
            df_chaside["Correo escrito CHASIDE"].fillna("").isin(correos_historial)
        )
    ].copy()

    if df_match.empty:
        return resultado_base

    mejor = df_match.iloc[0]

    return {
        "Diagnóstico CHASIDE": mejor["Diagnóstico CHASIDE"],
        "Carrera elegida CHASIDE": mejor["Carrera elegida CHASIDE"],
        "Área fuerte CHASIDE 1": mejor["Área fuerte CHASIDE 1"],
        "Área fuerte CHASIDE 2": mejor["Área fuerte CHASIDE 2"],
        "Área débil CHASIDE 1": mejor["Área débil CHASIDE 1"],
        "Área débil CHASIDE 2": mejor["Área débil CHASIDE 2"],
        "Score CHASIDE": mejor["Score CHASIDE"],
        "Estatus cruce CHASIDE": "Coincide por email en Historial"
    }
    
def obtener_dos_areas_evaluatec(fila, tipo="fuerte"):
    """
    Obtiene dos áreas fuertes o débiles de EVALUATEC.
    """

    registros = []

    for codigo in EVAL_ORDEN_AREAS:
        columna = f"eval_EVALUATEC {codigo}"

        if columna not in fila.index:
            continue

        valor = fila[columna]

        if pd.isna(valor):
            continue

        registros.append(
            {
                "Área": EVAL_ETIQUETAS_AREAS.get(codigo, codigo),
                "Resultado": float(valor)
            }
        )

    if not registros:
        return "Sin dato", "Sin dato"

    tabla = pd.DataFrame(registros)

    if tipo == "fuerte":
        tabla = tabla.sort_values(
            "Resultado",
            ascending=False
        )
    else:
        tabla = tabla.sort_values(
            "Resultado",
            ascending=True
        )

    area_1 = tabla.iloc[0]
    area_2 = tabla.iloc[1] if len(tabla) > 1 else tabla.iloc[0]

    texto_1 = f"{area_1['Área']} ({area_1['Resultado']:.1f}%)"
    texto_2 = f"{area_2['Área']} ({area_2['Resultado']:.1f}%)"

    return texto_1, texto_2
    


def generar_concentrado_maestro(
    df_historial_preparado,
    df_evaluatec_preparado,
    df_chaside_procesado
):
    """
    Genera una super base por estudiante sin boxplot.
    """

    df_cruzado = crear_base_cruzada_maestra(
        df_historial=df_historial_preparado,
        df_evaluatec=df_evaluatec_preparado
    )

    columnas_correo_historial = [
        columna
        for columna in df_cruzado.columns
        if columna.startswith("hist_")
        and (
            "correo" in util_limpiar_texto(columna)
            or "email" in util_limpiar_texto(columna)
            or "mail" in util_limpiar_texto(columna)
        )
    ]

    st.write(
        "Columnas de correo detectadas en Historial:",
        columnas_correo_historial
    )

    registros = []

    for _, fila in df_cruzado.iterrows():

        area_fuerte_eval_1, area_fuerte_eval_2 = obtener_dos_areas_evaluatec(
            fila,
            tipo="fuerte"
        )

        area_debil_eval_1, area_debil_eval_2 = obtener_dos_areas_evaluatec(
            fila,
            tipo="debil"
        )

        resultado_chaside = buscar_chaside_para_estudiante(
            fila,
            df_chaside_procesado
        )

        registro = {
            "Nombre": valor_seguro(fila, "Nombre"),
            "Matrícula/ID": valor_seguro(fila, "hist_Matrícula/ID"),
            "Carrera": valor_seguro(fila, "Carrera"),
            "Carrera Historial": valor_seguro(fila, "Carrera Historial"),
            "Carrera EVALUATEC": valor_seguro(fila, "Carrera EVALUATEC"),
            "Estatus cruce": valor_seguro(fila, "Estatus cruce"),
            "Sexo": valor_seguro(fila, "hist_Sexo"),
            "Escuela de procedencia": valor_seguro(
                fila,
                "hist_Escuela de procedencia original"
            ),
            "Escuela de procedencia normalizada": valor_seguro(
                fila,
                "hist_Escuela de procedencia normalizada"
            ),
            "Estado de procedencia": valor_seguro(
                fila,
                "hist_Estado de procedencia"
            ),
            "Promedio bachillerato": valor_seguro(
                fila,
                "hist_Promedio bachillerato 100",
                np.nan
            ),
            "Estatus promedio bachillerato": valor_seguro(
                fila,
                "hist_Estatus promedio bachillerato"
            ),
            "Estatus inicio EVALUATEC": valor_seguro(
                fila,
                "eval_Estatus inicio EVALUATEC"
            ),
            "Resultado global EVALUATEC": valor_seguro(
                fila,
                "eval_Promedio global EVALUATEC",
                np.nan
            ),
            "Área fuerte EVALUATEC 1": area_fuerte_eval_1,
            "Área fuerte EVALUATEC 2": area_fuerte_eval_2,
            "Área débil EVALUATEC 1": area_debil_eval_1,
            "Área débil EVALUATEC 2": area_debil_eval_2,
            "Diagnóstico CHASIDE": resultado_chaside["Diagnóstico CHASIDE"],
            "Carrera elegida CHASIDE": resultado_chaside["Carrera elegida CHASIDE"],
            "Área fuerte CHASIDE 1": resultado_chaside["Área fuerte CHASIDE 1"],
            "Área fuerte CHASIDE 2": resultado_chaside["Área fuerte CHASIDE 2"],
            "Área débil CHASIDE 1": resultado_chaside["Área débil CHASIDE 1"],
            "Área débil CHASIDE 2": resultado_chaside["Área débil CHASIDE 2"],
            "Score CHASIDE": resultado_chaside["Score CHASIDE"],
            "Estatus cruce CHASIDE": resultado_chaside["Estatus cruce CHASIDE"]
        }

        for codigo in EVAL_ORDEN_AREAS:
            columna = f"eval_EVALUATEC {codigo}"

            if columna in fila.index:
                registro[
                    f"EVALUATEC {EVAL_ETIQUETAS_AREAS.get(codigo, codigo)}"
                ] = fila[columna]

        registros.append(registro)

    df_maestro = pd.DataFrame(registros)

    if df_maestro.empty:
        return df_maestro

    df_maestro = df_maestro.sort_values(
        [
            "Carrera",
            "Nombre"
        ],
        ascending=[
            True,
            True
        ]
    ).reset_index(drop=True)

    return df_maestro



# ============================================================
# CLUSTERING ACADÉMICO POR CARRERA
# ============================================================

CLUSTER_PERFILES_POR_CANTIDAD = {
    2: [
        "Atención académica prioritaria",
        "Desempeño consolidado"
    ],
    3: [
        "Atención académica prioritaria",
        "Seguimiento académico",
        "Desempeño consolidado"
    ],
    4: [
        "Atención académica prioritaria",
        "Seguimiento académico",
        "Potencial con reforzamiento",
        "Desempeño consolidado"
    ],
    5: [
        "Atención académica prioritaria",
        "Seguimiento académico intensivo",
        "Seguimiento académico",
        "Potencial con reforzamiento",
        "Desempeño consolidado"
    ],
    6: [
        "Atención académica prioritaria",
        "Seguimiento académico intensivo",
        "Seguimiento académico",
        "Potencial con reforzamiento",
        "Buen desempeño",
        "Desempeño consolidado"
    ]
}


def convertir_chaside_binario(valor):
    """Convierte el diagnóstico CHASIDE a 1=coincide, 0=no coincide."""
    texto = util_limpiar_texto(valor)

    if texto == "" or "sin respuesta" in texto or "sin dato" in texto:
        return np.nan

    expresiones_no = [
        "no coincide",
        "no compatible",
        "sin coincidencia",
        "baja coincidencia",
        "desajuste"
    ]

    if any(expresion in texto for expresion in expresiones_no):
        return 0.0

    expresiones_si = [
        "coincide",
        "compatible",
        "alta coincidencia",
        "afinidad"
    ]

    if any(expresion in texto for expresion in expresiones_si):
        return 1.0

    return np.nan


def obtener_columnas_cluster(df):
    """Selecciona las variables numéricas disponibles para segmentar."""
    candidatas = [
        "Promedio bachillerato",
        "Resultado global EVALUATEC",
        "Score CHASIDE",
        "Coincidencia CHASIDE"
    ]

    candidatas.extend([
        columna for columna in df.columns
        if columna.startswith("EVALUATEC ")
    ])

    columnas = []

    for columna in candidatas:
        if columna not in df.columns:
            continue

        serie = pd.to_numeric(df[columna], errors="coerce")

        # Evitar variables casi vacías o constantes.
        if serie.notna().sum() < max(3, int(len(df) * 0.35)):
            continue

        if serie.nunique(dropna=True) <= 1:
            continue

        columnas.append(columna)

    return list(dict.fromkeys(columnas))


def seleccionar_k_automatico(X, random_state=42):
    """
    Evalúa k con silueta y Davies-Bouldin.
    La silueta es el criterio principal y Davies-Bouldin desempata.
    """
    n = len(X)
    k_max = min(6, n - 1)

    if n < 6 or k_max < 2:
        return None, pd.DataFrame()

    resultados = []

    for k in range(2, k_max + 1):
        modelo = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20
        )
        etiquetas = modelo.fit_predict(X)
        conteos = pd.Series(etiquetas).value_counts()

        if len(conteos) < 2:
            continue

        silueta = silhouette_score(X, etiquetas)
        davies = davies_bouldin_score(X, etiquetas)
        minimo_cluster = int(conteos.min())

        # Penaliza soluciones con clusters demasiado pequeños.
        penalizacion = 0.08 if minimo_cluster < 2 else 0.0
        puntuacion = silueta - penalizacion

        resultados.append({
            "k": k,
            "Silueta": float(silueta),
            "Davies-Bouldin": float(davies),
            "Inercia": float(modelo.inertia_),
            "Mínimo estudiantes por cluster": minimo_cluster,
            "Puntuación selección": float(puntuacion)
        })

    df_metricas = pd.DataFrame(resultados)

    if df_metricas.empty:
        return None, df_metricas

    mejor = df_metricas.sort_values(
        ["Puntuación selección", "Davies-Bouldin"],
        ascending=[False, True]
    ).iloc[0]

    return int(mejor["k"]), df_metricas


def asignar_nombres_perfiles(df_carrera, columnas_modelo):
    """Ordena los clusters de menor a mayor desempeño y les asigna nombres."""
    df = df_carrera.copy()

    columnas_desempeno = [
        columna for columna in columnas_modelo
        if columna != "Coincidencia CHASIDE"
    ]

    if not columnas_desempeno:
        columnas_desempeno = columnas_modelo.copy()

    resumen = (
        df.groupby("Cluster", dropna=False)[columnas_desempeno]
        .mean(numeric_only=True)
    )

    if "Coincidencia CHASIDE" in df.columns:
        resumen["Coincidencia CHASIDE"] = (
            df.groupby("Cluster")["Coincidencia CHASIDE"].mean()
        )

    # Estandariza centroides para que ninguna escala domine el orden.
    centroides = resumen.copy()
    for columna in centroides.columns:
        desviacion = centroides[columna].std(ddof=0)
        if pd.isna(desviacion) or desviacion == 0:
            centroides[columna] = 0.0
        else:
            centroides[columna] = (
                centroides[columna] - centroides[columna].mean()
            ) / desviacion

    centroides["Índice desempeño cluster"] = centroides.mean(axis=1)
    orden = centroides.sort_values("Índice desempeño cluster").index.tolist()
    k = len(orden)
    nombres = CLUSTER_PERFILES_POR_CANTIDAD.get(k)

    if nombres is None:
        nombres = [f"Perfil académico {i + 1}" for i in range(k)]

    mapa_nombre = {
        cluster: nombres[posicion]
        for posicion, cluster in enumerate(orden)
    }
    mapa_prioridad = {
        cluster: posicion + 1
        for posicion, cluster in enumerate(orden)
    }

    df["Perfil académico"] = df["Cluster"].map(mapa_nombre)
    df["Orden de prioridad"] = df["Cluster"].map(mapa_prioridad)
    df["Nivel de atención"] = np.select(
        [
            df["Orden de prioridad"] == 1,
            df["Orden de prioridad"] == 2
        ],
        [
            "Prioridad 1",
            "Prioridad 2"
        ],
        default="Seguimiento regular"
    )
    df["Atender prioritariamente"] = np.where(
        df["Orden de prioridad"].isin([1, 2]),
        "Sí",
        "No"
    )

    indice_mapa = centroides["Índice desempeño cluster"].to_dict()
    df["Índice desempeño del perfil"] = df["Cluster"].map(indice_mapa)

    return df


def aplicar_clustering_por_carrera(df_maestro, random_state=42):
    """
    Segmenta estudiantes de cada carrera por separado.

    Devuelve:
    - base individual con cluster y perfil;
    - resumen de perfiles;
    - métricas de selección de k.
    """
    df = df_maestro.copy()
    df["Coincidencia CHASIDE"] = df["Diagnóstico CHASIDE"].apply(
        convertir_chaside_binario
    )

    columnas_nuevas = {
        "Cluster": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "Perfil académico": "Sin segmentación",
        "Orden de prioridad": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "Nivel de atención": "Sin segmentación",
        "Atender prioritariamente": "No",
        "K seleccionado": pd.Series(pd.NA, index=df.index, dtype="Int64"),
        "Silueta del modelo": np.nan,
        "Davies-Bouldin del modelo": np.nan,
        "Distancia al centroide": np.nan,
        "Índice desempeño del perfil": np.nan,
        "Variables usadas en clustering": ""
    }

    for columna, valor in columnas_nuevas.items():
        df[columna] = valor

    resumenes = []
    metricas_todas = []

    for carrera, indices in df.groupby("Carrera", dropna=False).groups.items():
        indices = list(indices)
        grupo = df.loc[indices].copy()
        columnas_modelo = obtener_columnas_cluster(grupo)

        if len(grupo) < 6 or len(columnas_modelo) < 2:
            df.loc[indices, "Perfil académico"] = "Datos insuficientes para segmentar"
            df.loc[indices, "Nivel de atención"] = "Revisión individual"
            continue

        X_original = grupo[columnas_modelo].apply(
            pd.to_numeric,
            errors="coerce"
        )

        imputador = SimpleImputer(strategy="median")
        escalador = StandardScaler()
        X_imputado = imputador.fit_transform(X_original)
        X = escalador.fit_transform(X_imputado)

        k, metricas = seleccionar_k_automatico(
            X,
            random_state=random_state
        )

        if k is None:
            df.loc[indices, "Perfil académico"] = "Datos insuficientes para segmentar"
            df.loc[indices, "Nivel de atención"] = "Revisión individual"
            continue

        modelo = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20
        )
        etiquetas = modelo.fit_predict(X)
        distancias = np.linalg.norm(X - modelo.cluster_centers_[etiquetas], axis=1)

        grupo["Cluster"] = etiquetas
        grupo["Distancia al centroide"] = distancias
        grupo = asignar_nombres_perfiles(grupo, columnas_modelo)

        mejor_metrica = metricas.loc[metricas["k"] == k].iloc[0]

        df.loc[indices, "Cluster"] = grupo["Cluster"].astype("Int64").values
        df.loc[indices, "Perfil académico"] = grupo["Perfil académico"].values
        df.loc[indices, "Orden de prioridad"] = grupo["Orden de prioridad"].astype("Int64").values
        df.loc[indices, "Nivel de atención"] = grupo["Nivel de atención"].values
        df.loc[indices, "Atender prioritariamente"] = grupo["Atender prioritariamente"].values
        df.loc[indices, "K seleccionado"] = k
        df.loc[indices, "Silueta del modelo"] = mejor_metrica["Silueta"]
        df.loc[indices, "Davies-Bouldin del modelo"] = mejor_metrica["Davies-Bouldin"]
        df.loc[indices, "Distancia al centroide"] = grupo["Distancia al centroide"].values
        df.loc[indices, "Índice desempeño del perfil"] = grupo["Índice desempeño del perfil"].values
        df.loc[indices, "Variables usadas en clustering"] = ", ".join(columnas_modelo)

        metricas["Carrera"] = carrera
        metricas["K seleccionado"] = k
        metricas_todas.append(metricas)

        for perfil, subgrupo in grupo.groupby("Perfil académico"):
            resumen = {
                "Carrera": carrera,
                "Perfil académico": perfil,
                "Orden de prioridad": int(subgrupo["Orden de prioridad"].iloc[0]),
                "Nivel de atención": subgrupo["Nivel de atención"].iloc[0],
                "Estudiantes": len(subgrupo),
                "Porcentaje de la carrera": round(100 * len(subgrupo) / len(grupo), 1),
                "Promedio bachillerato": pd.to_numeric(
                    subgrupo["Promedio bachillerato"], errors="coerce"
                ).mean(),
                "Promedio EVALUATEC": pd.to_numeric(
                    subgrupo["Resultado global EVALUATEC"], errors="coerce"
                ).mean(),
                "Coincidencia CHASIDE (%)": 100 * pd.to_numeric(
                    subgrupo["Coincidencia CHASIDE"], errors="coerce"
                ).mean(),
                "K seleccionado": k,
                "Silueta": mejor_metrica["Silueta"],
                "Davies-Bouldin": mejor_metrica["Davies-Bouldin"]
            }
            resumenes.append(resumen)

    df["Canalización determinada por coordinación"] = ""
    df["Observaciones de coordinación"] = ""

    df_resumen = pd.DataFrame(resumenes)
    if not df_resumen.empty:
        df_resumen = df_resumen.sort_values(
            ["Carrera", "Orden de prioridad"]
        ).reset_index(drop=True)

        columnas_redondeo = [
            "Promedio bachillerato",
            "Promedio EVALUATEC",
            "Coincidencia CHASIDE (%)",
            "Silueta",
            "Davies-Bouldin"
        ]
        for columna in columnas_redondeo:
            if columna in df_resumen.columns:
                df_resumen[columna] = df_resumen[columna].round(3)

    df_metricas = (
        pd.concat(metricas_todas, ignore_index=True)
        if metricas_todas
        else pd.DataFrame()
    )

    return df, df_resumen, df_metricas


def render_vista_clusters(df_maestro):
    """Vista Streamlit con donut seleccionable y lista nominal."""
    st.markdown("## Segmentación académica por carrera")
    st.caption(
        "Selecciona una porción del gráfico para consultar la lista nominal. "
        "Los dos perfiles con menor desempeño quedan marcados como prioritarios; "
        "la canalización final corresponde a coordinación."
    )

    carreras = sorted(
        df_maestro.loc[
            df_maestro["Perfil académico"] != "Datos insuficientes para segmentar",
            "Carrera"
        ].dropna().astype(str).unique().tolist()
    )

    if not carreras:
        st.info("No hay carreras con información suficiente para generar clusters.")
        return

    carrera = st.selectbox(
        "Carrera",
        carreras,
        key="selector_carrera_cluster"
    )

    df_carrera = df_maestro[df_maestro["Carrera"].astype(str) == carrera].copy()
    resumen = (
        df_carrera.groupby(
            ["Perfil académico", "Orden de prioridad", "Nivel de atención"],
            dropna=False
        )
        .size()
        .reset_index(name="Estudiantes")
        .sort_values("Orden de prioridad")
    )

    fig = px.pie(
        resumen,
        names="Perfil académico",
        values="Estudiantes",
        hole=0.58,
        custom_data=["Orden de prioridad", "Nivel de atención"]
    )
    fig.update_traces(
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Estudiantes: %{value}<br>"
            "Porcentaje: %{percent}<extra></extra>"
        )
    )
    fig.update_layout(
        title=f"Distribución de perfiles — {carrera}",
        legend_title_text="Perfil",
        margin=dict(t=70, b=20, l=20, r=20)
    )

    evento = st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"donut_clusters_{util_limpiar_texto(carrera)}",
        on_select="rerun",
        selection_mode="points"
    )

    perfil_seleccionado = None
    try:
        puntos = evento.selection.points
        if puntos:
            perfil_seleccionado = puntos[0].get("label")
    except Exception:
        perfil_seleccionado = None

    perfiles = resumen["Perfil académico"].tolist()
    if perfil_seleccionado not in perfiles:
        perfiles_prioritarios = resumen.loc[
            resumen["Orden de prioridad"].isin([1, 2]),
            "Perfil académico"
        ].tolist()
        perfil_seleccionado = st.selectbox(
            "Perfil a consultar",
            perfiles,
            index=0,
            help=(
                "También puedes elegirlo aquí. Los perfiles prioritarios son: "
                + ", ".join(perfiles_prioritarios)
            )
        )
    else:
        st.success(f"Perfil seleccionado en el gráfico: {perfil_seleccionado}")

    lista = df_carrera[
        df_carrera["Perfil académico"] == perfil_seleccionado
    ].copy()
    lista = lista.sort_values(
        ["Distancia al centroide", "Resultado global EVALUATEC"],
        ascending=[True, True]
    )

    st.markdown(f"### Lista nominal — {perfil_seleccionado}")

    columnas_lista = [
        "Matrícula/ID",
        "Nombre",
        "Carrera",
        "Nivel de atención",
        "Promedio bachillerato",
        "Resultado global EVALUATEC",
        "Diagnóstico CHASIDE",
        "Área débil EVALUATEC 1",
        "Área débil EVALUATEC 2",
        "Área débil CHASIDE 1",
        "Área débil CHASIDE 2",
        "Distancia al centroide",
        "Canalización determinada por coordinación",
        "Observaciones de coordinación"
    ]
    columnas_lista = [c for c in columnas_lista if c in lista.columns]

    st.dataframe(
        lista[columnas_lista],
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# EXCEL MAESTRO
# ============================================================

def generar_excel_maestro(df_maestro):
    """
    Genera Excel con:
    - Concentrado maestro
    - Resumen por carrera
    - Resumen de talleres sugeridos por áreas débiles EVALUATEC
    """

    output = io.BytesIO()

    df_resumen_clusters = st.session_state.get(
        "df_resumen_clusters",
        pd.DataFrame()
    )
    df_metricas_clusters = st.session_state.get(
        "df_metricas_clusters",
        pd.DataFrame()
    )

    resumen_carrera = (
        df_maestro
        .groupby("Carrera", dropna=False)
        .agg(
            Estudiantes=("Nombre", "count"),
            Promedio_bachillerato=("Promedio bachillerato", "mean"),
            Promedio_EVALUATEC=("Resultado global EVALUATEC", "mean")
        )
        .reset_index()
    )

    resumen_carrera["Promedio_bachillerato"] = resumen_carrera[
        "Promedio_bachillerato"
    ].round(1)

    resumen_carrera["Promedio_EVALUATEC"] = resumen_carrera[
        "Promedio_EVALUATEC"
    ].round(1)

    talleres = []

    for columna in [
        "Área débil EVALUATEC 1",
        "Área débil EVALUATEC 2"
    ]:
        if columna not in df_maestro.columns:
            continue

        temp = (
            df_maestro[
                ["Carrera", "Nombre", columna]
            ]
            .rename(columns={columna: "Área sugerida para taller"})
            .copy()
        )

        temp = temp[
            temp["Área sugerida para taller"].notna()
            &
            (temp["Área sugerida para taller"].astype(str) != "Sin dato")
        ]

        talleres.append(temp)

    if talleres:
        df_talleres = pd.concat(
            talleres,
            ignore_index=True
        )

        resumen_talleres = (
            df_talleres
            .groupby(["Carrera", "Área sugerida para taller"])
            .size()
            .reset_index(name="Estudiantes sugeridos")
            .sort_values(
                ["Carrera", "Estudiantes sugeridos"],
                ascending=[True, False]
            )
        )

    else:
        resumen_talleres = pd.DataFrame(
            columns=[
                "Carrera",
                "Área sugerida para taller",
                "Estudiantes sugeridos"
            ]
        )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df_maestro.to_excel(
            writer,
            index=False,
            sheet_name="Concentrado maestro"
        )

        resumen_carrera.to_excel(
            writer,
            index=False,
            sheet_name="Resumen por carrera"
        )

        resumen_talleres.to_excel(
            writer,
            index=False,
            sheet_name="Talleres sugeridos"
        )

        if not df_resumen_clusters.empty:
            df_resumen_clusters.to_excel(
                writer,
                index=False,
                sheet_name="Resumen perfiles"
            )

        if not df_metricas_clusters.empty:
            df_metricas_clusters.to_excel(
                writer,
                index=False,
                sheet_name="Métricas clustering"
            )

        if "Atender prioritariamente" in df_maestro.columns:
            prioritarios = df_maestro[
                df_maestro["Atender prioritariamente"] == "Sí"
            ].sort_values(
                ["Carrera", "Orden de prioridad", "Nombre"]
            )
            prioritarios.to_excel(
                writer,
                index=False,
                sheet_name="Listas prioritarias"
            )

        workbook = writer.book

        for nombre_hoja in workbook.sheetnames:
            ws = workbook[nombre_hoja]

            ws.freeze_panes = "A2"

            for columna in ws.columns:
                max_length = 0
                letra_columna = columna[0].column_letter

                for celda in columna:
                    try:
                        valor = str(celda.value)
                        if valor:
                            max_length = max(max_length, len(valor))
                    except Exception:
                        pass

                ancho = min(max_length + 2, 45)
                ws.column_dimensions[letra_columna].width = ancho

    output.seek(0)
    return output.getvalue()


# ============================================================
# PANTALLA PRINCIPAL STREAMLIT
# ============================================================

def render_app_maestra():
    st.title("📚 Generador de Concentrado Maestro de Aspirantes")

    st.caption(
        "Integra Historial de Aspirantes, EVALUATEC y CHASIDE en una sola base "
        "sin aplicar boxplot. El archivo resultante puede alimentar una app HTML "
        "para consulta docente."
    )

    st.markdown("## 1. Carga de archivos")

    st.info(
        "Puedes usar los links precargados o sustituirlos por nuevos enlaces. "
        "Si prefieres carga manual, borra los links correspondientes y carga los archivos."
    )

    st.markdown("### Historial de Aspirantes")

    url_historial = st.text_input(
        "Link del Excel de Historial de Aspirantes",
        value=LINK_HISTORIAL_DEFAULT,
        key="url_historial_maestro"
    )

    archivo_historial = st.file_uploader(
        "O carga manualmente el Excel de Historial de Aspirantes",
        type=["xlsx", "xls"],
        key="archivo_historial_maestro"
    )

    st.markdown("### EVALUATEC")

    col_eval_1, col_eval_2, col_eval_3 = st.columns(3)

    with col_eval_1:
        url_evaluatec_adm = st.text_input(
            "Link CSV EVALUATEC Administración",
            value=LINK_EVALUATEC_ADM_DEFAULT,
            key="url_evaluatec_adm_maestro"
        )

    with col_eval_2:
        url_evaluatec_arq = st.text_input(
            "Link CSV EVALUATEC Arquitectura",
            value=LINK_EVALUATEC_ARQ_DEFAULT,
            key="url_evaluatec_arq_maestro"
        )

    with col_eval_3:
        url_evaluatec_ing = st.text_input(
            "Link CSV EVALUATEC Ingeniería",
            value=LINK_EVALUATEC_ING_DEFAULT,
            key="url_evaluatec_ing_maestro"
        )

    archivos_evaluatec = st.file_uploader(
        "O carga manualmente los 3 archivos CSV de EVALUATEC",
        type=["csv"],
        accept_multiple_files=True,
        key="archivos_evaluatec_maestro"
    )

    st.markdown("### CHASIDE")

    url_chaside = st.text_input(
        "Link de respuestas CHASIDE de Google Sheets",
        value=LINK_CHASIDE_DEFAULT,
        key="url_chaside_maestro"
    )

    peso_intereses = st.slider(
        "Peso de intereses CHASIDE",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        key="peso_intereses_maestro"
    )

    peso_aptitudes = round(1 - peso_intereses, 2)

    st.caption(
        f"Pesos CHASIDE activos → Intereses: {peso_intereses:.1f} | "
        f"Aptitudes: {peso_aptitudes:.1f}"
    )

    st.markdown("---")

    boton_generar = st.button(
        "🚀 Generar concentrado maestro",
        use_container_width=True
    )

    if not boton_generar and "df_maestro" not in st.session_state:
        st.info(
            "Carga los archivos o conserva los links precargados y presiona el botón para generar la super base."
        )
        st.stop()

    if boton_generar:

        contenido_historial = obtener_contenido_historial_desde_link_o_upload(
            url_historial=url_historial,
            archivo_historial=archivo_historial
        )

        if contenido_historial is None:
            st.error("Falta cargar o indicar el link del Historial de Aspirantes.")
            st.stop()

        archivos_evaluatec_finales = obtener_archivos_evaluatec_desde_links_o_uploads(
            url_adm=url_evaluatec_adm,
            url_arq=url_evaluatec_arq,
            url_ing=url_evaluatec_ing,
            archivos_evaluatec=archivos_evaluatec
        )

        if not archivos_evaluatec_finales or len(archivos_evaluatec_finales) != 3:
            st.error(
                "Debes tener exactamente 3 archivos de EVALUATEC. "
                "Puedes usar los 3 links precargados o cargar los 3 CSV manualmente."
            )
            st.stop()

        if url_chaside.strip() == "":
            st.warning(
                "No pegaste enlace CHASIDE. El concentrado se generará sin resultados vocacionales."
            )

        # ------------------------------------------------------------
        # Procesamiento Historial
        # ------------------------------------------------------------

        with st.spinner("Procesando Historial de Aspirantes..."):
            df_historial_raw, df_bitacora = procesar_archivo_historial_excel(
                contenido_historial
            )

            if df_historial_raw.empty:
                st.error("No se pudieron identificar estudiantes en el Historial.")
                st.dataframe(df_bitacora, use_container_width=True)
                st.stop()

            df_historial_preparado = preparar_historial_para_cruce(
                df_historial_raw
            )

        # ------------------------------------------------------------
        # Procesamiento EVALUATEC
        # ------------------------------------------------------------

        with st.spinner("Procesando archivos EVALUATEC..."):
            datos_eval_global = {}
            errores_eval = []

            for archivo in archivos_evaluatec_finales:
                try:
                    df_eval, areas_detectadas = procesar_archivo_evaluatec(
                        archivo
                    )

                    bloque = df_eval["Bloque EVALUATEC"].iloc[0]

                    datos_eval_global[bloque] = {
                        "df": df_eval,
                        "areas": areas_detectadas,
                        "archivo": archivo.name
                    }

                except Exception as error:
                    errores_eval.append(
                        f"{archivo.name}: {error}"
                    )

            if errores_eval:
                for error in errores_eval:
                    st.warning(error)

            if not datos_eval_global:
                st.error("No se pudo procesar ningún archivo EVALUATEC.")
                st.stop()

            df_evaluatec_preparado = preparar_evaluatec_desde_bloques(
                datos_eval_global
            )

            if df_evaluatec_preparado.empty:
                st.error("EVALUATEC se cargó, pero no se pudo preparar para cruce.")
                st.stop()

        # ------------------------------------------------------------
        # Procesamiento CHASIDE
        # ------------------------------------------------------------

        if url_chaside.strip() != "":
            with st.spinner("Procesando CHASIDE..."):
                try:
                    df_chaside_raw = cargar_respuestas_chaside(
                        url_chaside
                    )

                    df_chaside_procesado = procesar_respuestas_chaside(
                        df_chaside_raw,
                        peso_intereses=peso_intereses,
                        peso_aptitudes=peso_aptitudes
                    )

                except Exception as error:
                    st.warning(
                        f"No fue posible procesar CHASIDE. Se continuará sin CHASIDE. Detalle: {error}"
                    )
                    df_chaside_procesado = pd.DataFrame()
        else:
            df_chaside_procesado = pd.DataFrame()

        # ------------------------------------------------------------
        # Concentrado maestro
        # ------------------------------------------------------------

        with st.spinner("Generando concentrado maestro..."):
            df_maestro = generar_concentrado_maestro(
                df_historial_preparado=df_historial_preparado,
                df_evaluatec_preparado=df_evaluatec_preparado,
                df_chaside_procesado=df_chaside_procesado
            )

        if df_maestro.empty:
            st.error("No se pudo generar el concentrado maestro.")
            st.stop()

        with st.spinner("Aplicando clustering académico por carrera..."):
            df_maestro, df_resumen_clusters, df_metricas_clusters = (
                aplicar_clustering_por_carrera(df_maestro)
            )

        st.session_state["df_maestro"] = df_maestro.copy()
        st.session_state["df_resumen_clusters"] = df_resumen_clusters.copy()
        st.session_state["df_metricas_clusters"] = df_metricas_clusters.copy()

        if "archivo_excel_maestro" in st.session_state:
            del st.session_state["archivo_excel_maestro"]

    df_maestro = st.session_state["df_maestro"].copy()

    # ------------------------------------------------------------
    # Vista ejecutiva
    # ------------------------------------------------------------

    st.success(
        f"Concentrado maestro generado correctamente: {len(df_maestro):,} estudiantes."
    )

    total_estudiantes = len(df_maestro)
    total_carreras = df_maestro["Carrera"].dropna().nunique()

    con_evaluatec = df_maestro[
        df_maestro["Resultado global EVALUATEC"].notna()
    ].shape[0]

    con_chaside = df_maestro[
        df_maestro["Diagnóstico CHASIDE"] != "Sin respuesta CHASIDE"
    ].shape[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Estudiantes", f"{total_estudiantes:,}")
    col2.metric("Carreras", f"{total_carreras:,}")
    col3.metric("Con EVALUATEC", f"{con_evaluatec:,}")
    col4.metric("Con CHASIDE", f"{con_chaside:,}")

    st.markdown("## Vista previa del concentrado")

    columnas_vista = [
        "Nombre",
        "Matrícula/ID",
        "Carrera",
        "Estatus cruce",
        "Promedio bachillerato",
        "Resultado global EVALUATEC",
        "Área débil EVALUATEC 1",
        "Área débil EVALUATEC 2",
        "Diagnóstico CHASIDE",
        "Estatus cruce CHASIDE",
        "Perfil académico",
        "Nivel de atención",
        "Atender prioritariamente"
    ]

    columnas_vista = [
        columna
        for columna in columnas_vista
        if columna in df_maestro.columns
    ]

    st.dataframe(
        df_maestro[columnas_vista],
        use_container_width=True,
        hide_index=True
    )

    render_vista_clusters(df_maestro)

    # ------------------------------------------------------------
    # Descarga
    # ------------------------------------------------------------

    st.markdown("## Descarga")

    if st.button(
        "📄 Preparar archivo Excel",
        use_container_width=True
    ):
        with st.spinner("Preparando Excel maestro..."):
            archivo_excel = generar_excel_maestro(df_maestro)

        st.session_state["archivo_excel_maestro"] = archivo_excel

    if "archivo_excel_maestro" in st.session_state:
        st.download_button(
            label="⬇️ Descargar concentrado maestro en Excel",
            data=st.session_state["archivo_excel_maestro"],
            file_name="concentrado_maestro_aspirantes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
# ============================================================
# COMPATIBILIDAD DE NOMBRES
# ============================================================

def procesar_archivo_historial_excel(contenido_archivo):
    """
    Alias compatible para procesar Historial.
    """
    return hist_procesar_archivo_excel(contenido_archivo)


def procesar_archivo_evaluatec(archivo):
    """
    Alias compatible para procesar EVALUATEC.
    """
    return eval_procesar_archivo(archivo)


def cargar_respuestas_chaside(url):
    """
    Alias compatible para cargar CHASIDE.
    """
    return chaside_cargar_respuestas(url)


def procesar_respuestas_chaside(
    df_raw,
    peso_intereses=0.8,
    peso_aptitudes=0.2
):
    """
    Alias compatible para procesar CHASIDE.
    """
    return chaside_procesar_respuestas(
        df_raw=df_raw,
        peso_intereses=peso_intereses,
        peso_aptitudes=peso_aptitudes
    )


# ============================================================
# VALIDACIÓN FINAL DE FUNCIONES NECESARIAS
# ============================================================

def validar_funciones_requeridas():
    """
    Valida que existan las funciones principales antes de ejecutar la app.
    """

    funciones = [
        "hist_procesar_archivo_excel",
        "eval_procesar_archivo",
        "chaside_cargar_respuestas",
        "chaside_procesar_respuestas",
        "preparar_historial_para_cruce",
        "preparar_evaluatec_desde_bloques",
        "generar_concentrado_maestro",
        "aplicar_clustering_por_carrera",
        "render_vista_clusters",
        "generar_excel_maestro",
        "render_app_maestra"
    ]

    faltantes = []

    for funcion in funciones:
        if funcion not in globals():
            faltantes.append(funcion)

    if faltantes:
        st.error(
            "Faltan funciones necesarias para ejecutar la app: "
            + ", ".join(faltantes)
        )
        st.stop()


# ============================================================
# EJECUCIÓN
# ============================================================

validar_funciones_requeridas()
render_app_maestra()
