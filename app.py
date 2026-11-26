# ==============================================================================
# CHALLENGE LAB: SISTEMA PREDICTIVO DE MANTENIMIENTO INDUSTRIAL + GROQ API
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ==============================================================================

import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

# --- 1. Configuración de la página ---
st.set_page_config(
    page_title="Mantenimiento Predictivo", page_icon="🏭", layout="wide"
)

COLUMNAS_X = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
ESTADOS = {
    0: {"nombre": "Operación Normal", "icono": "✅"},
    1: {"nombre": "Mantenimiento Preventivo", "icono": "⚠️"},
    2: {"nombre": "Fallo Crítico", "icono": "🚨"},
}


# --- 2. API Key de Groq (st.secrets → .env → variable de entorno) ---
def leer_env_local(nombre, archivo=".env"):
    """Lee una variable de un archivo .env sin depender de librerías extra."""
    ruta = Path(__file__).parent / archivo
    if not ruta.exists():
        return None
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            if clave.strip() == nombre:
                return valor.strip().strip('"').strip("'")
    return None


def obtener_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return leer_env_local("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")


# --- 3. Cargar el modelo entrenado (la app solo depende del .pkl) ---
@st.cache_resource
def cargar_modelo():
    return joblib.load(Path(__file__).parent / "modelo_mantenimiento.pkl")


try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        "❌ No se encontró `modelo_mantenimiento.pkl`. Ejecuta primero `python entrenar.py`."
    )
    st.stop()


# --- 4. Detección dinámica de un modelo de chat activo en Groq ---
@st.cache_data(ttl=3600, show_spinner=False)
def detectar_modelo_groq(api_key):
    client = Groq(api_key=api_key)
    modelos_activos = [
        m.id
        for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "guard", "tts", "playai", "embed"])
    ]
    if not modelos_activos:
        raise RuntimeError("No hay modelos de texto disponibles en tu cuenta de Groq.")
    return next((m for m in modelos_activos if "llama" in m.lower()), modelos_activos[0])


# --- 5. Interfaz: encabezado y controles laterales ---
st.title("🏭 Sistema Predictivo de Mantenimiento Industrial")
st.write(
    "Sistema híbrido: **Machine Learning (Random Forest)** para clasificar el estado de la "
    "maquinaria + **Groq API** para generar la Orden de Trabajo técnico."
)

st.sidebar.header("📡 Telemetría de Sensores")
temperatura = st.sidebar.slider("Temperatura (°C)", 40.0, 110.0, 60.0, step=0.5)
vibracion = st.sidebar.slider("Vibración (mm/s)", 0.0, 12.0, 1.5, step=0.1)
horas = st.sidebar.slider("Horas de operación", 0, 5000, 300, step=10)
presion = st.sidebar.slider("Presión (PSI)", 80.0, 160.0, 92.0, step=0.5)

# Tarjetas con las lecturas actuales
col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temperatura", f"{temperatura} °C")
col2.metric("📳 Vibración", f"{vibracion} mm/s")
col3.metric("⏱️ Operación", f"{horas} hrs")
col4.metric("🔧 Presión", f"{presion} PSI")

st.divider()

# --- 6. Predicción con el modelo de ML ---
datos = pd.DataFrame(
    [[temperatura, vibracion, horas, presion]], columns=COLUMNAS_X
)
estado = int(modelo.predict(datos)[0])
probabilidades = modelo.predict_proba(datos)[0]
prob_por_clase = {int(c): p for c, p in zip(modelo.classes_, probabilidades)}
prob_estado = prob_por_clase[estado] * 100

st.subheader("📊 Diagnóstico del Modelo Predictivo")
mensaje = f"{ESTADOS[estado]['icono']} **Estado {estado}: {ESTADOS[estado]['nombre']}** (Probabilidad: {prob_estado:.1f}%)"
if estado == 0:
    st.success(mensaje)
elif estado == 1:
    st.warning(mensaje)
else:
    st.error(mensaje)

with st.expander("Ver probabilidad de cada estado"):
    for clase in (0, 1, 2):
        p = prob_por_clase.get(clase, 0.0)
        st.write(f"{ESTADOS[clase]['icono']} {clase} – {ESTADOS[clase]['nombre']}: **{p * 100:.1f}%**")
        st.progress(float(p))

# --- 7. Orden de Trabajo con Groq ---
st.subheader("🤖 Orden de Trabajo Técnico (Groq API)")

if st.button("Generar dictamen / Orden de Trabajo", type="primary"):
    folio = f"OT-{datetime.now():%Y%m%d-%H%M%S}"
    lecturas = (
        f"- Temperatura: {temperatura} °C\n"
        f"- Vibración: {vibracion} mm/s\n"
        f"- Horas de operación: {horas} hrs\n"
        f"- Presión: {presion} PSI\n"
        f"- Diagnóstico del modelo: Estado {estado} – {ESTADOS[estado]['nombre']} "
        f"(probabilidad {prob_estado:.1f}%)"
    )

    prompt_sistema = (
        "Eres un Jefe de Mantenimiento Industrial con amplia experiencia en plantas de "
        "manufactura automatizada. Respondes en español, de forma técnica, clara y concisa."
    )

    if estado == 0:
        prompt_usuario = f"""Un modelo de Machine Learning analizó la telemetría de una máquina:
{lecturas}

La máquina opera con normalidad. Redacta en Markdown un **dictamen breve de monitoreo preventivo**
(máximo 5 líneas) que confirme el estado estable e indique qué seguir vigilando. NO generes una orden de trabajo."""
    else:
        urgencia = (
            "URGENTE: incluye un protocolo de paro inmediato y bloqueo/etiquetado (LOTO) antes de cualquier intervención."
            if estado == 2
            else "PREVENTIVA: la intervención se debe programar en la próxima ventana de mantenimiento."
        )
        prompt_usuario = f"""Un modelo de Machine Learning analizó la telemetría de una máquina:
{lecturas}

Genera una **Orden de Trabajo** en Markdown. Prioridad: {urgencia}
Usa exactamente esta estructura:
## Orden de Trabajo – Folio {folio}
**Prioridad:** ...
### Diagnóstico
(qué lecturas están fuera de rango y la causa probable)
### Acciones Técnicas
(exactamente 3 acciones numeradas)
### Equipo de Protección Personal (EPP)
(lista del EPP requerido)"""

    api_key = obtener_api_key()
    if not api_key:
        st.error(
            "❌ No se encontró la API Key de Groq. Agrega `GROQ_API_KEY` en `.streamlit/secrets.toml` o en `.env`. "
            "El diagnóstico de Machine Learning de arriba sigue siendo válido."
        )
    else:
        try:
            with st.spinner("Generando con Groq..."):
                modelo_groq = detectar_modelo_groq(api_key)
                client = Groq(api_key=api_key)
                respuesta = client.chat.completions.create(
                    model=modelo_groq,
                    messages=[
                        {"role": "system", "content": prompt_sistema},
                        {"role": "user", "content": prompt_usuario},
                    ],
                    temperature=0.4,
                    max_tokens=1024,
                )
            st.caption(f"⚡ Generado con el modelo: `{modelo_groq}`")
            st.markdown(respuesta.choices[0].message.content)
        except Exception as e:
            # Resiliencia: si la API falla (clave inválida, límite de peticiones, sin red),
            # la app no colapsa y el diagnóstico de ML sigue visible.
            st.error(f"❌ No se pudo generar la orden con Groq: {e}")
            st.info("El diagnóstico de Machine Learning mostrado arriba sigue siendo válido.")
