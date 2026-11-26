# ==============================================================================
# CHALLENGE LAB: ENTRENAMIENTO DEL MODELO DE MANTENIMIENTO PREDICTIVO
# Clase: ISOF CLASS | Docente: Dra. Aiblis Susej Herrera Palafox
# ==============================================================================

import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ARCHIVO_CSV = "sensores_planta.csv"
ARCHIVO_MODELO = "modelo_mantenimiento.pkl"
COLUMNAS_X = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
COLUMNA_Y = "estado"

# --- 1. Cargar el dataset ---
print(f"📂 1. Cargando conjunto de datos desde '{ARCHIVO_CSV}'...")
try:
    df = pd.read_csv(ARCHIVO_CSV)
except FileNotFoundError:
    print(f"❌ Error: No se encontró '{ARCHIVO_CSV}'. Ejecuta el script desde la carpeta del proyecto.")
    sys.exit(1)

df.columns = df.columns.str.strip().str.lower()
print(f"✅ Registros cargados: {len(df)}")

# --- 2. Limpieza de nulos ---
df = df.dropna()
print(f"🧹 Registros después de dropna(): {len(df)}")

faltantes = [c for c in COLUMNAS_X + [COLUMNA_Y] if c not in df.columns]
if faltantes:
    print(f"❌ Error: Faltan las columnas {faltantes} en el CSV.")
    sys.exit(1)

# --- 3. Variables de entrada (X) y objetivo (y) ---
X = df[COLUMNAS_X]
y = df[COLUMNA_Y].astype(int)
print(f"📋 Registros por estado: {y.value_counts().sort_index().to_dict()}")

# --- 4. División 80% entrenamiento / 20% prueba ---
# stratify=y mantiene la misma proporción de cada estado en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Entrenamiento ---
print("🧠 2. Entrenando RandomForestClassifier...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# --- 6. Evaluación ---
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=["0 Normal", "1 Preventivo", "2 Crítico"],
        zero_division=0,
    )
)

if precision < 0.80:
    print("⚠️ La precisión es menor a 80%. Revisa o amplía el dataset.")

# --- 7. Exportar el modelo ---
joblib.dump(modelo, ARCHIVO_MODELO)
print(f"✅ 3. ¡Éxito! Modelo exportado como '{ARCHIVO_MODELO}'.")
