# src/api/app.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc as mlpyfunc


app = FastAPI(title="Telco Churn API", version="1.1.0")


# ───────────────────────────────────────────────────────────────────────────────
# 1) DÓNDE ENCONTRAR EL MODELO
# ───────────────────────────────────────────────────────────────────────────────
# Si no se encuentra automáticamente, se usa este respaldo.
# 👉 Cambia SOLO si tu carpeta real es distinta:
FALLBACK_MODEL_DIR = Path(
    "logs/mlruns/202835467819020042/models/m-e71d34a67e0d4b14a5975a6e4a7372fe/artifacts"
)

SEARCH_ROOTS = [Path("logs/mlruns"), Path("mlruns")]  # dónde buscar por defecto


def _find_latest_model_dir() -> Path:
    """
    Busca el 'model dir' que contiene un MLmodel bajo:
      logs/mlruns/**/artifacts/model/MLmodel  (y equivalente en mlruns/)
    Si no aparece, usa FALLBACK_MODEL_DIR.
    """
    candidates: List[Path] = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        # buscamos MLmodel exactamente en artifacts/model
        for mlmodel_file in root.glob("**/artifacts/model/MLmodel"):
            model_dir = mlmodel_file.parent  # .../artifacts/model
            candidates.append(model_dir)

    if candidates:
        # el más reciente por fecha de modificación
        return max(candidates, key=lambda p: p.stat().st_mtime)

    # Si no hay nada, probamos respaldo:
    if FALLBACK_MODEL_DIR.exists():
        print("⚠️ Usando modelo de respaldo:", FALLBACK_MODEL_DIR)
        return FALLBACK_MODEL_DIR

    # Nada de nada:
    raise FileNotFoundError(
        "No encontré modelos bajo logs/mlruns/**/artifacts/model o mlruns/**/artifacts/model.\n"
        "Revisa que hayas entrenado y logueado el modelo con MLflow.\n"
        "También puedes ajustar FALLBACK_MODEL_DIR con tu ruta real."
    )


# ───────────────────────────────────────────────────────────────────────────────
# 2) UTILIDADES DE TIPOS
# ───────────────────────────────────────────────────────────────────────────────
def _truthy_to_bool(val: Any) -> bool:
    """
    Convierte valores “truthy” a boolean:
    - 'yes', 'true', '1', 1  -> True
    - 'no', 'false', '0', 0  -> False
    - otros -> False por defecto (para robustez)
    """
    if isinstance(val, bool):
        return val
    if val is None:
        return False

    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "si", "sí"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False

    # Si llega algo raro, mejor False que romper:
    return False


def _to_float(val: Any) -> float:
    """Convierte a float con fallback a 0.0 para robustez."""
    try:
        return float(val)
    except Exception:
        return 0.0


# ───────────────────────────────────────────────────────────────────────────────
# 3) ARRANQUE: CARGAR MODELO Y FIRMAS
# ───────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _load_model() -> None:
    model_dir = _find_latest_model_dir()
    # mlflow.pyfunc.load_model trabaja con la carpeta que contiene MLmodel
    app.state.model_dir = model_dir
    app.state.model = mlpyfunc.load_model(str(model_dir))

    # Intentar obtener la firma (columnas esperadas) desde MLflow
    # (si no tiene firma, trabajaremos con lo que llegue y completaremos)
    try:
        model_meta = app.state.model.metadata
        signature = model_meta.get_input_schema()
        # Para pyfunc, podemos intentar:
        expected_cols = [c.name for c in signature.inputs] if signature and signature.inputs else None
    except Exception:
        expected_cols = None

    # Guardamos columnas esperadas; si no hay, None
    app.state.expected_cols: Optional[List[str]] = expected_cols

    print("✅ Modelo cargado desde:", model_dir.as_posix())
    if expected_cols:
        print(f"🧩 Columnas esperadas ({len(expected_cols)}):", expected_cols)
    else:
        print("ℹ️ El modelo no tiene firma declarada; se intentará alinear de forma flexible.")


# ───────────────────────────────────────────────────────────────────────────────
# 4) HOME
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Home"])
def home() -> Dict[str, Any]:
    return {
        "message": "API de predicción de Churn lista ✅",
        "model_path": str(app.state.model_dir),
        "tracking_uri_actual": mlflow.get_tracking_uri(),
        "expected_columns": app.state.expected_cols or "No declaradas (se alinean de forma flexible)",
    }


# ───────────────────────────────────────────────────────────────────────────────
# 5) PREDICCIÓN FLEXIBLE
# ───────────────────────────────────────────────────────────────────────────────
def _align_payload_to_model(df_input: pd.DataFrame, expected_cols: Optional[List[str]]) -> pd.DataFrame:
    """
    Alinea el DataFrame de entrada con lo que el modelo espera:
    - Si expected_cols existe: añade columnas faltantes con 0, reordena.
    - Convierte columnas *_Yes a boolean (0/1).
    - Convierte numéricas típicas a float (tenure, MonthlyCharges, TotalCharges).
    """
    df = df_input.copy()

    # Normalizamos nombres: sin espacios, tal cual como los guarda tu pipeline (si aplica)
    # (Si tu pipeline mantiene los nombres exactos, no cambies esto)
    # df.columns = [c.strip() for c in df.columns]

    # Tipos típicos del Telco (ajusta si usas otros nombres):
    numeric_like = {"tenure", "MonthlyCharges", "TotalCharges"}
    bool_like_suffix = "_Yes"  # por convención en one-hot

    for col in list(df.columns):
        if col in numeric_like:
            df[col] = df[col].apply(_to_float)

        if col.endswith(bool_like_suffix):
            # convertimos a {0,1} desde bool/string/num
            df[col] = df[col].apply(lambda v: int(_truthy_to_bool(v)))

    # Si el modelo declaró columnas esperadas, aseguramos presencia y orden:
    if expected_cols:
        # Añadir faltantes con 0
        for col in expected_cols:
            if col not in df.columns:
                # heurística: si es booleana por sufijo _Yes => 0; si no, 0.0
                default_val = 0 if col.endswith(bool_like_suffix) else 0.0
                df[col] = default_val

        # Reordenar
        df = df[[c for c in expected_cols]]

    else:
        # No hay firma: al menos convertimos *_Yes a 0/1 y rellenamos NaN con 0
        for col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)
        df = df.fillna(0)

    return df


@app.post("/predict", tags=["default"])
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recibe un JSON flexible. Ejemplos válidos:

    {
      "tenure": 12,
      "MonthlyCharges": 70.5,
      "TotalCharges": 845.2
    }

    O con columnas one-hot como "..._Yes": true/1/"yes".
    """
    try:
        # Construir DataFrame con una sola fila
        df_in = pd.DataFrame([payload])

        # Alinear con lo que espera el modelo
        df_aligned = _align_payload_to_model(df_in, app.state.expected_cols)

        # Predicción
        pred = app.state.model.predict(df_aligned)

        # MLflow/pyfunc puede devolver lista/ndarray
        y = pred[0] if isinstance(pred, (list, np.ndarray, pd.Series)) else pred

        # Normalizamos a int si es clasificatorio (0/1)
        try:
            y = int(y)
        except Exception:
            pass

        return {
            "ok": True,
            "churn_prediction": y,
            "used_columns": app.state.expected_cols or list(df_aligned.columns),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




