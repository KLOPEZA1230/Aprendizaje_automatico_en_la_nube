# src/api/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse, unquote

import pandas as pd
from fastapi import FastAPI, Body, HTTPException
import mlflow.pyfunc as mlpyfunc


app = FastAPI(title="Telco Churn API", version="1.1.1")


# ---------- utilidades ----------

def _path_from_maybe_file_uri(p: str) -> Path:
    """
    Convierte 'file:///C:/...' a ruta local en Windows/Linux si aplica.
    Si no es URI (no empieza por file:), lo devuelve tal cual.
    """
    if isinstance(p, str) and p.lower().startswith("file:"):
        u = urlparse(p)
        local = unquote(u.path or "")
        # En Windows, urlparse deja '/C:/...' -> quitar el primer '/'
        if os.name == "nt" and local.startswith("/") and len(local) > 2 and local[2] == ":":
            local = local[1:]
        return Path(local)
    return Path(p)


def _resolve_model_dir() -> Path:
    """
    Devuelve el directorio del modelo a cargar (carpeta que contiene 'MLmodel').

    1) Si existe MODEL_PATH:
       - Si es archivo MLmodel -> usar su carpeta padre
       - Si es carpeta con MLmodel -> usarla
       - Si apunta a .../artifacts -> usar .../artifacts/model
       - Si apunta a carpeta 'run' que contenga 'artifacts/model/MLmodel' -> usarla
       - Acepta tanto ruta local como URI file:///
    2) Si no hay MODEL_PATH, busca el MLmodel más reciente bajo:
       - logs/mlruns/**/artifacts/model
       - mlruns/**/artifacts/model
    """
    env_path = os.getenv("MODEL_PATH")

    def normalize_candidate_dir(p: Path) -> Optional[Path]:
        if p.is_file() and p.name == "MLmodel":
            return p.parent
        if p.is_dir():
            # caso exacto .../model
            if (p / "MLmodel").exists():
                return p
            # caso .../artifacts
            if (p / "model" / "MLmodel").exists():
                return p / "model"
            # caso carpeta del run con subcarpeta artifacts/model
            if (p / "artifacts" / "model" / "MLmodel").exists():
                return p / "artifacts" / "model"
        return None

    if env_path:
        p = _path_from_maybe_file_uri(env_path)
        cand = normalize_candidate_dir(p)
        if cand:
            return cand
        raise FileNotFoundError(
            "MODEL_PATH no es válido o no contiene un modelo de MLflow.\n"
            f"Valor recibido: {env_path}\n"
            "Debes apuntar a la carpeta que contiene 'MLmodel' (p. ej. .../artifacts/model)\n"
            "o a '.../MLmodel' directamente, o eliminar MODEL_PATH para autodetección."
        )

    # --- Búsqueda automática ---
    roots = [Path("logs/mlruns"), Path("mlruns")]
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        # .../artifacts/model/MLmodel
        for mlm in root.rglob("artifacts/model/MLmodel"):
            candidates.append(mlm.parent)
        # fallback: .../model/MLmodel por si el layout difiere
        for mlm in root.rglob("model/MLmodel"):
            candidates.append(mlm.parent)

    if not candidates:
        raise FileNotFoundError(
            "No encontré modelos bajo 'mlruns/**/artifacts/model' ni 'logs/mlruns/**/artifacts/model'.\n"
            "Ejecuta primero el entrenamiento para que MLflow guarde el modelo, "
            "o define MODEL_PATH apuntando a la carpeta del modelo."
        )

    # Elegir el más reciente por fecha de modificación
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_expected_columns_from_input_example(model_dir: Path) -> Optional[List[str]]:
    """
    Intenta leer columnas esperadas desde input_example.json (si fue registrado por MLflow).
    """
    ie = model_dir / "input_example.json"
    if ie.exists():
        # MLflow suele guardar el input_example con orient="split"
        try:
            df = pd.read_json(ie, orient="split")
            return list(df.columns)
        except Exception:
            pass
        # fallback: records
        try:
            df = pd.read_json(ie)
            if isinstance(df, pd.DataFrame):
                return list(df.columns)
        except Exception:
            pass
    return None


def _coerce_booleans_and_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte strings típicos a booleanos/números cuando es razonable.
    """
    truthy = {"true", "yes", "y", "1", "si", "sí"}
    falsy = {"false", "no", "n", "0"}

    def coerce_cell(x: Any) -> Any:
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in truthy:
                return 1
            if xs in falsy:
                return 0
            try:
                if "." in xs:
                    return float(xs)
                return int(xs)
            except Exception:
                return x
        return x

    return df.applymap(coerce_cell)


def _align_to_expected_columns(
    df: pd.DataFrame, expected_cols: Optional[List[str]]
) -> pd.DataFrame:
    """
    Si conocemos las columnas esperadas, rellena faltantes con 0 y reordena.
    Si no, devuelve df tal cual.
    """
    if not expected_cols:
        return df

    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0

    # descartar columnas extra no esperadas
    return df[expected_cols]


# ---------- ciclo de vida ----------

@app.on_event("startup")
def _load_model_on_startup():
    model_dir = _resolve_model_dir()
    app.state.model_dir = model_dir
    app.state.model = mlpyfunc.load_model(str(model_dir))

    # Intentar deducir columnas esperadas
    columns = _read_expected_columns_from_input_example(model_dir)
    if not columns:
        # Intento con el schema del modelo (MLflow 2.x)
        try:
            schema = app.state.model.metadata.get_input_schema()
            columns = [x.name for x in getattr(schema, "inputs", [])] or None
        except Exception:
            columns = None

    app.state.expected_columns = columns
    print("✅ Modelo cargado desde:", model_dir)
    if columns:
        print(f"✅ Columnas esperadas ({len(columns)}): {columns}")


# ---------- endpoints ----------

@app.get("/")
def home():
    return {
        "message": "API de predicción de Churn lista ✅",
        "model_path": str(getattr(app.state, "model_dir", "")),
        "expected_columns": app.state.expected_columns,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"expected_columns": app.state.expected_columns}


@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...)):
    """
    Acepta:
    - Un dict con un solo registro
      { "tenure": 12, "MonthlyCharges": 70.5, ... }
    - Una lista de dicts
      [ {..}, {..} ]
    Devuelve: {"predictions": [...]}
    """
    try:
        # Normalizar a lista de registros
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list):
            if not payload:
                raise ValueError("La lista de registros está vacía.")
            records = payload
        else:
            raise ValueError("El cuerpo debe ser un objeto JSON o una lista de objetos.")

        df = pd.DataFrame.from_records(records)
        df = _coerce_booleans_and_numbers(df)
        df = _align_to_expected_columns(df, app.state.expected_columns)

        # Predicción vía pyfunc
        preds = app.state.model.predict(df)

        # Convertir a tipos JSON-friendly
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
