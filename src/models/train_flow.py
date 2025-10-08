# src/models/train_flow.py
from pathlib import Path
import pandas as pd

from prefect import flow, task

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


@task
def load_data() -> pd.DataFrame:
    """Carga dataset Telco (usa features si existe)."""
    candidates = [
        "data/processed/features_telco.csv",                 # preferido (features)
        "data/Telco-Customer-Churn.csv",
        "data/raw/Telco-Customer-Churn.csv",
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    found = next((p for p in candidates if Path(p).exists()), None)
    if not found:
        raise FileNotFoundError(
            "No encontré dataset. Coloca alguno en:\n"
            " - data/processed/features_telco.csv\n"
            " - data/Telco-Customer-Churn.csv\n"
            " - data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv\n"
        )

    print(f"✅ Usando dataset: {found}")
    df = pd.read_csv(found)

    # Detectar/normalizar etiqueta
    y_col = None
    for k in ["Churn", "churn", "churn_yes", "churn_1", "target", "label"]:
        if k in df.columns:
            y_col = k
            break

    if y_col is None:
        raise ValueError("No encuentro columna de etiqueta (Churn).")

    if df[y_col].dtype == object:
        df[y_col] = df[y_col].map({"Yes": 1, "No": 0}).fillna(df[y_col])
        # si quedan strings, fuerza a int/0-1 si posible
        try:
            df[y_col] = df[y_col].astype(int)
        except Exception:
            pass

    # Para baseline: si hay muchas categóricas y no es el features, me quedo con numéricas
    if "processed" not in found:
        only_num = df.select_dtypes(include="number")
        if y_col not in only_num.columns:
            only_num[y_col] = df[y_col].values
        df = only_num

    print(f"✅ Dataset shape: {df.shape}")
    return df


@task
def train_and_log(df: pd.DataFrame) -> str:
    """Entrena, loguea en MLflow y devuelve la ruta del artifact (modelo)."""
    # separa X/y
    y_col = next(c for c in df.columns if c.lower().startswith("churn"))
    y = df[y_col]
    X = df.drop(columns=[y_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLflow config —> usa ./mlruns (tu tracking actual)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("telco_churn")

    with mlflow.start_run(run_name="rf_prefect"):
        params = {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", float(acc))

        signature = infer_signature(X_train, model.predict(X_train))
        # 🔴 IMPORTANTE: artifact_path="model" para que quede .../artifacts/model/MLmodel
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
        )

        artifact_uri = mlflow.get_artifact_uri()
        print("📦 ARTIFACT_URI:", artifact_uri)
        print("✅ Modelo logueado en:", artifact_uri + "/model")

        return artifact_uri + "/model"


@flow(name="TrainingFlow")
def training_flow():
    df = load_data()
    model_artifact = train_and_log(df)
    print("🔗 MODEL_ARTIFACT:", model_artifact)


if __name__ == "__main__":
    training_flow()



