from pathlib import Path
from prefect import flow, task
import mlflow
import mlflow.sklearn  # asegura el flavor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature

# -------- Rutas absolutas fiables --------
ROOT = Path(__file__).resolve().parents[2]          # .../Aprendizaje_automatico_en_la_nube
DATA_PATH = ROOT / "data" / "processed" / "features_telco.csv"
TRACKING_DIR = ROOT / "logs" / "mlruns"             # misma carpeta que abre tu UI

def _find_churn_col(cols):
    norm = {c: c.strip().lower() for c in cols}
    for c, lc in norm.items():
        if lc == "churn":
            return c
    for c, lc in norm.items():
        if lc in ("churn_1", "churn_yes", "churn_true"):
            return c
    raise ValueError("No encuentro la columna de etiqueta 'Churn'.")

@task
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded: {df.shape}  ({path})")
    return df

@task
def train_model(df: pd.DataFrame):
    y_col = _find_churn_col(df.columns)
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[y_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1}
    model = RandomForestClassifier(**params).fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained with accuracy: {acc:.4f}")

    return model, X_train, acc, params

@task
def log_model_mlflow(model, X_train, acc: float, params: dict):
    # IMPORTANTÍSIMO: que la UI y el código apunten al MISMO tracking
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())      # p.ej. file:///C:/.../logs/mlruns
    mlflow.set_experiment("telco_churn_prefect")
    print("🔎 TRACKING_URI   :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="rf_pipeline_prefect"):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(2)

        # 👇 CLAVE: artifact_path="model" (no uses name=)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        print("🔎 ARTIFACT_URI   :", mlflow.get_artifact_uri())
        print("✅ Model logged under artifact path: model/")

@flow(name="TrainingFlow")
def training_flow():
    df = load_data(DATA_PATH)
    model, X_train, acc, params = train_model(df)
    log_model_mlflow(model, X_train, acc, params)

if __name__ == "__main__":
    training_flow()

