# src/models/Train.py
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def _find_churn_col(cols):
    """Devuelve el nombre real de la columna objetivo (Churn)."""
    for c in cols:
        if c.strip().lower() == "churn":
            return c
    # fallback si quedó como dummy
    for c in cols:
        lc = c.strip().lower()
        if lc in ("churn_1", "churn_yes", "churn_true"):
            return c
    raise ValueError("No encuentro la columna de etiqueta 'Churn'.")


def main():
    # ---------- Cargar features ----------
    path = Path("data/processed/features_telco.csv")
    print(f"Reading features: {path.resolve()}")
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero src/features/Feature_Engineer.py"
        )

    df = pd.read_csv(path)

    # ---------- Separar X / y ----------
    y_col = _find_churn_col(df.columns)
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[y_col])

    print(f"Label column: {y_col}")
    print(f"Split -> X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- MLflow ----------
    mlflow.set_tracking_uri("file:./logs/mlruns")  # misma carpeta que la UI
    print("Tracking URI:", mlflow.get_tracking_uri())
    mlflow.set_experiment("telco_churn")

    with mlflow.start_run(run_name="rf_baseline"):
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "random_state": 42,
            "n_jobs": -1,
        }
        print("Training RandomForest with params:", params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"Accuracy: {acc:.4f}")

        # Log en MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Firma e input_example para evitar warnings
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(2),
            signature=signature,
        )

    print("Training finished and logged to MLflow.")


if __name__ == "__main__":
    main()


