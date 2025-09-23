import pandas as pd
from pathlib import Path
from typing import Optional

class FeatureEngineer:
    def __init__(self):
        pass

    def _find_churn_col(self, cols) -> Optional[str]:
        # busca una columna cuyo nombre, sin espacios y en minúsculas, sea "churn"
        for c in cols:
            if c.strip().lower() == "churn":
                return c
        return None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🛠️ Creando y transformando variables...")
        df = df.copy()

        # 0) Normaliza nombres de columnas (solo trim; mantenemos mayúsculas)
        df.columns = [c.strip() for c in df.columns]

        # 1) Asegura TotalCharges numérico si existe
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # 2) Detecta columna de churn (nombre flexible)
        churn_col = self._find_churn_col(df.columns)
        if churn_col is None:
            raise ValueError("No encuentro la columna objetivo 'Churn' (con o sin espacios).")

        # 3) Mapea churn a 0/1 si es texto
        if df[churn_col].dtype == "object":
            m = {
                "yes": 1, "no": 0,
                "si": 1, "sí": 1,
                "true": 1, "false": 0,
                "1": 1, "0": 0
            }
            df[churn_col] = (
                df[churn_col].astype(str).str.strip().str.lower().map(m)
            )

        # Si aún queda algo no convertible, pon 0 por defecto y convierte a int
        df[churn_col] = pd.to_numeric(df[churn_col], errors="coerce").fillna(0).astype(int)

        # 4) Quita identificadores que no aportan
        for col in ["customerID", "CustomerID", "customer_id"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # 5) One-hot a categóricas (excepto la etiqueta)
        cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != churn_col]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # 6) Rellena nulos
        df = df.fillna(0)

        print(f"✅ Features listas: {df.shape[0]} filas x {df.shape[1]} columnas. Etiqueta: {churn_col}")
        return df, churn_col


if __name__ == "__main__":
    inp = Path("data/processed/clean_telco.csv")
    out = Path("data/processed/features_telco.csv")
    print(f"📥 Leyendo {inp.resolve()}")
    df = pd.read_csv(inp)
    fe = FeatureEngineer()
    df_feat, churn_col = fe.create_features(df)
    print(f"💾 Guardando en {out.resolve()}")
    df_feat.to_csv(out, index=False)
    print("🎉 Feature engineering listo.")
