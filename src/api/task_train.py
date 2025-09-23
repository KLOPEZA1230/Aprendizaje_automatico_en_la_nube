from src.data.ETL import ETL
from src.features.Feature_Engineer import FeatureEngineer
from src.models.Train import Trainer
import pandas as pd

def main():
    # 1. ETL
    etl = ETL("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "data/processed/clean_telco.csv")
    df = etl.run()

    # 2. Feature Engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    df.to_csv("data/processed/features_telco.csv", index=False)

    # 3. Entrenamiento con Optuna + MLflow
    trainer = Trainer("data/processed/features_telco.csv")
    trainer.run()

if __name__ == "__main__":
    main()
