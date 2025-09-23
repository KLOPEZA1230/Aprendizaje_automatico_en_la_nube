import pandas as pd
from pathlib import Path

print(">>> Hola, estoy dentro de ETL.py")


class ETL:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def extract(self) -> pd.DataFrame:
        print("📥 [EXTRACT] Leyendo dataset...")
        print(f"   → Ruta: {self.input_path.resolve()}")
        df = pd.read_csv(self.input_path)
        print(f"   ✅ Datos cargados con {df.shape[0]} filas y {df.shape[1]} columnas")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("🔄 [TRANSFORM] Aplicando transformaciones...")
        df = df.dropna()  # ejemplo simple
        print(f"   ✅ Después de limpiar: {df.shape[0]} filas")
        return df

    def load(self, df: pd.DataFrame) -> pd.DataFrame:
        print("💾 [LOAD] Guardando dataset transformado...")
        print(f"   → Ruta destino: {self.output_path.resolve()}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print("   ✅ Guardado completo")
        return df

    def run(self) -> pd.DataFrame:
        print("🚀 Iniciando pipeline ETL...")
        df = self.extract()
        df = self.transform(df)
        df = self.load(df)
        print("🎉 ETL finalizado con éxito")
        return df


if __name__ == "__main__":
    print(">>> Entré al bloque main de ETL.py")
    etl = ETL("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
              "data/processed/clean_telco.csv")
    df = etl.run()
    print(">>> Terminé ETL, shape final:", df.shape)

