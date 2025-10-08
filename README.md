🧠 Proyecto Final MLOps – Universidad de Medellín
Análisis de Churn en Clientes de Telecomunicaciones
📌 Descripción general

Este proyecto se desarrolla en el marco de un análisis de negocio para una empresa de telecomunicaciones, con el objetivo de comprender los factores que impactan en la rotación de clientes (churn) y construir un pipeline completo de Machine Learning end-to-end que integre prácticas de MLOps.

🎯 Objetivos

Analizar en detalle la base de datos de clientes y detectar los principales factores asociados al churn.

Implementar un flujo MLOps completo:

Ingesta y preprocesamiento de datos.

Entrenamiento y registro de experimentos con MLflow.

Orquestación de tareas con Prefect.

Despliegue del modelo vía API REST (FastAPI).

Monitoreo del rendimiento del modelo.

Mejores prácticas: versionado, reproducibilidad y CI/CD.

🧩 Dataset

Nombre: Telco Customer Churn

Fuente: Telco Customer Churn – Kaggle

Variable objetivo: Churn (1 = cliente se dio de baja, 0 = cliente activo).

Procesamiento: variables codificadas y transformadas → data/processed/features_telco.csv.

⚙️ Tecnologías
Componente	Herramienta
Lenguaje	Python 3.11
Frameworks ML	Scikit-learn, Pandas, NumPy
Tracking	MLflow
Orquestación	Prefect
API	FastAPI + Uvicorn
Contenedores	Docker
Testing y calidad	Pytest, Flake8, Pre-commit
📁 Estructura del Repositorio
Aprendizaje_automatico_en_la_nube/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── api/app.py
│   ├── models/train_flow.py
│   ├── monitoring/monitor.py
│   └── features/Feature_Engineer.py
├── logs/
│   └── mlruns/
├── tests/
│   └── test_train.py
└── Dockerfile

🧰 Instalación y Configuración
# 1️⃣ Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2️⃣ Instalar dependencias
pip install -r requirements.txt

🚀 Entrenamiento y MLflow Tracking
python -m src.models.train_flow


Entrena un modelo RandomForestClassifier.

Registra parámetros, métricas y modelo en logs/mlruns/.

Crea artifacts/model/MLmodel para su posterior despliegue.

Visualización de resultados:

mlflow ui --backend-store-uri logs/mlruns --port 5000


Abrir en navegador: http://127.0.0.1:5000

🌐 Despliegue Local – API FastAPI
uvicorn src.api.app:app --reload


Documentación interactiva: http://127.0.0.1:8000/docs

Verificación de estado: http://127.0.0.1:8000/

Ejemplo de request:

{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.2
}


Respuesta:

{"prediction": 0}

📊 Monitoreo Simulado
python -m src.monitoring.monitor


Genera un log con métricas (accuracy, positive_rate) en logs/monitoring/metrics.csv.

🧪 Testing y Calidad
pytest -q
flake8
pre-commit run --all-files

🐳 Docker (opcional)
docker build -t churn-api .
docker run -p 8000:8000 churn-api

✅ Estado del Proyecto
Fase	Entregable	Estado
1	Setup y EDA	✅
2	MLflow Tracking	✅
3	Prefect Pipeline	✅
4	API REST (FastAPI)	✅
5	Monitoreo	✅
6	Testing y Documentación	🔵 En revisión final
7	Docker y CI/CD	🔵 En progreso
👤 Autores

Katerine López Arango – Ciencia de Datos & MLOps
Marcelo David – Inteligencia Artificial y Analítica Predictiva

GitHub: @KLOPEZA1230