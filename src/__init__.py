import mlflow

print("🔍 Tracking URI actual:", mlflow.get_tracking_uri())
print("🔍 Artifact URI (dentro del run):")

with mlflow.start_run():
    print("➡️", mlflow.get_artifact_uri())
