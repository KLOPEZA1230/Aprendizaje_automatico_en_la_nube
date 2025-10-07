from src.models.train_flow import training_flow

if __name__ == "__main__":
    # Servir el flow localmente con un schedule (ej.: todos los días a las 09:00)
    training_flow.serve(
        name="daily_training",
        cron="0 9 * * *",        # CRON: 09:00 todos los días
        pause_on_shutdown=False, # deja el served deployment activo mientras el proceso viva
        tags=["mlops", "churn"],
        description="Entrenamiento diario del modelo de churn (local, Prefect 3 serve)",
    )

