import mlflow

MODEL_NAME = "iris_classifier"
MODEL_VERSION = 1

client = mlflow.MlflowClient()

# Set alias "champion" ke version 1
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=str(MODEL_VERSION)
)

print(f"Alias 'champion' set to {MODEL_NAME} v{MODEL_VERSION}")
