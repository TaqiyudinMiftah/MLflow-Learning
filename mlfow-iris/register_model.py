import mlflow

RUN_ID = "3b923996e4f7448fa7e542076e40bea9"
MODEL_NAME = "iris_classifier"

model_uri = f"runs:/{RUN_ID}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered!")
print("Name:", result.name)
print("Version:", result.version)
