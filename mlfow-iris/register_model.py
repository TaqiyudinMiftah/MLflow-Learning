import mlflow

mlflow.set_tracking_uri("file:./mlruns")

RUN_ID = "e8b4add6a04248f49dd6f2bd19f4600a"
MODEL_NAME = "iris_classifier"

model_uri = f"runs:/{RUN_ID}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("Model registered!")
print("Name:", result.name)
print("Version:", result.version)
