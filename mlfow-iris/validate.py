from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input

# GANTI dengan run_id kamu
MODEL_URI = "runs:/3b923996e4f7448fa7e542076e40bea9/model"

INPUT_EXAMPLE = [
    [6.6, 3.0, 4.4, 1.4],
    [6.9, 3.1, 4.9, 1.5]
]

serving_payload = convert_input_example_to_serving_input(INPUT_EXAMPLE)

print("Serving payload:")
print(serving_payload)

print("\nValidasi:")
print(validate_serving_input(MODEL_URI, serving_payload))
