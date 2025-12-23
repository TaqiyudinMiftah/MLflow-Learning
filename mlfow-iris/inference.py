import mlflow.pyfunc
import pandas as pd

MODEL_URI = "runs:/3b923996e4f7448fa7e542076e40bea9/model"

model = mlflow.pyfunc.load_model(MODEL_URI)

input_data = pd.DataFrame([
    [6.7, 3.1, 5.6, 2.4]
])

prediction = model.predict(input_data)
print("Hasil prediksi:", prediction)
