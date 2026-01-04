import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import uvicorn

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# Model configuration
MODEL_URI = "models:/iris_classifier@champion"

# Initialize FastAPI app
app = FastAPI(
    title="Iris Classifier API",
    description="API untuk prediksi klasifikasi Iris menggunakan model MLflow",
    version="1.0.0"
)

# Load model saat startup
model = None

@app.on_event("startup")
async def load_model():
    """Load MLflow model saat aplikasi dimulai"""
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"✅ Model berhasil dimuat dari {MODEL_URI}")
    except Exception as e:
        print(f"❌ Error memuat model: {str(e)}")
        raise

# Pydantic models untuk request/response
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Panjang sepal (cm)", example=5.1)
    sepal_width: float = Field(..., description="Lebar sepal (cm)", example=3.5)
    petal_length: float = Field(..., description="Panjang petal (cm)", example=1.4)
    petal_width: float = Field(..., description="Lebar petal (cm)", example=0.2)

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 6.7,
                "sepal_width": 3.1,
                "petal_length": 5.6,
                "petal_width": 2.4
            }
        }

class BatchIrisFeatures(BaseModel):
    instances: List[IrisFeatures] = Field(..., description="List dari features untuk batch prediction")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Hasil prediksi (0: setosa, 1: versicolor, 2: virginica)")
    prediction_label: str = Field(..., description="Label nama species")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List dari hasil prediksi")

# Mapping label
IRIS_LABELS = {
    0: "setosa",
    1: "versicolor", 
    2: "virginica"
}

# Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Endpoint root untuk health check"""
    return {
        "message": "Iris Classifier API is running",
        "model_uri": MODEL_URI,
        "status": "healthy"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_uri": MODEL_URI
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """
    Prediksi single instance dari iris flower
    
    - **sepal_length**: Panjang sepal dalam cm
    - **sepal_width**: Lebar sepal dalam cm
    - **petal_length**: Panjang petal dalam cm
    - **petal_width**: Lebar petal dalam cm
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    try:
        # Konversi ke DataFrame
        input_data = pd.DataFrame([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Prediksi
        prediction = model.predict(input_data)
        pred_value = int(prediction[0])
        
        return PredictionResponse(
            prediction=pred_value,
            prediction_label=IRIS_LABELS.get(pred_value, "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat prediksi: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchIrisFeatures):
    """
    Prediksi batch/multiple instances dari iris flowers
    
    Kirim list dari features untuk mendapatkan prediksi batch
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    try:
        # Konversi ke DataFrame
        input_data = pd.DataFrame([
            [f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]
            for f in batch.instances
        ])
        
        # Prediksi
        predictions = model.predict(input_data)
        
        # Format response
        results = [
            PredictionResponse(
                prediction=int(pred),
                prediction_label=IRIS_LABELS.get(int(pred), "unknown")
            )
            for pred in predictions
        ]
        
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat prediksi batch: {str(e)}")

@app.get("/model/info", tags=["Model Info"])
async def model_info():
    """Informasi tentang model yang sedang digunakan"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat")
    
    return {
        "model_uri": MODEL_URI,
        "iris_labels": IRIS_LABELS,
        "expected_features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ],
        "feature_order": "Sesuai urutan di atas"
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "fastapi_serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
