# FastAPI Serving untuk Iris Classifier

Dokumentasi lengkap untuk deploy dan menggunakan model MLflow Iris Classifier dengan FastAPI.

## Ì≥ã Daftar Isi

- [Instalasi](#instalasi)
- [Menjalankan Server](#menjalankan-server)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Contoh Penggunaan](#contoh-penggunaan)

## Ì∫Ä Instalasi

1. Pastikan semua dependencies sudah terinstall:

```bash
uv sync
```

## ÌøÉ Menjalankan Server

### Metode 1: Langsung dengan Python

```bash
uv run python fastapi_serve.py
```

### Metode 2: Dengan Uvicorn (untuk development)

```bash
uv run uvicorn fastapi_serve:app --reload --host 0.0.0.0 --port 8000
```

### Metode 3: Dengan Uvicorn (untuk production)

```bash
uv run uvicorn fastapi_serve:app --host 0.0.0.0 --port 8000 --workers 4
```

Server akan berjalan di: `http://localhost:8000`

## Ì≥ö API Endpoints

### 1. Health Check

**GET** `/`
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Iris Classifier API is running",
  "model_uri": "models:/iris_classifier@champion",
  "status": "healthy"
}
```

### 2. Health Status

**GET** `/health`
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_uri": "models:/iris_classifier@champion"
}
```

### 3. Model Info

**GET** `/model/info`
```bash
curl http://localhost:8000/model/info
```

Response:
```json
{
  "model_uri": "models:/iris_classifier@champion",
  "iris_labels": {
    "0": "setosa",
    "1": "versicolor",
    "2": "virginica"
  },
  "expected_features": [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
  ],
  "feature_order": "Sesuai urutan di atas"
}
```

### 4. Single Prediction

**POST** `/predict`

Request body:
```json
{
  "sepal_length": 6.7,
  "sepal_width": 3.1,
  "petal_length": 5.6,
  "petal_width": 2.4
}
```

Contoh dengan curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 6.7,
    "sepal_width": 3.1,
    "petal_length": 5.6,
    "petal_width": 2.4
  }'
```

Response:
```json
{
  "prediction": 2,
  "prediction_label": "virginica"
}
```

### 5. Batch Prediction

**POST** `/predict/batch`

Request body:
```json
{
  "instances": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
    {
      "sepal_length": 6.7,
      "sepal_width": 3.1,
      "petal_length": 5.6,
      "petal_width": 2.4
    }
  ]
}
```

Contoh dengan curl:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      },
      {
        "sepal_length": 6.7,
        "sepal_width": 3.1,
        "petal_length": 5.6,
        "petal_width": 2.4
      }
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "setosa"
    },
    {
      "prediction": 2,
      "prediction_label": "virginica"
    }
  ]
}
```

## Ì∑™ Testing

Jalankan script test yang sudah disediakan:

```bash
uv run python test_api.py
```

Script ini akan mengetes semua endpoints dan menampilkan hasilnya.

## Ì≥ñ Interactive API Documentation

FastAPI secara otomatis generate dokumentasi interaktif:

### Swagger UI
Buka di browser: `http://localhost:8000/docs`

Di sini Anda bisa:
- Melihat semua endpoints
- Test endpoints langsung dari browser
- Melihat schema request/response

### ReDoc
Buka di browser: `http://localhost:8000/redoc`

Dokumentasi alternatif dengan tampilan yang lebih clean.

## Ì≤° Contoh Penggunaan

### Python dengan requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "sepal_length": 6.7,
        "sepal_width": 3.1,
        "petal_length": 5.6,
        "petal_width": 2.4
    }
)
print(response.json())
# Output: {"prediction": 2, "prediction_label": "virginica"}

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "instances": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.7,
                "sepal_width": 3.1,
                "petal_length": 5.6,
                "petal_width": 2.4
            }
        ]
    }
)
print(response.json())
```

### JavaScript dengan fetch

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    sepal_length: 6.7,
    sepal_width: 3.1,
    petal_length: 5.6,
    petal_width: 2.4
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Ì¥ß Konfigurasi

Anda bisa mengubah konfigurasi di file `fastapi_serve.py`:

- **MODEL_URI**: URI model yang ingin digunakan (default: `models:/iris_classifier@champion`)
- **Host**: IP address server (default: `0.0.0.0`)
- **Port**: Port server (default: `8000`)

## Ì∞≥ Docker (Opsional)

Untuk deploy dengan Docker, buat `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install uv && uv sync

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "fastapi_serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build dan run:
```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```

## ÔøΩÔøΩ Label Mapping

- **0**: Iris Setosa
- **1**: Iris Versicolor
- **2**: Iris Virginica

## ‚ö†Ô∏è Troubleshooting

### Model tidak ditemukan
Pastikan model sudah terdaftar di MLflow:
```bash
uv run python register_model.py
```

### Port sudah digunakan
Ubah port di command atau di `fastapi_serve.py`:
```bash
uv run uvicorn fastapi_serve:app --port 8001
```

### Import error
Pastikan semua dependencies terinstall:
```bash
uv sync
```

## ÌæØ Tips Production

1. **Gunakan multiple workers** untuk handle concurrent requests:
   ```bash
   uv run uvicorn fastapi_serve:app --workers 4
   ```

2. **Add logging** untuk monitoring

3. **Implement rate limiting** untuk prevent abuse

4. **Add authentication** jika diperlukan

5. **Use reverse proxy** seperti Nginx di depan uvicorn

6. **Monitor dengan tools** seperti Prometheus + Grafana

## Ì≥Ñ License

MIT
