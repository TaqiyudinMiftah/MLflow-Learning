# Quick Reference Guide

## 🚀 Instalasi Dependencies

```bash
cd mlflow-hyperparameter-tuning
pip install -e .
```

Atau manual:
```bash
pip install mlflow scikit-learn optuna pandas numpy matplotlib seaborn plotly
```

## 📝 Cheat Sheet

### 1. Parent-Child Runs
```python
with mlflow.start_run(run_name="parent") as parent:
    # Parent run code
    
    with mlflow.start_run(run_name="child", nested=True):
        # Child run code
        pass
```

### 2. Autologging
```python
# Enable autologging
mlflow.sklearn.autolog()

# Train model - everything logged automatically!
model.fit(X_train, y_train)

# Disable autologging
mlflow.sklearn.autolog(disable=True)
```

### 3. Search Runs
```python
# Search by accuracy
runs = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string="metrics.accuracy > 0.95",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Get best run
best_run = runs.iloc[0]
```

### 4. Optuna dengan MLflow
```python
from optuna.integration.mlflow import MLflowCallback

mlflc = MLflowCallback(
    tracking_uri="file:./mlruns",
    metric_name="accuracy",
    create_experiment=False,
    mlflow_kwargs={"nested": True}
)

study.optimize(objective, n_trials=50, callbacks=[mlflc])
```

### 5. Load Best Model
```python
# From run_id
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# From model registry
model = mlflow.pyfunc.load_model("models:/model_name/version")
```

## 🎯 Common MLflow Commands

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000

# View specific experiment
mlflow ui --backend-store-uri file:./mlruns --experiment-id 1

# Serve model
mlflow models serve -m runs:/{run_id}/model -p 5001

# Clean up old runs (be careful!)
# rm -rf mlruns/
```

## 📊 Metrics to Track

### Classification
- `accuracy`
- `precision`
- `recall`
- `f1_score`
- `roc_auc`
- `confusion_matrix`

### Regression
- `mse` (Mean Squared Error)
- `rmse` (Root Mean Squared Error)
- `mae` (Mean Absolute Error)
- `r2_score`

### Cross-Validation
- `cv_mean`
- `cv_std`
- `cv_scores`

## 🏷️ Useful Tags

```python
mlflow.set_tag("stage", "development")
mlflow.set_tag("model_type", "RandomForest")
mlflow.set_tag("is_best", "true")
mlflow.set_tag("data_version", "v1.0")
mlflow.set_tag("description", "Baseline model")
```

## 🔍 Filtering Runs

```python
# Filter by metric
filter_string = "metrics.accuracy > 0.95"

# Filter by parameter
filter_string = "params.n_estimators = '100'"

# Filter by tag
filter_string = "tags.is_best = 'true'"

# Combine filters
filter_string = "metrics.accuracy > 0.95 and params.n_estimators = '100'"

# Parent runs only
filter_string = "tags.mlflow.parentRunId = ''"

# Child runs only
filter_string = "tags.mlflow.parentRunId != ''"
```

## 💡 Best Practices

1. **Always use experiments** - Group related runs
2. **Use nested runs** - For hyperparameter tuning
3. **Tag meaningfully** - Easy to filter later
4. **Log input examples** - For model signatures
5. **Version your data** - Track data versions
6. **Clean up regularly** - Delete unsuccessful runs
7. **Use autologging** - Less boilerplate code
8. **Save best models** - With clear naming

## 🐛 Troubleshooting

### Error: Port already in use
```bash
# Kill process using port 5000
lsof -ti:5000 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :5000   # Windows (find PID then taskkill)
```

### Error: Run already active
```python
# End active run before starting new one
if mlflow.active_run():
    mlflow.end_run()
```

### Error: Model not found
```python
# Check if model exists in run
client = MlflowClient()
artifacts = client.list_artifacts(run_id)
print([a.path for a in artifacts])
```

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Sklearn Documentation](https://scikit-learn.org/)

## 🎓 Learning Path

1. ✅ Basic tracking (mlflow-iris)
2. ✅ Hyperparameter tuning (current folder)
3. ⏭️ MLflow Projects
4. ⏭️ Custom Models (mlflow.pyfunc)
5. ⏭️ Model Evaluation
6. ⏭️ Production Deployment
