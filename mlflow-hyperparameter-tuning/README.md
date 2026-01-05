# MLflow Hyperparameter Tuning

Folder ini berisi pembelajaran tentang hyperparameter tuning dengan MLflow, mencakup berbagai metode dan best practices.

## 📚 Materi yang Dipelajari

### 1. Basic Concepts
- **Parent-Child Runs**: Cara mengorganisir multiple experiments
- **Nested Runs**: Tracking hyperparameter tuning experiments
- **Run Comparison**: Membandingkan hasil dari berbagai hyperparameter

### 2. Metode Hyperparameter Tuning

#### Grid Search (`grid_search.py`)
- Exhaustive search pada semua kombinasi parameter
- Menggunakan nested runs untuk tracking
- Visualisasi hasil tuning

#### Random Search (`random_search.py`)
- Random sampling dari parameter space
- Lebih efisien untuk large parameter spaces
- Probability distributions untuk sampling

#### Bayesian Optimization (`optuna_tuning.py`)
- Menggunakan Optuna untuk intelligent search
- MLflow callback untuk automatic logging
- Visualization dengan Optuna plots

### 3. Advanced Features

#### Autologging (`autologging_demo.py`)
- Automatic parameter & metrics logging
- Framework-specific autologging
- Custom logging vs autologging

#### Comparison & Analysis (`compare_runs.py`)
- Search & filter runs berdasarkan metrics
- Parallel coordinates plot
- Best model selection

## 🚀 Quick Start

### Install Dependencies
```bash
pip install mlflow scikit-learn optuna pandas numpy matplotlib seaborn
```

### Jalankan Experiments

1. **Grid Search**:
```bash
python grid_search.py
```

2. **Random Search**:
```bash
python random_search.py
```

3. **Bayesian Optimization dengan Optuna**:
```bash
python optuna_tuning.py
```

4. **Autologging Demo**:
```bash
python autologging_demo.py
```

5. **Compare & Analyze Results**:
```bash
python compare_runs.py
```

## 📊 MLflow UI

Lihat hasil experiments:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

Buka browser: http://localhost:5000

## 🎯 Learning Path

1. ✅ Mulai dengan `grid_search.py` - Pahami konsep parent-child runs
2. ✅ Lanjut ke `random_search.py` - Lebih efisien dari grid search
3. ✅ Coba `optuna_tuning.py` - Bayesian optimization yang powerful
4. ✅ Eksplorasi `autologging_demo.py` - Simplify logging process
5. ✅ Analisis dengan `compare_runs.py` - Find best hyperparameters

## 📝 Key Concepts

### Parent-Child Runs Structure
```
Parent Run (Hyperparameter Tuning)
├── Child Run 1 (params: n_estimators=50, max_depth=5)
├── Child Run 2 (params: n_estimators=100, max_depth=10)
└── Child Run 3 (params: n_estimators=200, max_depth=15)
```

### Best Practices
- ✅ Gunakan parent run untuk grup related experiments
- ✅ Log semua hyperparameters sebagai params
- ✅ Log evaluation metrics untuk comparison
- ✅ Simpan best model dengan signature
- ✅ Tag runs dengan metadata yang meaningful
- ✅ Gunakan experiment naming yang konsisten

## 🔍 What's Next?

Setelah menguasai hyperparameter tuning:
- **MLflow Projects**: Reproducible ML workflows
- **Custom Models**: mlflow.pyfunc untuk custom inference logic
- **Model Evaluation**: mlflow.evaluate() untuk comprehensive assessment
- **Production Deployment**: Docker, Kubernetes, cloud platforms

## 📚 Resources

- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [Optuna Integration](https://optuna.readthedocs.io/)
