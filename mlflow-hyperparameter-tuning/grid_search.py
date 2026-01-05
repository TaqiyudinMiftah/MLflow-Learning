"""
Grid Search Hyperparameter Tuning dengan MLflow

Konsep yang dipelajari:
1. Parent-child runs untuk mengorganisir tuning experiments
2. Nested runs dengan context manager
3. Logging parameters dan metrics untuk setiap combination
4. Finding best hyperparameters berdasarkan metrics
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import itertools
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("hyperparameter-tuning-gridsearch")


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    """
    Train model dengan parameter tertentu dan return metrics
    """
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()
    
    return model, metrics


def grid_search_manual():
    """
    Manual Grid Search dengan MLflow nested runs
    """
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(param_combinations)}")
    
    # Parent run untuk mengelompokkan semua child runs
    with mlflow.start_run(run_name="grid_search_parent") as parent_run:
        # Log parent run metadata
        mlflow.set_tag("method", "grid_search")
        mlflow.set_tag("total_combinations", len(param_combinations))
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("test_size", 0.2)
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Iterate through all combinations
        for idx, params in enumerate(param_combinations, 1):
            # Child run untuk setiap kombinasi parameter
            with mlflow.start_run(
                run_name=f"run_{idx:03d}",
                nested=True  # Penting: tandai sebagai nested run
            ) as child_run:
                print(f"\n[{idx}/{len(param_combinations)}] Testing: {params}")
                
                # Log parameters
                mlflow.log_params(params)
                
                # Train and evaluate
                model, metrics = train_and_evaluate(
                    X_train, X_test, y_train, y_test, params
                )
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model artifact (optional - hanya untuk best model)
                # mlflow.sklearn.log_model(model, "model")
                
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
                
                # Track best model
                if metrics['accuracy'] > best_score:
                    best_score = metrics['accuracy']
                    best_params = params
                    best_model = model
                    
                    # Tag sebagai current best
                    mlflow.set_tag("is_best", "true")
                else:
                    mlflow.set_tag("is_best", "false")
        
        # Log best results ke parent run
        mlflow.log_param("best_params", str(best_params))
        mlflow.log_metric("best_accuracy", best_score)
        
        # Save best model
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            input_example=X_test[:5],
            signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train))
        )
        
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETED")
        print("="*60)
        print(f"Best Accuracy: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"Parent Run ID: {parent_run.info.run_id}")
        print("\nTip: Gunakan MLflow UI untuk compare semua runs:")
        print("   mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    grid_search_manual()
