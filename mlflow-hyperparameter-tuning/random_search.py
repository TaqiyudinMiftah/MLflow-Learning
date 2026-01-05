"""
Random Search Hyperparameter Tuning dengan MLflow

Konsep yang dipelajari:
1. Random sampling dari parameter space
2. Probability distributions untuk continuous & discrete parameters
3. Lebih efisien dari grid search untuk large parameter spaces
4. Trade-off antara exploration dan computation time
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("hyperparameter-tuning-randomsearch")


def sample_parameters(n_samples=30):
    """
    Generate random parameter samples
    
    Berbeda dengan grid search yang test semua kombinasi,
    random search sample secara random dari distribution
    """
    param_samples = []
    
    for _ in range(n_samples):
        params = {
            # Integer uniform distribution
            'n_estimators': np.random.choice([50, 100, 150, 200, 250, 300]),
            
            # Dapat None atau integer
            'max_depth': np.random.choice([5, 10, 15, 20, 25, None]),
            
            # Integer uniform
            'min_samples_split': np.random.randint(2, 11),
            
            # Integer uniform
            'min_samples_leaf': np.random.randint(1, 5),
            
            # Discrete choice
            'max_features': np.random.choice(['sqrt', 'log2', None]),
            
            # Boolean
            'bootstrap': np.random.choice([True, False]),
        }
        param_samples.append(params)
    
    return param_samples


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    """
    Train model dan return metrics
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
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()
    
    return model, metrics


def random_search(n_iter=30):
    """
    Random Search dengan MLflow nested runs
    
    Args:
        n_iter: Jumlah iterasi (random samples) yang akan di-test
    """
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Generate random parameter samples
    param_samples = sample_parameters(n_samples=n_iter)
    
    print(f"🎲 Random Search dengan {n_iter} iterations")
    print("="*60)
    
    # Parent run
    with mlflow.start_run(run_name="random_search_parent") as parent_run:
        # Log parent metadata
        mlflow.set_tag("method", "random_search")
        mlflow.set_tag("n_iterations", n_iter)
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("test_size", 0.2)
        
        best_score = 0
        best_params = None
        best_model = None
        all_scores = []
        
        # Test each random sample
        for idx, params in enumerate(param_samples, 1):
            with mlflow.start_run(
                run_name=f"random_run_{idx:03d}",
                nested=True
            ) as child_run:
                print(f"\n[{idx}/{n_iter}] Testing parameters:")
                for key, val in params.items():
                    print(f"   {key}: {val}")
                
                # Log parameters
                mlflow.log_params(params)
                
                # Train and evaluate
                try:
                    model, metrics = train_and_evaluate(
                        X_train, X_test, y_train, y_test, params
                    )
                    
                    # Log metrics
                    mlflow.log_metrics(metrics)
                    
                    all_scores.append(metrics['accuracy'])
                    
                    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
                    print(f"   ✅ CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
                    
                    # Track best
                    if metrics['accuracy'] > best_score:
                        best_score = metrics['accuracy']
                        best_params = params
                        best_model = model
                        mlflow.set_tag("is_best", "true")
                        print(f"   🏆 New best score!")
                    else:
                        mlflow.set_tag("is_best", "false")
                        
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
        
        # Calculate statistics
        scores_array = np.array(all_scores)
        
        # Log aggregated metrics ke parent run
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.log_metric("mean_accuracy", scores_array.mean())
        mlflow.log_metric("std_accuracy", scores_array.std())
        mlflow.log_metric("min_accuracy", scores_array.min())
        mlflow.log_metric("max_accuracy", scores_array.max())
        
        mlflow.log_param("best_params", str(best_params))
        
        # Save best model
        mlflow.sklearn.log_model(
            best_model,
            "best_model",
            input_example=X_test[:5],
            signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train))
        )
        
        # Summary
        print("\n" + "="*60)
        print("🎯 RANDOM SEARCH COMPLETED")
        print("="*60)
        print(f"Best Accuracy: {best_score:.4f}")
        print(f"Mean Accuracy: {scores_array.mean():.4f} (+/- {scores_array.std():.4f})")
        print(f"Range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
        print(f"\nBest Parameters:")
        for key, val in best_params.items():
            print(f"  {key}: {val}")
        print(f"\nParent Run ID: {parent_run.info.run_id}")
        
        print("\n💡 Tips:")
        print("  - Random search lebih efisien dari grid search")
        print("  - Cocok untuk large parameter spaces")
        print("  - Increase n_iter jika parameter space sangat besar")
        print("\n📊 Visualize di MLflow UI:")
        print("  mlflow ui --backend-store-uri file:./mlruns")


if __name__ == "__main__":
    # Test dengan 30 random samples
    random_search(n_iter=30)
