"""
MLflow Autologging Demo

Konsep yang dipelajari:
1. Automatic logging parameters, metrics, dan models
2. Framework-specific autologging (sklearn, tensorflow, pytorch, etc)
3. Comparison: Manual logging vs Autologging
4. Customization dan best practices
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("autologging-demo")


def manual_logging_example():
    """
    Example 1: Manual Logging (Traditional Way)
    
    Kita manually log semua parameters dan metrics
    """
    print("\n" + "="*60)
    print("📝 Example 1: MANUAL LOGGING")
    print("="*60)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="manual_logging_rf"):
        # Manually log parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Manually log metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Manually log model
        mlflow.sklearn.log_model(model, "model")
        
        # Manual tags
        mlflow.set_tag("method", "manual")
        
        print(f"✅ Manual logging completed")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Logged: params, metrics, model manually")


def autologging_example():
    """
    Example 2: Autologging (Automatic Way)
    
    MLflow automatically logs parameters, metrics, dan model!
    """
    print("\n" + "="*60)
    print("🤖 Example 2: AUTOLOGGING")
    print("="*60)
    
    # Enable autologging untuk sklearn
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        log_datasets=True,
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False
    )
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="autolog_rf"):
        # Hanya train model - MLflow will log everything automatically!
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # MLflow automatically logs:
        # - All model parameters
        # - Training score
        # - Model artifact dengan signature
        # - Input example
        # - Dan banyak lagi!
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Kita masih bisa add custom metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.set_tag("method", "autolog")
        
        print(f"✅ Autologging completed")
        print(f"   Test Accuracy: {accuracy:.4f}")
        print(f"   Logged: params, metrics, model AUTOMATICALLY!")
    
    # Disable autologging setelah selesai
    mlflow.sklearn.autolog(disable=True)


def autolog_with_gridsearch():
    """
    Example 3: Autologging dengan GridSearchCV
    
    MLflow automatically logs all trials dari GridSearch!
    """
    print("\n" + "="*60)
    print("🔍 Example 3: AUTOLOGGING + GRIDSEARCHCV")
    print("="*60)
    
    # Enable autologging
    mlflow.sklearn.autolog(log_models=True)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid (smaller untuk demo)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
    }
    
    with mlflow.start_run(run_name="autolog_gridsearch_parent"):
        # GridSearchCV
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit - MLflow will automatically create nested runs untuk setiap combination!
        grid_search.fit(X_train, y_train)
        
        # Best results automatically logged
        print(f"✅ GridSearch completed with autologging")
        print(f"   Best Score: {grid_search.best_score_:.4f}")
        print(f"   Best Params: {grid_search.best_params_}")
        print(f"   Total combinations: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])}")
        print(f"   All runs automatically logged as nested runs!")
    
    mlflow.sklearn.autolog(disable=True)


def autolog_multiple_models():
    """
    Example 4: Autologging untuk berbagai model types
    """
    print("\n" + "="*60)
    print("🎯 Example 4: AUTOLOGGING MULTIPLE MODELS")
    print("="*60)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"autolog_{model_name.lower()}"):
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy
            
            # Custom metric
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.set_tag("model_type", model_name)
            
            print(f"  ✅ {model_name}: {accuracy:.4f}")
    
    # Print comparison
    print("\n📊 Model Comparison:")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name}: {acc:.4f}")
    
    mlflow.sklearn.autolog(disable=True)


def autolog_best_practices():
    """
    Example 5: Best Practices untuk Autologging
    """
    print("\n" + "="*60)
    print("💡 Example 5: AUTOLOGGING BEST PRACTICES")
    print("="*60)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Configure autologging dengan options
    mlflow.sklearn.autolog(
        log_input_examples=True,      # Log sample input untuk inference
        log_model_signatures=True,    # Log model signature untuk validation
        log_models=True,               # Log model artifacts
        log_datasets=True,             # Log dataset information
        silent=False                   # Show autologging info
    )
    
    with mlflow.start_run(run_name="autolog_best_practices"):
        # 1. Train model (autologged)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 2. Add custom metrics yang tidak auto-logged
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("custom_test_accuracy", test_acc)
        
        # 3. Add meaningful tags
        mlflow.set_tag("stage", "development")
        mlflow.set_tag("data_version", "v1.0")
        mlflow.set_tag("description", "Best practices demo")
        
        # 4. Log additional artifacts
        report = classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # 5. Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        print("✅ Best practices applied:")
        print("   - Autologging enabled dengan full options")
        print("   - Custom metrics added")
        print("   - Meaningful tags set")
        print("   - Additional artifacts logged")
        print("   - Dataset info tracked")
    
    mlflow.sklearn.autolog(disable=True)


def main():
    """
    Run semua examples
    """
    print("\n" + "="*70)
    print(" 🚀 MLFLOW AUTOLOGGING COMPREHENSIVE DEMO")
    print("="*70)
    
    # Run all examples
    manual_logging_example()
    autologging_example()
    autolog_with_gridsearch()
    autolog_multiple_models()
    autolog_best_practices()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("  1. Autologging drastically reduces boilerplate code")
    print("  2. Works automatically with GridSearchCV, RandomizedSearchCV")
    print("  3. Creates nested runs untuk hyperparameter tuning")
    print("  4. Still allows custom metrics and tags")
    print("  5. Logs model signatures and input examples automatically")
    
    print("\n📊 View all results di MLflow UI:")
    print("  mlflow ui --backend-store-uri file:./mlruns")
    
    print("\n🎯 Next Steps:")
    print("  - Explore autologging untuk other frameworks (PyTorch, TensorFlow)")
    print("  - Learn when to use autologging vs manual logging")
    print("  - Combine autologging dengan custom logging untuk flexibility")


if __name__ == "__main__":
    main()
