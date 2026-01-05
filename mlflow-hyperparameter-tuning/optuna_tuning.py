"""
Bayesian Optimization dengan Optuna + MLflow

Konsep yang dipelajari:
1. Bayesian optimization untuk intelligent hyperparameter search
2. Optuna integration dengan MLflow
3. MLflowCallback untuk automatic logging
4. Visualization dengan Optuna plots
5. Pruning unpromising trials
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna.integration.mlflow import MLflowCallback

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("hyperparameter-tuning-optuna")


def objective(trial, X_train, X_test, y_train, y_test):
    """
    Optuna objective function
    
    Optuna akan call function ini berkali-kali dengan different parameters
    untuk find optimal hyperparameters
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }
    
    # Train model
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    
    # Use cross-validation untuk more robust estimate
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Train final model untuk test evaluation
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Log additional metrics (MLflowCallback akan log params otomatis)
    # Kita perlu manually log metrics yang kita inginkan
    trial.set_user_attr('test_accuracy', test_accuracy)
    trial.set_user_attr('test_f1', test_f1)
    trial.set_user_attr('cv_std', cv_std)
    
    # Return metric to optimize (Optuna akan maximize ini)
    return cv_mean


def optuna_bayesian_optimization(n_trials=50):
    """
    Run Bayesian Optimization menggunakan Optuna dengan MLflow integration
    
    Args:
        n_trials: Jumlah trials (combinations) yang akan ditest
    """
    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🧪 Optuna Bayesian Optimization dengan {n_trials} trials")
    print("="*60)
    
    # Parent MLflow run
    with mlflow.start_run(run_name="optuna_bayesian_parent") as parent_run:
        mlflow.set_tag("method", "bayesian_optimization")
        mlflow.set_tag("optimizer", "optuna")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("dataset", "iris")
        
        # MLflow callback untuk automatic logging
        mlflc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_accuracy",
            create_experiment=False,
            mlflow_kwargs={
                "nested": True,  # Create nested runs
            }
        )
        
        # Create Optuna study
        study = optuna.create_study(
            study_name="iris_rf_optimization",
            direction="maximize",  # Maximize accuracy
            sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian optimization
            pruner=optuna.pruners.MedianPruner()  # Prune unpromising trials
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
            callbacks=[mlflc],
            show_progress_bar=True
        )
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Log best results to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_accuracy", best_score)
        mlflow.log_metric("best_test_accuracy", best_trial.user_attrs['test_accuracy'])
        mlflow.log_metric("best_test_f1", best_trial.user_attrs['test_f1'])
        
        # Train final model dengan best parameters
        print("\n🔨 Training final model dengan best parameters...")
        final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        final_model.fit(X_train, y_train)
        
        # Save best model
        mlflow.sklearn.log_model(
            final_model,
            "best_model",
            input_example=X_test[:5],
            signature=mlflow.models.infer_signature(X_train, final_model.predict(X_train))
        )
        
        # Log optimization history sebagai artifact
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization as vis
            
            # Optimization history plot
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html("optimization_history.html")
            mlflow.log_artifact("optimization_history.html")
            
            # Parameter importance plot
            fig2 = vis.plot_param_importances(study)
            fig2.write_html("param_importances.html")
            mlflow.log_artifact("param_importances.html")
            
            # Parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html("parallel_coordinate.html")
            mlflow.log_artifact("parallel_coordinate.html")
            
            print("📊 Visualization plots saved sebagai artifacts")
            
        except ImportError:
            print("⚠️  Install plotly untuk visualizations: pip install plotly")
        
        # Print summary
        print("\n" + "="*60)
        print("🏆 OPTUNA OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Best CV Accuracy: {best_score:.4f}")
        print(f"Best Test Accuracy: {best_trial.user_attrs['test_accuracy']:.4f}")
        print(f"Number of trials: {len(study.trials)}")
        print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"\nBest Hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nParent Run ID: {parent_run.info.run_id}")
        
        print("\n💡 Tips:")
        print("  - Bayesian optimization lebih intelligent dari random search")
        print("  - Optuna learns dari previous trials")
        print("  - Pruning menghentikan unpromising trials early")
        print("  - Visualization plots membantu understand optimization process")
        
        print("\n📊 View results di MLflow UI:")
        print("  mlflow ui --backend-store-uri file:./mlruns")
        
        return study, final_model


def optuna_simple_example():
    """
    Contoh sederhana tanpa MLflow integration
    Untuk memahami Optuna basics
    """
    print("\n" + "="*60)
    print("📚 BONUS: Simple Optuna Example (tanpa MLflow)")
    print("="*60)
    
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def simple_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
        }
        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(simple_objective, n_trials=20, show_progress_bar=True)
    
    print(f"\nBest score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    # Run optimization dengan 50 trials
    study, model = optuna_bayesian_optimization(n_trials=50)
    
    # Uncomment untuk lihat simple example
    # optuna_simple_example()
