import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("file:./mlruns")

# SET EXPERIMENT (AMAN DI SEMUA VERSI)
mlflow.set_experiment("iris-classification-rf")

def main():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="rf-baseline-v1"):    
        mlflow.set_tag(
            "experiment_description",
            "Eksperimen klasifikasi Iris menggunakan RandomForest sebagai baseline model"
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test[:5]
        )

        run_id = mlflow.active_run().info.run_id
        print("Accuracy:", acc)
        print("Run ID:", run_id)

if __name__ == "__main__":
    main()
