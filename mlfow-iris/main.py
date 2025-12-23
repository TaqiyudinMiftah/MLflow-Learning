import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    # Load dataset
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Log model + input example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test[:5]
        )

        print("Training selesai")
        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
