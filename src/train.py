
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# 1. Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Handwritten-Digit-Classification")


# 2. Load dataset
data = pd.read_csv("data/raw/dataset.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# 4. Train and track models
results = []

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="weighted"
        )
        recall = recall_score(
            y_test, predictions, average="weighted"
        )
        f1 = f1_score(
            y_test, predictions, average="weighted"
        )

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "model_name": model_name,
            "accuracy": accuracy,
            "run_id": mlflow.active_run().info.run_id
        })

        print(model_name, "Accuracy:", accuracy)


# 5. Select best model
best_result = max(results, key=lambda x: x["accuracy"])

print("\nBest Model:", best_result["model_name"])
print("Best Accuracy:", best_result["accuracy"])


# 6. Register best model
model_uri = f"runs:/{best_result['run_id']}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="HandwrittenDigitClassifier"
)

# Set champion alias
client = MlflowClient()

client.set_registered_model_alias(
    "HandwrittenDigitClassifier",
    "champion",
    registered_model.version
)

print("\nBest model registered successfully!")
print("Model Version:", registered_model.version)
print("Alias: champion")
