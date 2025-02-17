import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data(config, spark):
    """Load data from table."""
    df = spark.table(f"{config['catalog']}.{config['schema']}.{config['table_name']}")
    data = df.toPandas()
    X = data[config['num_features']]
    y = data[config['target']]
    return X, y

def train_model(X, y):
    """Train the logistic regression model."""
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model

def log_model(model, X, y, config):
    """Log the model and its metrics within the active run."""
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=X, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic-regression-model", signature=signature)

def register_model(run_id):
    """Register model in the default MLflow model registry."""
    model_uri = f"runs:/{run_id}/logistic-regression-model"
    model_name = "basic_reservations_model"
    mlflow.register_model(model_uri, model_name)
    print(f"Model registered with URI: {model_uri}")

def load_latest_model_and_predict(input_data):
    """Load the latest model from MLflow and make predictions."""
    client = mlflow.client.MlflowClient()
    model_name = "basic_reservations_model"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(input_data)
    return predictions
