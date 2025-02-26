import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.reservations.config import databricks_config


def create_feature_table():
    """Create or replace the feature table and make splits in train and test sets"""
    config = databricks_config
    spark = SparkSession.builder.getOrCreate()
    feature_table_name = f"{config['catalog']}.{config['schema']}.processed_reservations"
    feature_df = spark.table(feature_table_name)
    train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=1234)
    train_df.createOrReplaceTempView("train_temp_view")
    test_df.createOrReplaceTempView("test_temp_view")
    train_table_name = f"{config['catalog']}.{config['schema']}.reservation_features_train"
    test_table_name = f"{config['catalog']}.{config['schema']}.reservation_features_test"
    train_set_sql = f"""CREATE OR REPLACE TABLE {train_table_name} AS SELECT * FROM train_temp_view"""
    test_set_sql = f"""CREATE OR REPLACE TABLE {test_table_name} AS SELECT * FROM test_temp_view"""
    spark.sql(train_set_sql)
    spark.sql(test_set_sql)


def calculate_lead_time_cat(lead_time):
    """Calculate the lead_time_cat feature for PySpark DataFrame."""
    return (
        when(lead_time < 100, 1)
        .when((lead_time >= 100) & (lead_time < 200), 2)
        .when((lead_time >= 200) & (lead_time < 300), 3)
        .when((lead_time >= 300) & (lead_time < 400), 4)
        .otherwise(5)
    )


def load_data():
    """Load training and testing data from tables and apply lead_time_cat function"""
    config = databricks_config
    spark = SparkSession.builder.getOrCreate()
    train_set = spark.table(f"{config['catalog']}.{config['schema']}.reservation_features_train")
    test_set = spark.table(f"{config['catalog']}.{config['schema']}.reservation_features_test")
    return train_set, test_set


def feature_engineering():
    """Perform feature engineering by adding the lead_time_cat feature."""
    config = databricks_config
    train_set, test_set = load_data()
    train_set = train_set.withColumn("lead_time_cat", calculate_lead_time_cat(train_set["lead_time"])).toPandas()
    test_set = test_set.withColumn("lead_time_cat", calculate_lead_time_cat(test_set["lead_time"])).toPandas()
    X_train = train_set[config["num_features"] + ["lead_time_cat"]]
    y_train = train_set[config["target"]]
    X_test = test_set[config["num_features"] + ["lead_time_cat"]]
    y_test = test_set[config["target"]]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the logistic regression model."""
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model


def log_model(model, X_test, y_test):
    """Log the model and its metrics within the active run."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)
    signature = infer_signature(model_input=X_test, model_output=y_pred)
    mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic-regression-model-fe", signature=signature)


def register_model(run_id):
    """Register model in the default MLflow model registry."""
    model_uri = f"runs:/{run_id}/logistic-regression-model-fe"
    model_name = "fe_reservations_model"
    mlflow.register_model(model_uri, model_name)
    print(f"Model registered with URI: {model_uri}")


def load_latest_model_and_predict(input_data):
    """Load the latest model from MLflow and make predictions."""
    client = MlflowClient()
    model_name = "fe_reservations_model"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)
    predictions = model.predict(input_data)
    return predictions
