import mlflow
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from src.reservations.config import databricks_config
from src.reservations.models.basic_model import (
    load_data,
    load_latest_model_and_predict,
    log_model,
    register_model,
    train_model,
)

# Set the tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# Set the active experiment
mlflow.set_experiment(databricks_config["experiment_name_basic"])

# Create a Spark session
spark = SparkSession.builder.getOrCreate()

# Load the dataset
X, y = load_data(databricks_config, spark)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLFlow run to train, log and register the model
with mlflow.start_run(run_name="basic_reservations_run") as run:
    run_id = run.info.run_id
    model = train_model(X_train, y_train)
    log_model(model, X_test, y_test, databricks_config)
    register_model(run_id)

# Example of making predictions with the latest model
all_data = spark.table(
    f"{databricks_config['catalog']}.{databricks_config['schema']}.{databricks_config['table_name']}"
).toPandas()

X = all_data[databricks_config["num_features"]]
y = load_latest_model_and_predict(X)

print("MLflow experiment and model registration completed!")
print("Predictions:", y)
