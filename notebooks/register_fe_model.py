import mlflow
from pyspark.sql import SparkSession
from src.reservations.config import databricks_config
from src.reservations.models.feature_lookup_model import (create_feature_table, load_data, feature_engineering, train_model, log_model, register_model,
                                                    load_latest_model_and_predict, calculate_lead_time_cat)

# Set the tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

# Set the active experiment
mlflow.set_experiment(databricks_config['experiment_name_fe'])

# Create a Spark session
spark = SparkSession.builder.getOrCreate()

# Create the feature table
create_feature_table()

# Load the data
train_set, test_set = load_data()

# Perform feature engineering
X_train, X_test, y_train, y_test = feature_engineering()

# Start an MLFlow run to train, log and register the model
with mlflow.start_run(run_name="fe_reservations_run") as run:
    run_id = run.info.run_id
    model = train_model(X_train, y_train)
    log_model(model, X_test, y_test)
    register_model(run_id)

# Example of making predictions with the latest model
all_data = spark.table(f"{databricks_config['catalog']}.{databricks_config['schema']}.{databricks_config['table_name']}")
all_data = all_data.withColumn("lead_time_cat", calculate_lead_time_cat(all_data["lead_time"])).toPandas()

X = all_data[databricks_config['num_features'] + databricks_config['num_features_fe']]
y = load_latest_model_and_predict(X)

print("MLflow experiment and model registration completed!")
print("Predictions:", y)
