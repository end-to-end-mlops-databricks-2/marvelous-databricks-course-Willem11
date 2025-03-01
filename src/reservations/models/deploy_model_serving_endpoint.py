import os
import time

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split

from src.reservations.config import databricks_config
from src.reservations.models.basic_model import load_data
from src.reservations.serving.model_serving import call_endpoint, deploy_or_update_serving_endpoint

# Initialize Spark session and DBUtils
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Set environment variables
os.environ["DBR_TOKEN"] = databricks_config["token"]
os.environ["DBR_HOST"] = databricks_config["host"]

# Load project config
catalog_name = databricks_config["catalog"]
schema_name = databricks_config["schema"]

# Define model and endpoint names
model_name = "basic_reservations_model"
endpoint_name = "basic_reservations_model_serving"

# Deploy the model serving endpoint
deploy_or_update_serving_endpoint(model_name, endpoint_name)

# # Create a sample request body
required_columns = databricks_config["num_features"]

# # Sample records from the test set
X, y = load_data(databricks_config, spark)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sampled_records = X_test[required_columns].to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# Call the endpoint with one sample record
status_code, response_text = call_endpoint(endpoint_name, dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")
print(len(dataframe_records))

# Load test
for record in dataframe_records[:10]:
    call_endpoint(endpoint_name, record)
    time.sleep(0.5)
