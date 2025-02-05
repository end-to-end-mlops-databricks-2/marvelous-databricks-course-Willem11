from pyspark.sql import SparkSession
import pandas as pd
from src.reservations.config import databricks_config
from src.reservations.clean_data import clean_data

# Initialize Spark session
spark = SparkSession.builder.appName("HotelReservations").getOrCreate()

# Load dataset
file_path = "../data/Hotel Reservations.csv"
df = pd.read_csv(file_path)

# Clean dataset
df_cleaned = clean_data(df)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df_cleaned)

# Write to Databricks Delta Table
spark_df.write.mode("overwrite").format("delta").saveAsTable(f"{databricks_config['catalog']}.{databricks_config['schema']}.{databricks_config['table_name']}")