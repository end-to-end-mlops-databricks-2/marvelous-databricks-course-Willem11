import pandas as pd
from pyspark.sql import SparkSession

from src.reservations.config import databricks_config
from src.reservations.preprocessor import transform_string_columns

# Initialize Spark session
spark = SparkSession.builder.appName("HotelReservations").getOrCreate()

# Load dataset
file_path = "../data/Hotel Reservations.csv"
df = pd.read_csv(file_path)

# Clean dataset
df = transform_string_columns(df, drop_cols=['Booking_ID'], x_cols=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], y_col=['booking_status'])

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Write to Databricks Delta Table
spark_df.write.format("delta").saveAsTable(f"{databricks_config['catalog']}.{databricks_config['schema']}.{databricks_config['table_name']}")
