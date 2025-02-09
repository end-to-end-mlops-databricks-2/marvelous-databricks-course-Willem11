def clean_data(df):
    """Function to clean data by removing empty values. I know it's a silly function but main objective is connecting with Databricks for now"""
    df_cleaned = df.dropna().reset_index(drop=True)
    return df_cleaned
