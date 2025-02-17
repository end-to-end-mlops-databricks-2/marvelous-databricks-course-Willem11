import pandas as pd


def transform_string_columns(df, drop_cols, x_cols, y_col):
    """Function to transform the string columns to integer columns so that they can be used by an ML model"""
    for col in drop_cols:
        df = df.drop(columns=[col])
    for col in x_cols:
        temp = pd.get_dummies(df[col]).astype(int).add_prefix(col + "_")
        temp.columns = temp.columns.str.lower().str.replace(" ", "_")
        df = pd.concat([df, temp], axis=1)
        df = df.drop(columns=[col])
    for col in y_col:
        df[col] = df[col].map({"Not_Canceled": 0, "Canceled": 1})
    return df
