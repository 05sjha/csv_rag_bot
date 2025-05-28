# app/data_preparation/prepare_chunks.py

import pandas as pd

def create_chunks(df: pd.DataFrame, file_name: str) -> list:
    chunks = []

    # General overview
    chunks.append(f"Dataset '{file_name}' with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Column metadata
    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        missing = col_data.isna().sum()
        sample_vals = col_data.dropna().unique()[:5]
        sample_vals_str = ', '.join(map(str, sample_vals))

        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            stats = f"Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}, Min: {desc['min']}, Max: {desc['max']}"
        else:
            stats = "Non-numeric column."

        chunk = (
            f"Column: '{col}'\n"
            f"Type: {dtype}\n"
            f"Missing Values: {missing}\n"
            f"Sample Values: {sample_vals_str}\n"
            f"{stats}"
        )
        chunks.append(chunk)

    return chunks
