import pandas as pd
import os

UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded CSV to a local directory and return the path."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def read_csv(file_path):
    """Read the uploaded CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return None

def clean_csv(df):
    """Do basic cleaning – drop NA columns, strip column names, etc."""
    df = df.dropna(axis=1, how="all")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df
