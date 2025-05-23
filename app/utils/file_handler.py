import pandas as pd

def read_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

def get_metadata(df: pd.DataFrame) -> dict:
    metadata = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict()
    }
    return metadata
