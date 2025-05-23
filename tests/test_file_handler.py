import pandas as pd
from app.utils.file_handler import read_csv, get_metadata
from io import StringIO

def test_read_csv():
    sample_csv = StringIO("A,B,C\n1,2,3\n4,5,6")
    df = read_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)

def test_metadata():
    df = pd.DataFrame({
        "A": [1, 2, None],
        "B": ["x", "y", "z"]
    })
    metadata = get_metadata(df)
    assert metadata["dtypes"]["A"] == "float64"
    assert metadata["null_counts"]["A"] == 1
    assert "A" in metadata["summary_stats"]
