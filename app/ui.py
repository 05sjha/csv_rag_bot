import streamlit as st
from app.utils.file_handler import read_csv, get_metadata

st.set_page_config(page_title="CSV RAG Bot - Upload", layout="wide")

st.title("📄 CSV Upload & Preview")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = read_csv(uploaded_file)
        st.subheader("🔍 Preview of Data")
        st.dataframe(df.head())

        st.subheader("🧠 Metadata")
        metadata = get_metadata(df)

        st.markdown("**Columns & Data Types**")
        st.json(metadata["dtypes"])

        st.markdown("**Null Values Per Column**")
        st.json(metadata["null_counts"])

        st.markdown("**Summary Statistics**")
        st.json(metadata["summary_stats"])

    except ValueError as e:
        st.error(str(e))
