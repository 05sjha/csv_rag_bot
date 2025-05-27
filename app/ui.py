import streamlit as st
from app.data_handler import save_uploaded_file, read_csv, clean_csv

st.title("CSV RAG Bot - Phase 3")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    
    file_path = save_uploaded_file(uploaded_file)
    df = read_csv(file_path)

    if df is not None:
        st.info("Showing top 5 rows of the raw data:")
        st.dataframe(df.head())

        df_clean = clean_csv(df)
        st.success("✅ Data cleaned successfully.")
        st.dataframe(df_clean.head())

        # (Optionally save cleaned dataframe to session)
        st.session_state["clean_df"] = df_clean
    else:
        st.error("❌ Failed to read CSV. Please check file format.")
