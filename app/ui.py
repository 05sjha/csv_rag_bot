import streamlit as st
from app.data_handler import save_uploaded_file, read_csv, clean_csv

st.title("CSV RAG Bot - Phase 3")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    
    file_path = save_uploaded_file(uploaded_file)
    df = read_csv(file_path)

    if df is not None:
        df_clean = clean_csv(df)

        with st.expander("📊 Raw File Metadata", expanded=False):
            st.write(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("🧠 Column Names:", list(df.columns))
            st.write("🧮 Data Types:")
            st.write(df.dtypes)
            st.write("❓ Missing Values Count:")
            st.write(df.isnull().sum())

        with st.expander("🧼 Cleaned File Metadata", expanded=False):
            st.write(f"🔢 Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
            st.write("🧠 Column Names:", list(df_clean.columns))
            st.write("🧮 Data Types:")
            st.write(df_clean.dtypes)
            st.write("❓ Missing Values Count:")
            st.write(df_clean.isnull().sum())


        with st.expander("🔍 Preview: Top 5 Rows of Raw Data", expanded=False):
            st.dataframe(df.head())

        with st.expander("🔍 Preview: Top 5 Rows of Cleaned Data", expanded=False):
            st.dataframe(df_clean.head())

        


        # (Optionally save cleaned dataframe to session)
        st.session_state["clean_df"] = df_clean
    else:
        st.error("❌ Failed to read CSV. Please check file format.")
