# Add the Project Root to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.data_handler import read_csv, clean_csv
from app.embeddings.embedding_generator import structure_row_prompt_dict_and_text
from app.vectorstore.faiss_db import create_vector_db, query_vector_db
from app.llm.ollama_llm import query_ollama

st.title("📂 CSV RAG Bot")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully.")
    
    df = read_csv(uploaded_file)
    
    if df is not None:
        df_clean = clean_csv(df)

        with st.expander("📊 File Metadata", expanded=False):
            st.write(f"🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("🧠 Column Names:", list(df.columns))
            st.write("🧮 Data Types:", df.dtypes)
            st.write("❓ Missing Values Count:", df.isnull().sum())
        
        with st.expander("🔍 Preview Top 5 Rows", expanded=False):
            st.dataframe(df_clean.head())

        # Save cleaned df in session state
        st.session_state["clean_df"] = df_clean

        # Initialize vector_db_created state
        if "vector_db_created" not in st.session_state:
            st.session_state["vector_db_created"] = False

        # Step 1: Create vector DB only once
        if not st.session_state["vector_db_created"]:
            st.info("Creating vector database...")

            row_prompts = df_clean.to_dict(orient="records")
            text_chunks = [structure_row_prompt_dict_and_text(row) for row in row_prompts]

            num_chunks = create_vector_db(text_chunks)
            st.session_state["vector_db_created"] = True

            st.success(f"✅ Created vector store with {num_chunks} entries.")

        # Step 2: Ask a question
        st.markdown("---")
        st.header("🔍 Ask a Question About Your Data")

        query_text = st.text_input("Ask a question from the CSV")

        if st.button("Ask"):
            if not query_text.strip():
                st.warning("Please enter a valid question.")
            else:
                results = query_vector_db(query_text, k=5)

                if results:
                    context = "\n".join(results)
                    prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
                    llm_response = query_ollama(prompt)

                    st.markdown("### 🤖 LLM Answer")
                    st.write(llm_response)
                else:
                    st.info("No relevant chunks found.")
    else:
        st.error("❌ Failed to read CSV. Please check file format.")
