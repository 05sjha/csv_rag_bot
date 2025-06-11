# # app/vectorstore/chroma_db.py

# import os
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from chromadb.config import Settings

# # Directory where Chroma will store vector data
# CHROMA_DB_DIR = "app/vectorstore/chromadb"
# COLLECTION_NAME = "csv_rag_collection"

# # Set Chroma DB settings
# chroma_settings = Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory=CHROMA_DB_DIR
# )

# # Load the embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")


# def create_vector_db(chunks: list[str]):
#     """
#     Create a Chroma DB and store the document chunks.
#     """
#     db = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embedding_model,
#         persist_directory=CHROMA_DB_DIR,
#         client_settings=chroma_settings
#     )
#     db.add_texts(texts=chunks)
#     db.persist()
#     return db


# def load_vector_db():
#     """
#     Load an existing Chroma DB.
#     """
#     db = Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embedding_model,
#         persist_directory=CHROMA_DB_DIR,
#         client_settings=chroma_settings
#     )
#     return db


# def query_vector_db(query_text: str, k: int = 3):
#     """
#     Query the Chroma DB for similar documents.
#     """
#     db = load_vector_db()
#     results = db.similarity_search(query_text, k=k)
#     return results


import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define the directory where Chroma will store its data
CHROMA_DB_DIR = "app/vectorstore/chromadb"
COLLECTION_NAME = "csv_rag_collection"

# Initialize Chroma settings
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=CHROMA_DB_DIR
)

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

# Create a persistent Chroma client
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR,
    settings=chroma_settings
)

def create_vector_db(chunks: list[str]):
    """
    Create a Chroma collection and store the document chunks.
    """
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    # Generate embeddings for the chunks
    embeddings = embedding_model.embed_documents(chunks)
    # Add documents and their embeddings to the collection
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    return collection

def query_vector_db(query_text: str, k: int = 3):
    """
    Query the Chroma collection for similar documents.
    """
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    # Generate embedding for the query
    query_embedding = embedding_model.embed_query(query_text)
    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results
