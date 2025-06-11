import faiss
import numpy as np
import os
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")

FAISS_INDEX_FILE = "app/vectorstore/faiss_index.idx"
FAISS_META_FILE = "app/vectorstore/faiss_meta.pkl"

def create_vector_db(chunks: list[str]):
    vectors = embedding_model.embed_documents(chunks)
    dim = len(vectors[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))

    # Save the index and metadata
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)

def query_vector_db(query: str, k=3):
    # Load index and metadata
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError("FAISS index not found. Run `create_vector_db()` first.")

    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "rb") as f:
        chunks = pickle.load(f)

    query_vec = embedding_model.embed_query(query)
    D, I = index.search(np.array([query_vec]).astype('float32'), k)

    results = [chunks[i] for i in I[0]]
    return results
