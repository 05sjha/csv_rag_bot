# from sentence_transformers import SentenceTransformer
# from typing import List

# # Load E5 model
# # This will download the model the first time you run
# model = SentenceTransformer("intfloat/e5-small-v2")

# def generate_embeddings(chunks: List[str]) -> List[List[float]]:
#     """
#     Generates vector embeddings for a list of text chunks using e5-small-v2.
    
#     Args:
#         chunks (List[str]): List of string chunks
    
#     Returns:
#         List[List[float]]: Embeddings for each chunk
#     """
#     # Add instruction prefix as required by E5
#     instruction_prompt = "passage: "
#     formatted_chunks = [instruction_prompt + chunk for chunk in chunks]

#     # Compute embeddings
#     embeddings = model.encode(formatted_chunks, show_progress_bar=True, convert_to_numpy=True)

#     return embeddings

from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Initialize E5 model
e5_model = SentenceTransformer("intfloat/e5-base")

def structure_row_prompt(row: Dict) -> str:
    """
    Converts a row dict to a structured prompt for better embedding.
    """
    return " | ".join([f"{k}: {v}" for k, v in row.items()])

def structure_row_prompt_dict_and_text(row) -> str:
    """
    Converts either a dict (structured row) or string into a prompt.
    """
    if isinstance(row, dict):
        return " | ".join([f"{k}: {v}" for k, v in row.items()])
    return row  # assume it's already a string


def prepare_texts_for_embedding(chunks: List[Dict]) -> List[str]:
    """
    Prefix each structured row prompt with 'passage:' for e5 model.
    """
    return [f"passage: {structure_row_prompt_dict_and_text(row)}" for row in chunks]

def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    """
    Generate E5 embeddings from row-chunks.
    """
    texts = prepare_texts_for_embedding(chunks)
    return e5_model.encode(texts, show_progress_bar=True)

