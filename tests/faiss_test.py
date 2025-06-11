from app.vectorstore.faiss_db import create_vector_db, query_vector_db

sample_chunks = [
    "This vehicle has an average mileage of 15 kmpl.",
    "Fuel consumption increased in the last quarter.",
    "Breakdowns are frequent after 1,00,000 km usage."
]

print("Creating vector DB...")
create_vector_db(sample_chunks)

print("Querying...")
query = "Tell me about mileage"
results = query_vector_db(query)

for i, res in enumerate(results):
    print(f"[{i+1}] {res}")
