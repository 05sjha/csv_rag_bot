from app.vectorstore.chromadb import create_vector_db, query_vector_db

# Sample document chunks
chunks = [
    "This vehicle has an average mileage of 15 kmpl.",
    "Fuel consumption increased in the last quarter.",
    "Breakdowns are frequent after 1,00,000 km usage."
]

# Create the vector database
create_vector_db(chunks)

# Query the vector database
query = "How is the fuel efficiency?"
results = query_vector_db(query)

# Display the results
for i, doc in enumerate(results['documents'][0]):
    print(f"[{i+1}] {doc}")
