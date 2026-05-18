from app.loader import load_and_split
from app.embedder import embed_and_store

# Step 1 — Load and chunk the PDF
print("Loading and splitting PDF...")
chunks = load_and_split("test.pdf")
print(f"Total chunks: {len(chunks)}")

# Step 2 — Embed and store in ChromaDB
print("\nSending chunks to OpenAI and storing in ChromaDB...")
print("This will take 30-60 seconds for 297 chunks...")
vector_store = embed_and_store(chunks)
print("Done! ChromaDB created successfully.")

# Step 3 — Test similarity search
print("\n--- Testing similarity search ---")
question = "How do I clean the filter?"
results = vector_store.similarity_search(question, k=3)

print(f"\nQuestion: {question}")
print(f"\nTop 3 most relevant chunks:\n")

for i, doc in enumerate(results):
    print(f"Result {i+1} — Page {doc.metadata['page']}:")
    print(doc.page_content)
    print()