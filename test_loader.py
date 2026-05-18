from app.loader import load_and_split

chunks = load_and_split("test.pdf")

print(f"Total chunks created: {len(chunks)}")
print("\n--- First chunk ---")
print(chunks[0].page_content)
print("\n--- Second chunk ---")
print(chunks[1].page_content)
print("\n--- Metadata of first chunk ---")
print(chunks[0].metadata)