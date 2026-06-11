from app.data_loader.loader import load_documents


def chunk_text(text, chunk_size=800, overlap=100):
    """
    Splits one document text into smaller overlapping chunks.

    chunk_size = number of words per chunk
    overlap = number of words repeated between chunks
    """

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def create_chunks(documents, chunk_size=800, overlap=100):
    """
    Converts loaded documents into smaller chunks with metadata.
    """

    all_chunks = []

    for doc_index, document in enumerate(documents):
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk in enumerate(text_chunks):
            chunk_data = {
                "chunk_id": f"{doc_index}_{chunk_index}",
                "source_type": document["source_type"],
                "file_name": document["file_name"],
                "file_path": document["file_path"],
                "chunk_index": chunk_index,
                "text": chunk,
            }

            all_chunks.append(chunk_data)

    return all_chunks


if __name__ == "__main__":
    docs = load_documents()
    chunks = create_chunks(docs)

    print(f"Total documents loaded: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nSample chunk:")
        print("Chunk ID:", chunks[0]["chunk_id"])
        print("Source:", chunks[0]["source_type"])
        print("File name:", chunks[0]["file_name"])
        print("Chunk index:", chunks[0]["chunk_index"])
        print("Text preview:")
        print(chunks[0]["text"][:700])