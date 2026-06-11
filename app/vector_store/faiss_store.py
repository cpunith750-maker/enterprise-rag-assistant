import json
from pathlib import Path

import faiss

from app.data_loader.loader import load_documents
from app.chunking.chunker import create_chunks
from app.embeddings.embedder import load_embedding_model, generate_embeddings


def build_faiss_index(embeddings):
    """
    Builds a FAISS index from embedding vectors.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def save_chunks(chunks, output_path="vector_index/chunks.json"):
    """
    Saves chunk text and metadata to a JSON file.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def save_faiss_index(index, output_path="vector_index/faiss.index"):
    """
    Saves FAISS index to disk.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_file))


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print("Creating chunks...")
    chunks = create_chunks(docs)

    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving FAISS index...")
    save_faiss_index(index)

    print("Saving chunks metadata...")
    save_chunks(chunks)

    print("FAISS vector store created successfully.")
    print("Saved files:")
    print("- vector_index/faiss.index")
    print("- vector_index/chunks.json")