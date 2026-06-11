from sentence_transformers import SentenceTransformer
from app.data_loader.loader import load_documents
from app.chunking.chunker import create_chunks


def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Loads a free sentence-transformer embedding model.
    This model converts text into vectors.
    """
    model = SentenceTransformer(model_name)
    return model


def generate_embeddings(chunks, model):
    """
    Converts each chunk text into an embedding vector.
    """

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


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

    print("Embeddings generated successfully.")
    print("Number of embeddings:", len(embeddings))
    print("Shape of embeddings:", embeddings.shape)

    print("\nSample chunk text:")
    print(chunks[0]["text"][:300])

    print("\nSample embedding vector first 10 numbers:")
    print(embeddings[0][:10])