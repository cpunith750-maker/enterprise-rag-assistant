import json
import re
from collections import Counter

import faiss
from sentence_transformers import CrossEncoder

from app.embeddings.embedder import load_embedding_model


def load_faiss_index(index_path="vector_index/faiss.index"):
    """
    Loads the saved FAISS index from disk.
    FAISS stores the embedding vectors.
    """
    return faiss.read_index(index_path)


def load_chunks(chunks_path="vector_index/chunks.json"):
    """
    Loads chunk text and metadata from chunks.json.
    FAISS only stores vectors, so this file gives us the actual text and citations.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks


def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Loads a general cross-encoder reranker.
    It compares question + chunk and gives a relevance score.
    """
    return CrossEncoder(model_name)


def tokenize(text):
    """
    Simple tokenizer for keyword search.
    Keeps useful terms like 5xx, p99, 60m, etc.
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)


def semantic_search(query, index, chunks, model, top_k=30):
    """
    First retrieval method: semantic search using embeddings + FAISS.
    Good when question wording and document wording are different.
    """
    query_embedding = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for rank, chunk_index in enumerate(indices[0]):
        chunk = chunks[chunk_index]

        results.append({
            "rank": rank + 1,
            "retrieval_method": "semantic",
            "faiss_score": float(distances[0][rank]),
            "keyword_score": None,
            "chunk_id": chunk["chunk_id"],
            "source_type": chunk["source_type"],
            "file_name": chunk["file_name"],
            "file_path": chunk["file_path"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        })

    return results


def keyword_search(query, chunks, top_k=30):
    """
    Second retrieval method: keyword search over chunks.json.
    Good for exact terms like 5xx, p99, AUTH-REQ, SOC2, dates, IDs, and thresholds.
    """

    query_terms = tokenize(query)

    stopwords = {
        "the", "and", "for", "that", "this", "with", "what", "when", "where",
        "who", "how", "are", "is", "was", "were", "during", "about", "into",
        "from", "our", "can", "you", "does", "did", "has", "have", "had",
        "first", "then", "than", "they", "their", "there", "which", "while",
        "would", "could", "should", "specific", "concrete"
    }

    query_terms = [
        term for term in query_terms
        if len(term) > 2 and term not in stopwords
    ]

    query_counter = Counter(query_terms)

    results = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"].lower()
        file_name = chunk["file_name"].lower()

        score = 0

        for term, count in query_counter.items():
            text_matches = text.count(term)
            file_matches = file_name.count(term)

            score += text_matches * count
            score += file_matches * count * 3

        if score > 0:
            results.append({
                "rank": None,
                "retrieval_method": "keyword",
                "faiss_score": None,
                "keyword_score": score,
                "chunk_id": chunk["chunk_id"],
                "source_type": chunk["source_type"],
                "file_name": chunk["file_name"],
                "file_path": chunk["file_path"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            })

    results = sorted(results, key=lambda x: x["keyword_score"], reverse=True)

    return results[:top_k]


def hybrid_search(query, index, chunks, model, semantic_top_k=30, keyword_top_k=30):
    """
    Combines semantic search and keyword search.
    Removes duplicate chunks using chunk_id.
    """

    semantic_results = semantic_search(
        query=query,
        index=index,
        chunks=chunks,
        model=model,
        top_k=semantic_top_k
    )

    keyword_results = keyword_search(
        query=query,
        chunks=chunks,
        top_k=keyword_top_k
    )

    combined = {}
    combined_order = []

    for result in semantic_results + keyword_results:
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = result
            combined_order.append(chunk_id)
        else:
            existing = combined[chunk_id]

            existing_method = existing["retrieval_method"]
            new_method = result["retrieval_method"]

            if existing_method != new_method:
                existing["retrieval_method"] = "semantic+keyword"

            if existing.get("faiss_score") is None:
                existing["faiss_score"] = result.get("faiss_score")

            if existing.get("keyword_score") is None:
                existing["keyword_score"] = result.get("keyword_score")

    return [combined[chunk_id] for chunk_id in combined_order]


def rerank_results(query, results, reranker, top_k=5):
    """
    Reranks combined semantic + keyword candidates using a cross-encoder.
    Higher rerank_score is better.
    """

    pairs = []

    for result in results:
        pairs.append([query, result["text"][:3000]])

    rerank_scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, rerank_scores):
        result["rerank_score"] = float(score)
        reranked_results.append(result)

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    final_results = reranked_results[:top_k]

    for new_rank, result in enumerate(final_results, start=1):
        result["rank"] = new_rank

    return final_results
def expand_with_neighbor_chunks(results, chunks, window=1, max_chunks=8):
    """
    Adds neighboring chunks from the same document.

    Example:
    If chunk 2 is retrieved, also include chunk 1 and chunk 3
    from the same file.

    window=1 means add one previous and one next chunk.
    """

    expanded = []
    seen = set()

    # Build lookup: (file_name, chunk_index) -> chunk
    chunk_lookup = {}

    for chunk in chunks:
        key = (chunk["file_name"], chunk["chunk_index"])
        chunk_lookup[key] = chunk

    for result in results:
        file_name = result["file_name"]
        current_chunk_index = result["chunk_index"]

        neighbor_indexes = range(
            current_chunk_index - window,
            current_chunk_index + window + 1
        )

        for neighbor_index in neighbor_indexes:
            key = (file_name, neighbor_index)

            if key not in chunk_lookup:
                continue

            neighbor_chunk = chunk_lookup[key]
            unique_key = neighbor_chunk["chunk_id"]

            if unique_key in seen:
                continue

            seen.add(unique_key)

            expanded.append({
                "rank": len(expanded) + 1,
                "retrieval_method": result.get("retrieval_method", "neighbor_expansion"),
                "faiss_score": result.get("faiss_score"),
                "keyword_score": result.get("keyword_score"),
                "rerank_score": result.get("rerank_score"),
                "chunk_id": neighbor_chunk["chunk_id"],
                "source_type": neighbor_chunk["source_type"],
                "file_name": neighbor_chunk["file_name"],
                "file_path": neighbor_chunk["file_path"],
                "chunk_index": neighbor_chunk["chunk_index"],
                "text": neighbor_chunk["text"],
            })

            if len(expanded) >= max_chunks:
                return expanded

    return expanded

if __name__ == "__main__":
    print("Loading FAISS index...")
    index = load_faiss_index()

    print("Loading chunks metadata...")
    chunks = load_chunks()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading reranker model...")
    reranker = load_reranker()

    query = input("\nAsk a question: ")

    print("\nRunning hybrid search...")
    initial_results = hybrid_search(
        query=query,
        index=index,
        chunks=chunks,
        model=embedding_model,
        semantic_top_k=30,
        keyword_top_k=30
    )

    print(f"\nTotal hybrid candidates: {len(initial_results)}")

    print("\nReranking results...")
    final_results = rerank_results(query, initial_results, reranker, top_k=5)

    print("\nTop reranked chunks:\n")

    for result in final_results:
        print("=" * 80)
        print("Rank:", result["rank"])
        print("Retrieval Method:", result["retrieval_method"])
        print("FAISS Score:", result["faiss_score"])
        print("Keyword Score:", result["keyword_score"])
        print("Rerank Score:", result["rerank_score"])
        print("Source:", result["source_type"])
        print("File:", result["file_name"])
        print("Chunk ID:", result["chunk_id"])
        print("Chunk Index:", result["chunk_index"])
        print("\nText preview:")
        print(result["text"][:700])
        print()