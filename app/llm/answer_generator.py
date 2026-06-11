import os

from dotenv import load_dotenv
from groq import Groq

from app.retrieval.retriever import (
    load_faiss_index,
    load_chunks,
    load_reranker,
    hybrid_search,
    rerank_results,
    expand_with_neighbor_chunks,
)
from app.embeddings.embedder import load_embedding_model


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)


def build_context(retrieved_chunks):
    """
    Converts retrieved chunks into a clean context block for Groq.
    """

    context_parts = []

    for chunk in retrieved_chunks:
        source_label = (
            f"[Source {chunk['rank']}: "
            f"{chunk['source_type']} | "
            f"{chunk['file_name']} | "
            f"chunk {chunk['chunk_index']}]"
        )

        context_parts.append(source_label + "\n" + chunk["text"][:2500])

    return "\n\n".join(context_parts)


def generate_answer(question, retrieved_chunks):
    """
    Sends retrieved context + user question to Groq
    and generates a grounded answer.
    """

    context = build_context(retrieved_chunks)

    prompt = f"""
You are an enterprise document assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Answer only from the provided context.
- Give a complete answer, not just one sentence.
- If the context contains a process, explain it step by step.
- Mention the relevant source numbers in the answer.
- If the answer is not found in the context, say:
  "I could not find this information in the available documents."
- Do not use sources that do not directly support the answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("Loading FAISS index...")
    index = load_faiss_index()

    print("Loading chunks metadata...")
    chunks = load_chunks()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading reranker model...")
    reranker = load_reranker()

    question = input("\nAsk a question: ")

    print("\nRetrieving candidate chunks using hybrid search...")
    initial_chunks = hybrid_search(
        query=question,
        index=index,
        chunks=chunks,
        model=embedding_model,
        semantic_top_k=30,
        keyword_top_k=30
    )

    print(f"\nTotal candidates before reranking: {len(initial_chunks)}")

    print("\nReranking retrieved chunks...")
    reranked_chunks = rerank_results(question, initial_chunks, reranker, top_k=5)

    print("\nExpanding with neighboring chunks...")
    retrieved_chunks = expand_with_neighbor_chunks(
        results=reranked_chunks,
        chunks=chunks,
        window=1,
        max_chunks=8
    )

    print("\nGenerating answer with Groq...")

    try:
        answer = generate_answer(question, retrieved_chunks)

        print("\nAnswer:")
        print(answer)

    except Exception as e:
        print("\nGroq answer generation failed.")
        print("Reason:", e)

        print("\nFallback Answer:")
        print("The system retrieved relevant chunks, but the LLM call failed.")
        print("Here are the retrieved chunks:\n")

        for chunk in retrieved_chunks:
            print("=" * 80)
            print(
                f"Source {chunk['rank']}: "
                f"{chunk['source_type']} | "
                f"{chunk['file_name']} | "
                f"chunk {chunk['chunk_index']}"
            )
            print()
            print(chunk["text"][:1200])
            print()

    print("\nSources:")
    for chunk in retrieved_chunks:
        print(
            f"- Source {chunk['rank']}: "
            f"{chunk['source_type']} | "
            f"{chunk['file_name']} | "
            f"chunk {chunk['chunk_index']} "
            f"| method: {chunk['retrieval_method']} "
            f"| keyword_score: {chunk['keyword_score']} "
            f"| rerank_score: {chunk['rerank_score']:.4f}"
        )