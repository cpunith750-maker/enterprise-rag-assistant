import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st

from app.retrieval.retriever import (
    load_faiss_index,
    load_chunks,
    load_reranker,
    hybrid_search,
    rerank_results,
    expand_with_neighbor_chunks,
)
from app.embeddings.embedder import load_embedding_model
from app.llm.answer_generator import generate_answer


st.set_page_config(
    page_title="Enterprise Document Assistant",
    page_icon="📄",
    layout="wide"
)


@st.cache(allow_output_mutation=True)
def load_rag_components():
    """
    Loads all heavy components once and caches them.
    This prevents models/indexes from reloading every time the user asks a question.
    """
    index = load_faiss_index()
    chunks = load_chunks()
    embedding_model = load_embedding_model()
    reranker = load_reranker()

    return index, chunks, embedding_model, reranker


def run_rag_pipeline(question, index, chunks, embedding_model, reranker):
    """
    Runs the full RAG pipeline:
    hybrid retrieval -> reranking -> neighbor expansion -> Groq answer generation.
    """

    initial_chunks = hybrid_search(
        query=question,
        index=index,
        chunks=chunks,
        model=embedding_model,
        semantic_top_k=200,
        keyword_top_k=100
    )

    reranked_chunks = rerank_results(
        query=question,
        results=initial_chunks,
        reranker=reranker,
        top_k=5
    )

    retrieved_chunks = expand_with_neighbor_chunks(
        results=reranked_chunks,
        chunks=chunks,
        window=1,
        max_chunks=8
    )

    answer = generate_answer(question, retrieved_chunks)

    return answer, retrieved_chunks, len(initial_chunks)


st.title("📄 Enterprise Document Assistant")
st.write(
    "Ask questions over your EnterpriseRAG-Bench sample documents. "
    "The system uses hybrid retrieval, reranking, neighbor chunk expansion, and Groq answer generation."
)

with st.sidebar:
    st.header("Pipeline")
    st.write("✅ FAISS semantic search")
    st.write("✅ Keyword search")
    st.write("✅ Cross-encoder reranking")
    st.write("✅ Neighbor chunk expansion")
    st.write("✅ Groq grounded answer")
    st.write("✅ Source citations")

st.markdown("---")

question = st.text_area(
    "Ask a question:",
    height=120,
    placeholder="Example: What is the standard amount of time a new hire buddy is expected to spend per day during the first two weeks?"
)

run_button = st.button("Generate Answer")

if run_button:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Loading RAG components..."):
            index, chunks, embedding_model, reranker = load_rag_components()

        with st.spinner("Running RAG pipeline..."):
            answer, retrieved_chunks, candidate_count = run_rag_pipeline(
                question=question,
                index=index,
                chunks=chunks,
                embedding_model=embedding_model,
                reranker=reranker
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        st.write(f"Total candidates before reranking: {candidate_count}")

        for chunk in retrieved_chunks:
            with st.expander(
                f"Source {chunk['rank']} | {chunk['source_type']} | {chunk['file_name']} | chunk {chunk['chunk_index']}"
            ):
                st.write("**Retrieval method:**", chunk.get("retrieval_method"))
                st.write("**Keyword score:**", chunk.get("keyword_score"))
                st.write("**Rerank score:**", chunk.get("rerank_score"))
                st.write("**File path:**", chunk.get("file_path"))
                st.write("**Text preview:**")
                st.write(chunk["text"][:1500])