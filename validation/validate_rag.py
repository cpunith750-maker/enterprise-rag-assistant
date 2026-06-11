import json
import csv
from pathlib import Path

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


QUESTIONS_PATH = "questions.jsonl"
OUTPUT_JSON_PATH = "validation/validation_results.json"
OUTPUT_CSV_PATH = "validation/validation_results.csv"

MAX_QUESTIONS = 8


def load_questions(questions_path):
    """
    Loads benchmark questions from questions.jsonl.
    Each line in the file is one JSON object.
    """

    questions = []

    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    return questions


def get_available_doc_ids(chunks):
    """
    Gets document IDs that are available in our sample dataset.

    Our file names look like:
    dsid_005f7a937cad4b3cbb30d9d93199e22a__file-name.txt

    So we extract the dsid part before "__".
    """

    available_doc_ids = set()

    for chunk in chunks:
        file_name = chunk["file_name"]

        if "__" in file_name:
            doc_id = file_name.split("__")[0]
            available_doc_ids.add(doc_id)

    return available_doc_ids


def filter_questions_for_sample(questions, available_doc_ids):
    """
    Keeps only questions whose expected documents exist in our sample dataset.
    """

    filtered_questions = []

    for question in questions:
        expected_doc_ids = question.get("expected_doc_ids", [])

        for expected_doc_id in expected_doc_ids:
            if expected_doc_id in available_doc_ids:
                filtered_questions.append(question)
                break

    return filtered_questions


def check_retrieval_hit(retrieved_chunks, expected_doc_ids):
    """
    Checks whether any retrieved chunk came from the expected document.
    """

    for chunk in retrieved_chunks:
        file_name = chunk.get("file_name", "")

        for expected_doc_id in expected_doc_ids:
            if expected_doc_id in file_name:
                return True

    return False


def run_single_question(
    question_data,
    index,
    chunks,
    embedding_model,
    reranker
):
    """
    Runs the full RAG pipeline for one benchmark question.
    """

    question = question_data["question"]
    expected_doc_ids = question_data.get("expected_doc_ids", [])
    gold_answer = question_data.get("gold_answer", "")

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

    retrieval_hit = check_retrieval_hit(
        retrieved_chunks=retrieved_chunks,
        expected_doc_ids=expected_doc_ids
    )

    retrieved_sources = []

    for chunk in retrieved_chunks:
        retrieved_sources.append({
            "source_type": chunk.get("source_type"),
            "file_name": chunk.get("file_name"),
            "chunk_index": chunk.get("chunk_index"),
            "retrieval_method": chunk.get("retrieval_method"),
            "rerank_score": chunk.get("rerank_score"),
        })

    result = {
        "question_id": question_data.get("question_id"),
        "question": question,
        "expected_doc_ids": expected_doc_ids,
        "retrieval_hit": retrieval_hit,
        "gold_answer": gold_answer,
        "generated_answer": answer,
        "retrieved_sources": retrieved_sources,
    }

    return result


def save_results(results):
    """
    Saves validation results into JSON and CSV files.
    """

    output_json_path = Path(OUTPUT_JSON_PATH)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question_id",
                "question",
                "retrieval_hit",
                "expected_doc_ids",
                "gold_answer",
                "generated_answer",
            ]
        )

        writer.writeheader()

        for result in results:
            writer.writerow({
                "question_id": result["question_id"],
                "question": result["question"],
                "retrieval_hit": result["retrieval_hit"],
                "expected_doc_ids": result["expected_doc_ids"],
                "gold_answer": result["gold_answer"],
                "generated_answer": result["generated_answer"],
            })


if __name__ == "__main__":
    print("Loading FAISS index...")
    index = load_faiss_index()

    print("Loading chunks metadata...")
    chunks = load_chunks()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Loading reranker model...")
    reranker = load_reranker()

    print("Loading benchmark questions...")
    questions = load_questions(QUESTIONS_PATH)

    print("Finding questions that exist in our sample dataset...")
    available_doc_ids = get_available_doc_ids(chunks)
    sample_questions = filter_questions_for_sample(
        questions=questions,
        available_doc_ids=available_doc_ids
    )

    print(f"Total benchmark questions: {len(questions)}")
    print(f"Questions available in our sample: {len(sample_questions)}")

    questions_to_run = sample_questions[:MAX_QUESTIONS]

    print(f"Running validation on {len(questions_to_run)} questions...")

    results = []

    for i, question_data in enumerate(questions_to_run, start=1):
        print("\n" + "=" * 80)
        print(f"Running question {i}/{len(questions_to_run)}")
        print("Question ID:", question_data.get("question_id"))
        print("Question:", question_data["question"])

        try:
            result = run_single_question(
                question_data=question_data,
                index=index,
                chunks=chunks,
                embedding_model=embedding_model,
                reranker=reranker
            )

            results.append(result)

            print("Retrieval hit:", result["retrieval_hit"])
            print("Generated answer:")
            print(result["generated_answer"])

        except Exception as e:
            print("Error while validating this question:", e)

            results.append({
                "question_id": question_data.get("question_id"),
                "question": question_data["question"],
                "expected_doc_ids": question_data.get("expected_doc_ids", []),
                "retrieval_hit": False,
                "gold_answer": question_data.get("gold_answer", ""),
                "generated_answer": f"ERROR: {e}",
                "retrieved_sources": [],
            })

    save_results(results)

    total = len(results)
    hits = sum(1 for result in results if result["retrieval_hit"])

    print("\n" + "=" * 80)
    print("Validation complete.")
    print(f"Retrieval hits: {hits}/{total}")

    if total > 0:
        print(f"Retrieval hit rate: {(hits / total) * 100:.2f}%")

    print("\nSaved results:")
    print(OUTPUT_JSON_PATH)
    print(OUTPUT_CSV_PATH)