# enterprise-rag-assistant

Benchmark-driven Enterprise RAG Assistant using EnterpriseRAG-Bench, FAISS, FastAPI, Streamlit, and LLM-based citation-backed responses.





\# Enterprise Document Assistant



A RAG-based LLM application that answers questions from enterprise documents using hybrid retrieval, reranking, neighbor chunk expansion, and grounded answer generation.



\## Live Demo



Streamlit App: https://punith12.streamlit.app/



GitHub Repository: https://github.com/cpunith750-maker/enterprise-rag-assistant





\## Project Overview



This project uses a sample of Redwood enterprise documents from the EnterpriseRAG-Bench dataset. Redwood is a fictional enterprise company with internal documents such as Confluence pages, Jira tickets, GitHub documents, Google Drive files, policies, playbooks, onboarding guides, and technical runbooks.



The goal of this project is to build an AI-powered assistant that allows users to ask questions over company documents and receive answers with source support.



Instead of relying only on an LLM’s general knowledge, the system first retrieves relevant document chunks from the enterprise knowledge base and then sends that context to the LLM to generate a grounded answer.



\## Problem Statement



Companies store important information across many internal systems. Employees often waste time searching through long documents manually to find one specific answer.



A normal LLM does not automatically know a company’s private documents, so it may guess or hallucinate. This project solves that problem using Retrieval-Augmented Generation, also known as RAG.



The system retrieves relevant information first and then generates an answer using only that retrieved context.



\## Features



\- Loads enterprise documents from multiple source types

\- Splits long documents into smaller searchable chunks

\- Converts chunks into embeddings using Sentence Transformers

\- Stores embeddings in a FAISS vector index

\- Uses hybrid retrieval with semantic search and keyword search

\- Reranks retrieved chunks using a cross-encoder reranker

\- Expands neighboring chunks to capture nearby context

\- Generates grounded answers using Groq/Llama

\- Displays source documents used for each answer

\- Provides a Streamlit web interface

\- Includes a validation script for benchmark-style testing



\## RAG Pipeline



The system follows this pipeline:



1\. \*\*Document Loader\*\*  

&#x20;  Loads Redwood `.txt` documents from the sample dataset and converts them into structured Python dictionaries.



2\. \*\*Chunker\*\*  

&#x20;  Splits long documents into smaller searchable chunks with overlap.



3\. \*\*Embeddings\*\*  

&#x20;  Converts each chunk into a 384-dimensional embedding using `all-MiniLM-L6-v2`.



4\. \*\*FAISS Vector Store\*\*  

&#x20;  Stores chunk embeddings in `faiss.index` and saves chunk text/metadata in `chunks.json`.



5\. \*\*Hybrid Retrieval\*\*  

&#x20;  Combines semantic search using FAISS with keyword search for exact terms like IDs, emails, metrics, policy names, and technical keywords.



6\. \*\*Cross-Encoder Reranking\*\*  

&#x20;  Reranks retrieved chunks using a cross-encoder model to select the most relevant context.



7\. \*\*Neighbor Chunk Expansion\*\*  

&#x20;  Adds nearby chunks from the same document when the answer may be spread across chunk boundaries.



8\. \*\*Groq/Llama Answer Generation\*\*  

&#x20;  Sends the final retrieved context to Groq and generates an answer using a Llama model.



9\. \*\*Streamlit UI\*\*  

&#x20;  Provides a browser-based interface where users can ask questions and view answers with sources.



\## Tech Stack



\- Python

\- Streamlit

\- Sentence Transformers

\- FAISS

\- Cross-Encoder Reranker

\- Groq API

\- Llama model

\- JSON / CSV

\- GitHub

\- Streamlit Cloud



\## Project Structure



```text

enterprise-rag-assistant/

│

├── app/

│   ├── data\_loader/

│   │   └── loader.py

│   ├── chunking/

│   │   └── chunker.py

│   ├── embeddings/

│   │   └── embedder.py

│   ├── vector\_store/

│   │   └── faiss\_store.py

│   ├── retrieval/

│   │   └── retriever.py

│   └── llm/

│       └── answer\_generator.py

│

├── ui/

│   └── streamlit\_app.py

│

├── validation/

│   └── validate\_rag.py

│

├── vector\_index/

│   ├── faiss.index

│   └── chunks.json

│

├── assets/

│   └── streamlit-ui.png

│

├── questions.jsonl

├── requirements.txt

├── .gitignore

└── README.md


Validation



A validation script was created to test the RAG pipeline on benchmark questions available in the sample dataset.



Validation result:



Total benchmark questions: 500

Questions available in sample: 18

Questions tested: 8

Retrieval hits: 6/8

Retrieval hit rate: 75%



This means the system retrieved the expected source document for 6 out of 8 tested questions.



Some questions were difficult because the wording in the question was heavily paraphrased compared to the wording inside the source document. This is a realistic RAG challenge and shows why retrieval tuning, query rewriting, and reranking are important.



Example Questions:



Example 1

What is the standard amount of time a new hire buddy is expected to spend per day during the first two weeks when a long-term contractor is converted to a full-time employee?



Expected answer:



The new hire buddy is expected to spend 1 hour per day during the first two weeks.



Source:



embedded-contractor-conversion-playbook-2026.txt

Example 2

What uptime and latency service level targets are stated for the hosted versus reserved/dedicated LLM inference offering in the SaaS go-to-market brief?



Expected answer:



Hosted inference targets 99.9% regional uptime and P95 latency of 80–180ms for chat flows. Reserved or dedicated inference targets 99.95% uptime and P99 latency under 250ms.

Example 3

In our hiring workflow, what happens when a candidate is strong but the pay we want to offer is about 20 percent above the middle of the internal range, specifically who has to approve it?



Expected answer:



An offer around 20% above the internal range midpoint requires approval from the Hiring Manager and HRBP.


