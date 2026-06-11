# enterprise-rag-assistant

Benchmark-driven Enterprise RAG Assistant using EnterpriseRAG-Bench, FAISS, FastAPI, Streamlit, and LLM-based citation-backed responses.





\# Enterprise Document Assistant



A RAG-based LLM system that answers questions from enterprise documents using hybrid retrieval, reranking, neighbor chunk expansion, and grounded answer generation.



\## Project Overview



This project uses a sample of Redwood enterprise documents from the EnterpriseRAG-Bench dataset. Redwood is a fictional enterprise company with internal documents such as Confluence pages, Jira tickets, GitHub documents, Google Drive files, policies, playbooks, onboarding guides, and technical runbooks.



The goal of this project is to build an AI-powered assistant that allows users to ask questions over company documents and receive answers with source support.



\## Problem Statement



Companies store important information across many internal systems. Employees often waste time searching through long documents manually to find one specific answer.



A normal LLM does not automatically know a company’s private documents, so it may guess. This project solves that problem using Retrieval-Augmented Generation (RAG). The system first retrieves relevant document chunks and then sends that context to an LLM to generate a grounded answer.



\## RAG Pipeline



The system follows this pipeline:



1\. Document Loader  

&#x20;  Loads Redwood `.txt` documents from the sample dataset and converts them into structured Python dictionaries.



2\. Chunker  

&#x20;  Splits long documents into smaller searchable chunks with overlap.



3\. Embeddings  

&#x20;  Converts each chunk into a 384-dimensional embedding using `all-MiniLM-L6-v2`.



4\. FAISS Vector Store  

&#x20;  Stores chunk embeddings in `faiss.index` and saves chunk text/metadata in `chunks.json`.



5\. Hybrid Retrieval  

&#x20;  Combines semantic search using FAISS with keyword search for exact terms like IDs, emails, metrics, and policy names.



6\. Cross-Encoder Reranking  

&#x20;  Reranks retrieved chunks using a cross-encoder model to select the most relevant context.



7\. Neighbor Chunk Expansion  

&#x20;  Adds nearby chunks from the same document when the answer may be spread across chunk boundaries.



8\. Groq/Llama Answer Generation  

&#x20;  Sends the final retrieved context to Groq and generates an answer using a Llama model.



9\. Streamlit UI  

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

├── questions.jsonl

├── requirements.txt

├── .gitignore

└── README.md

