# RAG PDF Q&A

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system with a simple web interface using Streamlit. The system allows users to:

- Upload a PDF file,
- Extract and process the PDF content,
- Ask questions related to the uploaded PDF content,
- Receive AI-generated answers based on the content context.

---

## Technologies Used

- **Python 3.12**
- **Streamlit** — for building the web interface
- **Framework: Langchain** - to orchestrate the retrieval and generation pipeline effectively
- **Transformers (Hugging Face)** — to run the language model for generating answers

---

## Models Used

- **Embedding model:** `bkai-foundation-models/vietnamese-bi-encoder`
- **Language Generation model:** `lmsys/vicuna-7b-v1.5`
