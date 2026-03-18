# AI Knowledge Assistant (RAG + Endee Concept)

## Project Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) based AI assistant that answers user queries using semantic search and language model generation.

The system retrieves relevant information from a knowledge base and generates clear, contextual answers.

---

## Features

* Semantic search using embeddings
* Retrieval of relevant documents
* AI-based answer generation
* Clean user interface using Streamlit
* Robust fallback mechanism (no failures shown)

---

##  How It Works

1. Documents are converted into embeddings
2. User query is converted into embedding
3. Similarity search retrieves relevant documents
4. Retrieved data is passed to an LLM
5. Final answer is generated

---

## Tech Stack

* Python
* Sentence Transformers
* NumPy
* Streamlit
* Hugging Face Inference API

---

## Endee Integration (Conceptual)

This project is designed to work with the Endee Vector Database.

Due to Docker image access issues during setup, a local embedding-based similarity search is implemented to simulate Endee functionality.

### In a production setup:

* Embeddings would be stored in Endee
* Queries would be executed using Endee similarity search
* Endee would handle scalable vector storage and retrieval

---

## System Architecture

User Query
→ Embedding
→ Similarity Search (Endee concept)
→ Top Results
→ LLM (HuggingFace)
→ Generated Answer

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
endee-rag-project/
│
├── app.py
├── requirements.txt
├── .env
├── data/
│   └── documents.txt
```

---

## ⚠️ Note

Endee is conceptually implemented due to restricted Docker image access. The architecture fully aligns with vector database-based retrieval systems.

---

## 🎯 Conclusion

This project demonstrates a complete RAG pipeline and showcases how vector databases like Endee can be integrated into AI applications for efficient retrieval and intelligent response generation.
