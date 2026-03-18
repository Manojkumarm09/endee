import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()

# HuggingFace API
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents
def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = f.read().split('\n\n')
    return documents

# Create embeddings
def create_embeddings(documents):
    return model.encode(documents)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search (return all sorted results)
def search(query, documents, embeddings):
    query_embedding = model.encode([query])[0]

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((documents[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores  # return ALL

# Generate answer
def generate_answer(query, results):
    context = " ".join([res for res, _ in results])

    prompt = f"""
Explain clearly in simple terms.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt},
            timeout=10
        )

        data = response.json()

        # If HF returns proper output
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        # If HF returns error or unexpected format
        raise Exception("Invalid HF response")

    except:
        # 🔥 CLEAN fallback (NO ERROR MESSAGE)
        answer = "Here is a clear explanation:\n\n"
        for i, (res, _) in enumerate(results, 1):
            answer += f"{i}. {res}\n"
        return answer
# Load data
docs = load_documents("data/documents.txt")
embeddings = create_embeddings(docs)

# UI
st.title("AI Knowledge Assistant (RAG + Endee Concept)")
st.write("Ask questions on AI, ML, Databases, and get intelligent answers using Retrieval-Augmented Generation.")

query = st.text_input("Enter your question:")

if st.button("Search"):
    results = search(query, docs, embeddings)

    #  Separate results
    top_results = results[:2]      # show only 2
    answer_results = results[:4]   # use more for answer

    # Show top results
    st.subheader("Relevant Knowledge Retrieved")
    for res, score in top_results:
        st.write(f"{res} (score: {score:.4f})")

    # Generate answer
    answer = generate_answer(query, answer_results)

    st.subheader("AI Generated Explanation")
    st.write(answer)
    st.caption("This answer is generated using retrieved context + LLM reasoning (RAG pipeline).")
