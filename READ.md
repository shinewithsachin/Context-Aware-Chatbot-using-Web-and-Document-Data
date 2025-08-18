# 🤖 End-to-End AI Chatbot with RAG (Streamlit)

**Tech**: Hugging Face Transformers (FLAN-T5 Small), Sentence Transformers (all-MiniLM-L6-v2), FAISS, Streamlit, pypdf.

## Features
- Document-grounded Q&A via **RAG** (retrieve top chunks + generate).
- Supports **TXT** and **PDF** uploads.
- Small, CPU-friendly models (free, no OpenAI needed).
- Streamlit UI, deployable on **Hugging Face Spaces**.

## Local Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
