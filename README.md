# 🚀 Context-Aware-Chatbot-using-Web-and-Document-Data
An end-to-end AI chatbot application built with Streamlit that provides context-aware answers by grounding a Large Language Model (LLM) in user-provided knowledge sources. This project uses a complete Retrieval-Augmented Generation (RAG) pipeline to answer questions from web pages or uploaded documents.

---

---

### 🧠 Project Overview

Standard Large Language Models can hallucinate or provide answers based on outdated information. Retrieval-Augmented Generation (RAG) solves this by providing the model with relevant, up-to-date context before it answers a question. This project implements a full RAG pipeline and provides:

*   A user-friendly interface to ingest knowledge from URLs or documents (PDF/TXT).
*   An intelligent "relevance gate" to decide when to use RAG vs. general knowledge.
*   Tunable parameters for the RAG process (chunking, retrieval top-k).

---

### 🗃 Knowledge Source Information

Unlike projects with a fixed dataset, this application builds its knowledge base dynamically from user-provided sources.

*   **URL Input**: Scrapes the main content from a given web page.
*   **Document Upload**: Extracts text from user-uploaded `.pdf` or `.txt` files.

The text from these sources is cleaned, chunked, and converted into vector embeddings to power the retrieval system.

---

### 🔍 Technologies Used

*   **Frontend**: Streamlit
*   **Backend Models**: Hugging Face Transformers (`google/flan-t5-base`, `BAAI/bge-base-en-v1.5`)
*   **Vector DB & Retrieval**: FAISS (CPU), Sentence-Transformers
*   **Data Processing**: Trafilatura, PyPDF, NumPy
*   **Deployment**: Hugging Face Spaces

---

### 🏗 Project Structure
├── app.py # Streamlit app interface

├── rag_pipeline.py # Core RAG pipeline logic

├── utils.py # Helper functions for data loading & processing

├── chatbot_core.py # Smalltalk and basic chat logic

├── requirements.txt # Python dependencies for deployment

├── .gitignore # Files and folders to ignore


---

### 💻 Getting Started Locally

**1. Clone the Repository**

https://github.com/shinewithsachin/Context-Aware-Chatbot-using-Web-and-Document-Data

**2. Create and activate a virtual environment (e.g., venv)**
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.\.venv\Scripts\activate`

**3. Install dependencies**
pip install -r requirements.txt```

**4. Run the Streamlit App**
streamlit run app.py

---

### 🧪 Core Component Details

* Architecture: Retrieval-Augmented Generation (RAG)
* Frameworks: PyTorch, Hugging Face Transformers
* Generator Model: - google/flan-t5-base
* Embedding Model: - BAAI/bge-base-en-v1.5
* Vector Index: - faiss.IndexFlatIP
* Web Scraper: - trafilatura


## Project Preview



![Input-Output](https://github.com/shinewithsachin/Context-Aware-Chatbot-using-Web-and-Document-Data/blob/main/Screenshot%202025-08-18%20175736.png)


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


