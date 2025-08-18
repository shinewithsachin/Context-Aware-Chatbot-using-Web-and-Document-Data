import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from utils import clean_text, chunk_text, load_url, load_txt, load_pdf
from rag_pipeline import RAGPipeline
from chatbot_core import smalltalk

# --- Page Configuration ---
st.set_page_config(page_title="Robust AI Chatbot with RAG", page_icon="🚀", layout="wide")

st.title("🚀  Context-Aware Chatbot using Web and Document Data")
st.caption("Models: FLAN-T5 Base • BGE-Base-EN-v1.5 • DB: FAISS")

# --- Session State and Resource Caching ---
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline()

st.session_state.setdefault("rag", get_rag_pipeline())
st.session_state.setdefault("kb_built", False)
st.session_state.setdefault("history", [])
st.session_state.setdefault("active_kb_source", "None")

# --- Sidebar ---
with st.sidebar:
    # (The Knowledge Base building sections for URL and File Upload remain the same)
    st.header("⚙️ Knowledge Base Configuration")
    url_input = st.text_input("Enter a URL to scrape:", placeholder="https://example.com")
    if st.button("Build KB from URL"):
        # ... (no changes here)
        if not url_input:
            st.warning("Please enter a URL.")
        else:
            with st.spinner(f"Processing: {url_input}"):
                txt = load_url(url_input)
                if txt:
                    cleaned_txt = clean_text(txt)
                    chunks = chunk_text(cleaned_txt, st.session_state.chunk_size, st.session_state.overlap)
                    st.session_state.rag.build_index(chunks)
                    st.session_state.kb_built = True
                    st.session_state.active_kb_source = f"URL: {url_input[:50]}..."
                    st.success(f"KB built from URL with {len(chunks)} chunks.")
    
    st.markdown("<p style='text-align: center; font-weight: bold;'>OR</p>", unsafe_allow_html=True)
    
    uploads = st.file_uploader("Upload TXT/PDF documents", type=["txt", "pdf"], accept_multiple_files=True)
    if st.button("Build KB from Files"):
        # ... (no changes here)
        if not uploads:
            st.warning("Please upload at least one file.")
        else:
            with st.spinner("Processing Files..."):
                # ... (rest of the file processing logic is unchanged)
                chunks_all = []
                for f in uploads:
                    data = f.read()
                    if f.name.lower().endswith(".txt"): txt = load_txt(data)
                    else: txt = load_pdf(data)
                    cleaned = clean_text(txt)
                    chunks_all.extend(chunk_text(cleaned, st.session_state.chunk_size, st.session_state.overlap))
                
                if chunks_all:
                    st.session_state.rag.build_index(chunks_all)
                    st.session_state.kb_built = True
                    st.session_state.active_kb_source = f"{len(uploads)} file(s) uploaded"
                    st.success(f"KB built from {len(uploads)} file(s) with {len(chunks_all)} chunks.")

    st.divider()
    st.subheader("RAG Parameters")
    st.session_state.chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    st.session_state.overlap = st.slider("Overlap", 0, 200, 100, 25)
    st.session_state.top_k = st.slider("Top-K Chunks", 1, 10, 4, 1)

    # --- KEY CHANGE: Added a slider for the relevance threshold ---
    st.session_state.relevance_threshold = st.slider(
        "Relevance Threshold", 0.0, 1.0, 0.35, 0.05,
        help="Chunks below this similarity score will be ignored. Higher values make the chatbot stricter about using context."
    )
    
    st.divider()
    st.info(f"**Active KB:** {st.session_state.active_kb_source}")
    use_rag = st.checkbox("Use RAG (answers from source)", value=True)

# --- Main Chat Interface ---
st.subheader("💬 Chat")
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ""
            smalltalk_response = smalltalk(prompt)
            
            if smalltalk_response:
                response = smalltalk_response
            elif use_rag and st.session_state.kb_built:
                # --- KEY CHANGE: Pass the relevance threshold to the answer method ---
                response, rag_used = st.session_state.rag.answer(prompt, st.session_state.top_k, st.session_state.relevance_threshold)
                if not rag_used:
                    response += "\n\n*(Note: No relevant context found in the source. Replying from general knowledge.)*"
            else:
                response = st.session_state.rag.generate_without_context(prompt)
            
            st.markdown(response)
    
    st.session_state.history.append({"role": "assistant", "content": response})