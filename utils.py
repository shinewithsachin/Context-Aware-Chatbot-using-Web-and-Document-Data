import re
from pypdf import PdfReader
from typing import List
from io import BytesIO
import trafilatura
import streamlit as st

def clean_text(text: str) -> str:
    """Removes null bytes and extra whitespace from text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into chunks of words with a specified overlap."""
    if overlap >= chunk_size:
        st.warning(f"Overlap ({overlap}) is greater than chunk size ({chunk_size}). Setting overlap to 0.")
        overlap = 0

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_end = i + chunk_size
        chunk = words[i:chunk_end]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    
    return [c for c in chunks if c]

def load_txt(bytes_data: bytes) -> str:
    """Loads text from bytes, ignoring errors."""
    return bytes_data.decode("utf-8", errors="ignore")

def load_pdf(bytes_data: bytes) -> str:
    """Extracts text from a PDF provided as bytes."""
    reader = PdfReader(BytesIO(bytes_data))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)

def load_url(url: str) -> str:
    """
    Fetches and extracts the main article text from a URL using trafilatura.
    Returns an empty string if fetching or extraction fails.
    """
    try:
        # Download the HTML content
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            st.error(f"Failed to fetch content from URL: {url}. The site may be blocking scrapers.")
            return ""
        
        # Extract the main text content
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if text is None:
            st.warning("Could not extract main content from the page. The content might be rendered by JavaScript.")
            return ""
            
        return text
    except Exception as e:
        st.error(f"An error occurred while processing the URL: {e}")
        return ""