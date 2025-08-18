from typing import List, Tuple
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# --- THIS IS THE FIX ---
# Corrected "flan-tT5-base" to "flan-t5-base"
GEN_MODEL_NAME = "google/flan-t5-base"

class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.generator = pipeline("text2text-generation", model=GEN_MODEL_NAME, device=-1)
        self.index = None
        self.chunks: List[str] = []

    def build_index(self, chunks: List[str]):
        """Builds a FAISS index from a list of text chunks."""
        self.chunks = chunks
        chunk_embeddings = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        dim = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(chunk_embeddings.astype('float32'))

    def retrieve(self, question: str, k: int) -> List[Tuple[str, float]]:
        """Retrieves the top-k most relevant chunks and their similarity scores."""
        if self.index is None or not self.chunks:
            return []
        
        question_embedding = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(question_embedding.astype('float32'), k)
        
        return [(self.chunks[idx], float(score)) for score, idx in zip(scores[0], idxs[0]) if 0 <= idx < len(self.chunks)]

    def generate(self, question: str, retrieved_chunks: List[str]) -> str:
        """Generates an answer based on the question and retrieved context."""
        context = "\n\n".join(retrieved_chunks)
        prompt = (
            "You are an expert assistant. Synthesize a comprehensive answer to the question based "
            "solely on the provided context. List all relevant items if the question asks for a list. "
            "If the context does not contain the answer, state that the answer is not available in the provided documents.\n\n"
            f"Context:\n---\n{context}\n---\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        output = self.generator(prompt, max_new_tokens=250, do_sample=False)
        return output[0]["generated_text"].strip()

    def generate_without_context(self, question: str) -> str:
        """Generates an answer using only the model's general knowledge."""
        prompt = f"Answer the following question based on your general knowledge.\n\nQuestion: {question}\n\nAnswer:"
        output = self.generator(prompt, max_new_tokens=200, do_sample=False)
        return output[0]["generated_text"].strip()

    def answer(self, question: str, top_k: int, relevance_threshold: float) -> Tuple[str, bool]:
        """
        Full RAG pipeline with a relevance gate.
        Only uses context if the retrieved chunks are above the similarity threshold.
        """
        hits = self.retrieve(question, k=top_k)
        
        if not hits or hits[0][1] < relevance_threshold:
            response = self.generate_without_context(question)
            return response, False
        
        retrieved_chunks = [h[0] for h in hits]
        return self.generate(question, retrieved_chunks), True