"""
=============================================================
STEP 3: Retrieval Pipeline
=============================================================
This module handles:
- Taking a user query
- Embedding the query
- Searching the vector store for relevant chunks
- Re-ranking results for better accuracy
- Returning the best context for the LLM

Run this file standalone to test retrieval:
    python step3_retrieve.py
=============================================================
"""

import json
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb


# ── Configuration ──────────────────────────────────────────
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "jaysynth_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5                                    # Number of chunks to retrieve
RERANK = True                                # Enable cross-encoder reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── 3A: Initialize Retrieval Components ───────────────────
class JaysynthRetriever:
    """
    Retriever for Jaysynth product documentation.
    
    Features:
    - Semantic search using dense embeddings
    - Optional cross-encoder reranking for better precision
    - Metadata filtering (by page, section, etc.)
    - Context window assembly for the LLM
    """
    
    def __init__(
        self,
        chroma_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        use_reranker: bool = RERANK,
        rerank_model: str = RERANK_MODEL
    ):
        # Load vector store
        print("📦 Loading vector store...")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(collection_name)
        print(f"   Collection: {collection_name} ({self.collection.count()} docs)")
        
        # Load embedding model
        print("📦 Loading embedding model...")
        self.embed_model = SentenceTransformer(embedding_model)
        
        # Optionally load cross-encoder for reranking
        self.reranker = None
        if use_reranker:
            print("📦 Loading reranker model...")
            self.reranker = CrossEncoder(rerank_model)
        
        print("✅ Retriever initialized!\n")
    
    # ── 3B: Basic Semantic Search ─────────────────────────
    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform semantic search on the vector store.
        
        Args:
            query: User's question
            top_k: Number of results to return
            filter_metadata: Optional ChromaDB where filter
                e.g., {"page_number": 9} to only search page 9
        
        Returns:
            List of result dicts with text, metadata, and score
        """
        # Embed the query
        query_embedding = self.embed_model.encode(
            [query],
            normalize_embeddings=True
        )
        
        # Search ChromaDB
        search_kwargs = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k * 2 if self.reranker else top_k,  # Fetch more if reranking
        }
        if filter_metadata:
            search_kwargs["where"] = filter_metadata
        
        results = self.collection.query(**search_kwargs)
        
        # Format results
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else None,
                "id": results['ids'][0][i]
            })
        
        return formatted
    
    # ── 3C: Cross-Encoder Reranking ──────────────────────
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = TOP_K
    ) -> List[Dict]:
        """
        Rerank search results using a cross-encoder.
        
        Cross-encoders are more accurate than bi-encoders for
        ranking because they see query and document together.
        This is slower but significantly improves precision.
        """
        if not self.reranker or not results:
            return results[:top_k]
        
        # Create query-document pairs
        pairs = [(query, r["text"]) for r in results]
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Attach scores and sort
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:top_k]
    
    # ── 3D: Full Retrieval Pipeline ──────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        filter_metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict], str]:
        """
        Full retrieval pipeline:
        1. Semantic search
        2. Rerank (if enabled)
        3. Assemble context string for the LLM
        
        Returns:
            (results, context_string)
        """
        # Step 1: Search
        results = self.search(query, top_k, filter_metadata)
        
        # Step 2: Rerank
        if self.reranker:
            results = self.rerank(query, results, top_k)
        
        # Step 3: Assemble context
        context = self._assemble_context(results)
        
        return results, context
    
    # ── 3E: Context Assembly ─────────────────────────────
    def _assemble_context(self, results: List[Dict]) -> str:
        """
        Assemble retrieved chunks into a formatted context string
        that will be passed to the LLM.
        
        Includes source attribution for traceability.
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            section = result["metadata"].get("section_title", "N/A")
            page = result["metadata"].get("page_number", "N/A")
            
            context_parts.append(
                f"--- Source {i} (Page {page} | {section}) ---\n"
                f"{result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    # ── 3F: Debug / Inspection ───────────────────────────
    def explain_retrieval(self, query: str, top_k: int = TOP_K):
        """Print detailed retrieval results for debugging."""
        results, context = self.retrieve(query, top_k)
        
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")
        
        for i, r in enumerate(results, 1):
            score_info = ""
            if r.get("rerank_score") is not None:
                score_info = f" | Rerank: {r['rerank_score']:.4f}"
            if r.get("distance") is not None:
                score_info += f" | Distance: {r['distance']:.4f}"
            
            print(f"\n[{i}] Page {r['metadata']['page_number']} | "
                  f"{r['metadata']['section_title']}{score_info}")
            print(f"    {r['text'][:200]}...")
        
        print(f"\n📏 Total context length: {len(context)} characters")
        return results, context


# ── Sample Queries for Testing ─────────────────────────────
SAMPLE_QUERIES = [
    "What pigments are best for water-based flexo ink applications?",
    "Which Jaysynth products have heat stability above 280°C for plastics?",
    "Tell me about Pigmefine series for decorative paints",
    "What certifications does Jaysynth hold?",
    "What is Jaysynth's expertise in particle size distribution?",
    "Do you have pigments suitable for automotive coatings?",
    "What are the packaging options available?",
    "Which pigments work for both paint and plastic applications?",
]


# ── Main ──────────────────────────────────────────────────
def run_retrieval_tests():
    """Test retrieval with sample queries."""
    print("\n" + "="*60)
    print("STEP 3: Retrieval Pipeline Testing")
    print("="*60 + "\n")
    
    retriever = JaysynthRetriever(use_reranker=False)  # Set True if you have the model
    
    for query in SAMPLE_QUERIES[:3]:  # Test first 3
        retriever.explain_retrieval(query)
    
    return retriever


if __name__ == "__main__":
    retriever = run_retrieval_tests()
