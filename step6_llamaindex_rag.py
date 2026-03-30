"""
=============================================================
ALTERNATIVE: LlamaIndex-Based RAG Pipeline
=============================================================
This is an alternative implementation using LlamaIndex,
which provides a higher-level abstraction over the same 
concepts (ingest → embed → retrieve → generate).

LlamaIndex handles a lot of the plumbing automatically.
Use this if you prefer less code and more built-in features.

Usage:
    python step6_llamaindex_rag.py
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LlamaIndex Imports ────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

# For ChromaDB integration
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore


# ── Configuration ──────────────────────────────────────────
PDF_DIR = "data/"                            # Directory containing the PDF
CHROMA_DIR = "data/llamaindex_chroma_db"
COLLECTION_NAME = "jaysynth_llamaindex"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GPT_MODEL = "gpt-4o-mini"


# ── Sales System Prompt ────────────────────────────────────
SALES_PROMPT = """You are a knowledgeable sales assistant for Jaysynth Orgochem Limited, 
a 50+ year old Indian company leading in colorants, pigments, and dispersions.

When answering:
- Use ONLY information from the provided context
- Recommend specific products with grades and C.I. numbers
- Be professional yet approachable
- Highlight relevant certifications (ISO, REACH, FDA) when appropriate
- If information isn't available in context, say so honestly
"""


def build_llamaindex_rag():
    """
    Build the complete LlamaIndex RAG pipeline.
    """
    print("\n" + "="*60)
    print("📦 Building LlamaIndex RAG Pipeline")
    print("="*60 + "\n")
    
    # ── Step 1: Configure LlamaIndex Settings ─────────────
    print("1️⃣  Configuring models...")
    
    # Set embedding model (Sentence Transformer)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )
    
    # Set LLM (OpenAI GPT)
    Settings.llm = LlamaOpenAI(
        model=GPT_MODEL,
        temperature=0.3,
        system_prompt=SALES_PROMPT
    )
    
    # Set chunking strategy
    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=100,
    )
    
    print("   ✅ Models configured\n")
    
    # ── Step 2: Load Documents ────────────────────────────
    print("2️⃣  Loading documents...")
    
    documents = SimpleDirectoryReader(
        input_dir=PDF_DIR,
        required_exts=[".pdf"],
        filename_as_id=True
    ).load_data()
    
    print(f"   ✅ Loaded {len(documents)} document pages\n")
    
    # ── Step 3: Setup ChromaDB Vector Store ───────────────
    print("3️⃣  Setting up vector store...")
    
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete if exists (for rebuilding)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print("   ✅ ChromaDB ready\n")
    
    # ── Step 4: Build Index ───────────────────────────────
    print("4️⃣  Building vector index (this may take a minute)...")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Save storage context for later loading
    index.storage_context.persist(persist_dir=CHROMA_DIR + "_storage")
    
    print(f"   ✅ Index built and persisted\n")
    
    return index


def load_existing_index():
    """Load a previously built index."""
    print("📦 Loading existing index...")
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL
    )
    Settings.llm = LlamaOpenAI(
        model=GPT_MODEL,
        temperature=0.3,
        system_prompt=SALES_PROMPT
    )
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    print("✅ Index loaded\n")
    
    return index


def query_rag(index, question: str, top_k: int = 5) -> str:
    """
    Query the RAG system.
    
    LlamaIndex's query engine handles:
    - Embedding the query
    - Retrieving top-k chunks
    - Passing context to the LLM
    - Generating the response
    
    All in one call!
    """
    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",  # Options: "compact", "tree_summarize", "refine"
    )
    
    response = query_engine.query(question)
    
    # Extract source information
    sources = []
    for node in response.source_nodes:
        score = node.score if node.score else "N/A"
        page = node.metadata.get("page_label", "?")
        sources.append(f"Page {page} (relevance: {score:.4f})" if score != "N/A" 
                      else f"Page {page}")
    
    source_text = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(sources))
    
    return f"{response.response}\n\n📚 Sources:\n{source_text}"


# ── Interactive Chat with LlamaIndex ──────────────────────
def interactive_chat(index):
    """Start an interactive chat session."""
    print("\n" + "="*60)
    print("💬 Jaysynth Sales Assistant (LlamaIndex)")
    print("   Type 'quit' to exit")
    print("="*60 + "\n")
    
    # For multi-turn, use chat engine instead
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=5,
        system_prompt=SALES_PROMPT
    )
    
    while True:
        try:
            query = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        response = chat_engine.chat(query)
        print(f"\n🤖 Assistant: {response.response}\n")
        print("-" * 40 + "\n")


# ── Main ──────────────────────────────────────────────────
def main():
    import sys
    
    if "--build" in sys.argv:
        index = build_llamaindex_rag()
    else:
        try:
            index = load_existing_index()
        except Exception:
            print("❌ No existing index found. Building from scratch...")
            index = build_llamaindex_rag()
    
    # Test queries
    test_questions = [
        "What pigments are suitable for water-based flexo inks?",
        "Which products have heat stability above 280°C?",
        "Tell me about your certifications and compliance.",
    ]
    
    if "--chat" in sys.argv:
        interactive_chat(index)
    else:
        for question in test_questions:
            print(f"\n🔍 Q: {question}")
            print("-" * 40)
            answer = query_rag(index, question)
            print(f"💬 A: {answer}\n")


if __name__ == "__main__":
    main()
