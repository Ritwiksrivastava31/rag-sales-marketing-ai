"""
=============================================================
STEP 5: Complete RAG Pipeline
=============================================================
This is the main entry point that combines all steps:
- Step 1: Ingest PDF → chunks
- Step 2: Embed chunks → vector store
- Step 3: Retrieve relevant context
- Step 4: Generate response with GPT

Usage:
    # First time (builds the index):
    python step5_rag_pipeline.py --build

    # Query mode (interactive chat):
    python step5_rag_pipeline.py --chat

    # Single query:
    python step5_rag_pipeline.py --query "What pigments for flexo inks?"
=============================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Import our modules
from step1_ingest import run_ingestion
from step2_embed import run_embedding, load_chunks
from step3_retrieve import Pigment CompanyRetriever
from step4_generate import Pigment CompanyGenerator, format_response_with_sources


# ── Configuration ──────────────────────────────────────────
PDF_PATH = "data/Pigment Company-details.pdf"
CHROMA_DIR = "data/chroma_db"


# ── 5A: Full RAG Chain ────────────────────────────────────
class Pigment CompanyRAG:
    """
    Complete RAG system for Pigment Company Orgochem.
    
    Combines retrieval and generation into a single
    easy-to-use interface.
    """
    
    def __init__(
        self,
        use_reranker: bool = False,
        gpt_model: str = "gpt-4o-mini",
        mode: str = "sales"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            use_reranker: Enable cross-encoder reranking (slower but more accurate)
            gpt_model: OpenAI model to use
            mode: Default response style (sales/marketing/technical)
        """
        self.mode = mode
        
        print("\n" + "="*60)
        print("🚀 Initializing Pigment Company RAG System")
        print("="*60 + "\n")
        
        # Initialize retriever
        self.retriever = Pigment CompanyRetriever(use_reranker=use_reranker)
        
        # Initialize generator
        try:
            self.generator = Pigment CompanyGenerator(model=gpt_model)
        except ValueError as e:
            print(f"⚠️  Generator not available: {e}")
            print("   RAG will work in retrieval-only mode.\n")
            self.generator = None
        
        print("\n✅ Pigment Company RAG System ready!\n")
    
    def ask(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        show_sources: bool = True,
        stream: bool = False
    ) -> str:
        """
        Ask a question and get a RAG-powered response.
        
        Args:
            query: Your question about Pigment Company products
            top_k: Number of context chunks to retrieve
            mode: Override default mode (sales/marketing/technical)
            show_sources: Include source references in response
            stream: Stream the response token by token
        
        Returns:
            Generated response string
        """
        mode = mode or self.mode
        
        # Step 1: Retrieve relevant context
        print(f"🔍 Retrieving context for: '{query}'")
        results, context = self.retriever.retrieve(query, top_k=top_k)
        
        if not results:
            return "I couldn't find relevant information in the Pigment Company documentation for this query."
        
        print(f"   Found {len(results)} relevant chunks\n")
        
        # Step 2: Generate response
        if self.generator is None:
            # Fallback: return raw retrieved context
            return f"📄 Retrieved Context (no LLM available):\n\n{context}"
        
        print(f"💬 Generating {mode} response...")
        answer = self.generator.generate(
            query=query,
            context=context,
            mode=mode,
            stream=stream
        )
        
        # Step 3: Add sources if requested
        if show_sources:
            answer = format_response_with_sources(answer, results)
        
        return answer
    
    def interactive_chat(self):
        """
        Start an interactive chat session.
        Type 'quit' to exit, 'reset' to clear history,
        'mode:sales/marketing/technical' to switch modes.
        """
        print("\n" + "="*60)
        print("💬 Pigment Company Sales Assistant - Interactive Chat")
        print("="*60)
        print("Commands: 'quit' | 'reset' | 'mode:sales' | 'mode:technical' | 'mode:marketing'")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'reset':
                if self.generator:
                    self.generator.reset_conversation()
                continue
            
            if query.lower().startswith('mode:'):
                self.mode = query.split(':')[1].strip()
                print(f"   Switched to {self.mode} mode\n")
                continue
            
            # Get response
            answer = self.ask(query, stream=False)
            print(f"\n🤖 Pigment Company Assistant:\n{answer}\n")
            print("-" * 40 + "\n")


# ── 5B: Build Index from Scratch ──────────────────────────
def build_index(pdf_path: str = PDF_PATH):
    """
    Build the complete vector store from the PDF.
    Run this once, or whenever the PDF is updated.
    """
    print("\n" + "="*60)
    print("📦 Building Pigment Company RAG Index")
    print("="*60 + "\n")
    
    # Check PDF exists
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found at: {pdf_path}")
        print(f"   Please place 'Pigment Company-details.pdf' in the data/ directory")
        return False
    
    # Step 1: Ingest
    chunks = run_ingestion(pdf_path)
    
    # Step 2: Embed & Store
    client, collection, model = run_embedding(chunks)
    
    print("\n" + "="*60)
    print(f"🎉 Index built successfully!")
    print(f"   Total chunks indexed: {collection.count()}")
    print(f"   Vector store location: {CHROMA_DIR}")
    print("="*60 + "\n")
    
    return True


# ── Main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pigment Company RAG System")
    parser.add_argument('--build', action='store_true',
                        help='Build the vector index from PDF')
    parser.add_argument('--chat', action='store_true',
                        help='Start interactive chat')
    parser.add_argument('--query', type=str,
                        help='Ask a single question')
    parser.add_argument('--mode', type=str, default='sales',
                        choices=['sales', 'marketing', 'technical'],
                        help='Response mode')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='GPT model to use')
    parser.add_argument('--rerank', action='store_true',
                        help='Enable cross-encoder reranking')
    
    args = parser.parse_args()
    
    # Build index
    if args.build:
        build_index()
        return
    
    # Check if index exists
    if not Path(CHROMA_DIR).exists():
        print("❌ Vector store not found. Run with --build first:")
        print("   python step5_rag_pipeline.py --build")
        return
    
    # Initialize RAG
    rag = Pigment CompanyRAG(
        use_reranker=args.rerank,
        gpt_model=args.model,
        mode=args.mode
    )
    
    # Single query mode
    if args.query:
        answer = rag.ask(args.query)
        print(f"\n{answer}")
        return
    
    # Interactive chat mode
    if args.chat:
        rag.interactive_chat()
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
