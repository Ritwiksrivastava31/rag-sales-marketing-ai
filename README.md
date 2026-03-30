# RAG System — Sales & Marketing AI Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system built for **Jaysynth Orgochem Limited**, enabling intelligent Q&A about their colorants, pigments, dispersions, and digital ink products.

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Embedding   │───▶│   Vector Store   │───▶│   Retrieval     │
│  (Sentence   │    │   (ChromaDB /    │    │   (Top-K +      │
│  Transformer)│    │    FAISS)        │    │   Reranking)    │
└──────────────┘    └──────────────────┘    └────────┬────────┘
                                                      │
                                                      ▼ Context
                                            ┌─────────────────┐
                                            │   GPT API       │
                                            │   (Generation)  │
                                            └────────┬────────┘
                                                      │
                                                      ▼
                                              Sales Response
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Prepare Data

```bash
# Create data directory and place the PDF
mkdir -p data
cp /path/to/Jaysynth-details.pdf data/
```

### 4. Build the Index

```bash
# This runs Steps 1 & 2: Ingestion + Embedding
python step5_rag_pipeline.py --build
```

### 5. Start Chatting

```bash
# Interactive chat mode
python step5_rag_pipeline.py --chat

# Single query
python step5_rag_pipeline.py --query "What pigments for flexo inks?"

# Technical mode
python step5_rag_pipeline.py --query "Heat stability specs for plastics" --mode technical
```

---

## Project Structure

```
jaysynth_rag/
├── data/
│   ├── Jaysynth-details.pdf     # Source document
│   ├── chunks.json              # Generated chunks (after Step 1)
│   └── chroma_db/               # Vector store (after Step 2)
│
├── step1_ingest.py              # PDF → Text → Chunks
├── step2_embed.py               # Chunks → Embeddings → Vector Store
├── step3_retrieve.py            # Query → Relevant Chunks
├── step4_generate.py            # Context + Query → GPT Response
├── step5_rag_pipeline.py        # Complete pipeline (main entry)
├── step6_llamaindex_rag.py      # Alternative: LlamaIndex implementation
├── step7_evaluate.py            # Evaluation & tuning
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Step-by-Step Guide

| Step | File | What It Does |
|------|------|-------------|
| **1** | `step1_ingest.py` | Parses the PDF, cleans text, creates overlapping chunks with metadata |
| **2** | `step2_embed.py` | Embeds chunks using Sentence Transformers, stores in ChromaDB |
| **3** | `step3_retrieve.py` | Semantic search + optional cross-encoder reranking |
| **4** | `step4_generate.py` | Constructs prompts, calls GPT API with sales/marketing/technical modes |
| **5** | `step5_rag_pipeline.py` | Combines everything into a single interface |
| **6** | `step6_llamaindex_rag.py` | Alternative implementation using LlamaIndex framework |
| **7** | `step7_evaluate.py` | Evaluates retrieval quality with ground-truth test cases |

---

## Two Implementation Options

### Option A: Custom Pipeline (Steps 1–5)
- Full control over every component
- Easier to debug and customize
- Better for learning RAG internals

### Option B: LlamaIndex (Step 6)
- Less code, more abstraction
- Built-in chat memory and query modes
- Faster to prototype

---

## Tuning Tips

**If retrieval misses relevant chunks:**
- Decrease `CHUNK_SIZE` (try 300–400)
- Increase `CHUNK_OVERLAP` (try 150)
- Increase `TOP_K` (try 7–10)
- Enable cross-encoder reranking (`--rerank`)

**If responses are too generic:**
- Switch to `gpt-4o` for better quality
- Lower `TEMPERATURE` to 0.1–0.2
- Add more specific instructions to the system prompt

**If responses hallucinate:**
- Strengthen the "only use provided context" instruction
- Reduce `TEMPERATURE` to 0.0
- Add "If unsure, say so" to the prompt

---

## Next Steps (Agentic AI Roadmap)

Once the RAG foundation is solid, you can extend it to:

1. **Multi-document RAG** — Add TDS sheets, MSDS, pricing docs
2. **Conversational Memory** — Track customer context across sessions  
3. **Product Recommender Agent** — Ask application requirements, recommend products
4. **Email/Quote Generator** — Auto-draft responses to customer inquiries
5. **CRM Integration** — Connect to customer data for personalized interactions
6. **Analytics Dashboard** — Track common queries, gaps in documentation
