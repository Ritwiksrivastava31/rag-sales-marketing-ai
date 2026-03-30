"""
=============================================================
STEP 1: Document Ingestion & Chunking
=============================================================
This module handles:
- Reading the Pigment Company PDF document
- Extracting text from each page
- Cleaning and preprocessing the text
- Splitting into smart chunks that preserve context

Run this file standalone to test ingestion:
    python step1_ingest.py
=============================================================
"""

import pdfplumber
import re
import json
from pathlib import Path
from typing import List, Dict


# ── Configuration ──────────────────────────────────────────
PDF_PATH = "data/Pigment Company-details.pdf"       # Place your PDF here
CHUNKS_OUTPUT = "data/chunks.json"           # Output chunked data
CHUNK_SIZE = 500                             # Characters per chunk
CHUNK_OVERLAP = 100                          # Overlap between chunks


# ── 1A: Extract Raw Text from PDF ─────────────────────────
def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from each page of the PDF.
    Returns a list of dicts with page number and text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page_number": i + 1,
                    "raw_text": text.strip()
                })
    
    print(f"✅ Extracted text from {len(pages)} pages")
    return pages


# ── 1B: Clean and Preprocess Text ─────────────────────────
def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Remove excessive whitespace
    - Fix common OCR artifacts
    - Normalize line breaks
    """
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines at start and end
    text = text.strip()
    
    return text


# ── 1C: Smart Chunking Strategy ───────────────────────────
def create_chunks(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split document into overlapping chunks while preserving context.
    
    Strategy:
    - Each chunk includes metadata (page number, section title)
    - Chunks respect paragraph boundaries where possible
    - Overlap ensures no information is lost at boundaries
    """
    chunks = []
    chunk_id = 0
    
    for page_data in pages:
        page_num = page_data["page_number"]
        text = clean_text(page_data["raw_text"])
        
        # Try to detect section title (first line of page)
        lines = text.split('\n')
        section_title = lines[0] if lines else "Unknown Section"
        
        # If the page text is short enough, keep it as one chunk
        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "page_number": page_num,
                    "section_title": section_title,
                    "source": "Pigment Company-details.pdf"
                }
            })
            chunk_id += 1
            continue
        
        # Split into paragraphs first, then chunk
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size, save current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "metadata": {
                        "page_number": page_num,
                        "section_title": section_title,
                        "source": "Pigment Company-details.pdf"
                    }
                })
                chunk_id += 1
                
                # Keep overlap from end of current chunk
                if chunk_overlap > 0:
                    current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "metadata": {
                    "page_number": page_num,
                    "section_title": section_title,
                    "source": "Pigment Company-details.pdf"
                }
            })
            chunk_id += 1
    
    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


# ── 1D: Save Chunks to JSON ───────────────────────────────
def save_chunks(chunks: List[Dict], output_path: str):
    """Save chunks to JSON file for inspection and later use."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Chunks saved to {output_path}")


# ── Main ──────────────────────────────────────────────────
def run_ingestion(pdf_path: str = PDF_PATH) -> List[Dict]:
    """Complete ingestion pipeline."""
    print("\n" + "="*60)
    print("STEP 1: Document Ingestion & Chunking")
    print("="*60)
    
    # Extract
    pages = extract_text_from_pdf(pdf_path)
    
    # Chunk
    chunks = create_chunks(pages)
    
    # Save for inspection
    save_chunks(chunks, CHUNKS_OUTPUT)
    
    # Print sample
    print(f"\n📄 Sample chunk (chunk_id=0):")
    print(f"   Section: {chunks[0]['metadata']['section_title']}")
    print(f"   Text preview: {chunks[0]['text'][:200]}...")
    
    return chunks


if __name__ == "__main__":
    chunks = run_ingestion()
    print(f"\n🎉 Ingestion complete! Total chunks: {len(chunks)}")
