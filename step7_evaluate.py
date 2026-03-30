"""
=============================================================
STEP 7: RAG Evaluation
=============================================================
This module helps you evaluate and tune your RAG system.

It tests:
- Retrieval quality: Are the right chunks being found?
- Answer quality: Is the LLM giving accurate responses?
- End-to-end: Does the full pipeline work well?

Usage:
    python step7_evaluate.py
=============================================================
"""

import json
from typing import List, Dict
from step3_retrieve import Pigment CompanyRetriever


# ── Evaluation Test Cases ──────────────────────────────────
# These are ground-truth Q&A pairs based on the Pigment Company PDF.
# You should expand this list as you refine the system.

EVAL_CASES = [
    {
        "query": "What year was Pigment Company founded?",
        "expected_answer_contains": ["1971"],
        "expected_pages": [2, 3],
        "category": "company_info"
    },
    {
        "query": "What is the heat stability of Green GFP M-900 for plastic moldings?",
        "expected_answer_contains": ["300"],
        "expected_pages": [10],
        "category": "product_specs"
    },
    {
        "query": "What pigments are available for solvent-based inks?",
        "expected_answer_contains": ["Beta Blue BFLX", "I-705"],
        "expected_pages": [14],
        "category": "product_recommendation"
    },
    {
        "query": "What certifications does Pigment Company have?",
        "expected_answer_contains": ["ISO", "REACH", "FDA"],
        "expected_pages": [48, 43],
        "category": "compliance"
    },
    {
        "query": "What is Pigmefine used for?",
        "expected_answer_contains": ["Decorative", "Paint"],
        "expected_pages": [24, 25],
        "category": "product_info"
    },
    {
        "query": "What particle size can Pigment Company achieve?",
        "expected_answer_contains": ["200", "nano"],
        "expected_pages": [3, 7],
        "category": "technical"
    },
    {
        "query": "Which pigments are non-crystalline and non-flocculating?",
        "expected_answer_contains": ["Alpha Blue AC", "P-861"],
        "expected_pages": [12],
        "category": "product_specs"
    },
    {
        "query": "What packaging sizes are available for pigment powders?",
        "expected_answer_contains": ["20 Kg", "25 Kg"],
        "expected_pages": [47],
        "category": "logistics"
    },
]


# ── 7A: Retrieval Evaluation ──────────────────────────────
def evaluate_retrieval(retriever: Pigment CompanyRetriever, top_k: int = 5) -> Dict:
    """
    Evaluate retrieval quality.
    
    Metrics:
    - Hit Rate: % of queries where at least one expected page appears in top-k
    - MRR (Mean Reciprocal Rank): Average 1/rank of first relevant result
    - Content Match: % of queries where expected keywords appear in retrieved text
    """
    results_summary = {
        "total_queries": len(EVAL_CASES),
        "hit_rate": 0,
        "mrr": 0,
        "content_match_rate": 0,
        "details": []
    }
    
    hits = 0
    reciprocal_ranks = []
    content_matches = 0
    
    for case in EVAL_CASES:
        query = case["query"]
        expected_pages = set(case["expected_pages"])
        expected_keywords = case["expected_answer_contains"]
        
        # Retrieve
        search_results, context = retriever.retrieve(query, top_k=top_k)
        
        # Check page hit
        retrieved_pages = [r["metadata"]["page_number"] for r in search_results]
        page_hit = bool(expected_pages.intersection(retrieved_pages))
        
        if page_hit:
            hits += 1
            # Find rank of first relevant page
            for rank, page in enumerate(retrieved_pages, 1):
                if page in expected_pages:
                    reciprocal_ranks.append(1.0 / rank)
                    break
        else:
            reciprocal_ranks.append(0.0)
        
        # Check content match
        context_lower = context.lower()
        keyword_found = any(kw.lower() in context_lower for kw in expected_keywords)
        if keyword_found:
            content_matches += 1
        
        # Record details
        results_summary["details"].append({
            "query": query,
            "category": case["category"],
            "page_hit": page_hit,
            "content_match": keyword_found,
            "expected_pages": list(expected_pages),
            "retrieved_pages": retrieved_pages,
            "expected_keywords": expected_keywords,
        })
    
    # Calculate metrics
    results_summary["hit_rate"] = hits / len(EVAL_CASES)
    results_summary["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    results_summary["content_match_rate"] = content_matches / len(EVAL_CASES)
    
    return results_summary


# ── 7B: Print Evaluation Report ───────────────────────────
def print_eval_report(results: Dict):
    """Pretty print the evaluation results."""
    print("\n" + "="*60)
    print("📊 RAG Retrieval Evaluation Report")
    print("="*60)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Hit Rate:          {results['hit_rate']:.1%}")
    print(f"   MRR:               {results['mrr']:.3f}")
    print(f"   Content Match:     {results['content_match_rate']:.1%}")
    print(f"   Total Queries:     {results['total_queries']}")
    
    print(f"\n📋 Per-Query Details:")
    print(f"   {'Query':<55} {'Pages':>6} {'Content':>8}")
    print(f"   {'-'*55} {'-'*6} {'-'*8}")
    
    for detail in results["details"]:
        query_short = detail["query"][:52] + "..." if len(detail["query"]) > 55 else detail["query"]
        page_mark = "✅" if detail["page_hit"] else "❌"
        content_mark = "✅" if detail["content_match"] else "❌"
        print(f"   {query_short:<55} {page_mark:>6} {content_mark:>8}")
    
    # Category breakdown
    categories = {}
    for detail in results["details"]:
        cat = detail["category"]
        if cat not in categories:
            categories[cat] = {"hit": 0, "total": 0}
        categories[cat]["total"] += 1
        if detail["page_hit"]:
            categories[cat]["hit"] += 1
    
    print(f"\n📂 By Category:")
    for cat, counts in categories.items():
        rate = counts["hit"] / counts["total"]
        print(f"   {cat:<25} {rate:.0%} ({counts['hit']}/{counts['total']})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if results["hit_rate"] < 0.7:
        print("   ⚠️  Hit rate is low. Consider:")
        print("      - Reducing chunk size for more granular retrieval")
        print("      - Increasing top_k to retrieve more candidates")
        print("      - Enabling cross-encoder reranking")
    if results["content_match_rate"] < 0.8:
        print("   ⚠️  Content match rate could be improved. Consider:")
        print("      - Using a better embedding model (e.g., multi-qa-mpnet-base-dot-v1)")
        print("      - Adding metadata-enhanced embeddings")
    if results["hit_rate"] >= 0.8 and results["content_match_rate"] >= 0.8:
        print("   ✅ Retrieval performance looks good!")
    
    print()


# ── 7C: Tuning Experiments ────────────────────────────────
def run_tuning_experiments():
    """
    Run retrieval with different settings to find the best configuration.
    Uncomment and modify as needed.
    """
    print("\n" + "="*60)
    print("🔧 Running Tuning Experiments")
    print("="*60)
    
    # Experiment 1: Different top_k values
    retriever = Pigment CompanyRetriever(use_reranker=False)
    
    for top_k in [3, 5, 7, 10]:
        results = evaluate_retrieval(retriever, top_k=top_k)
        print(f"\n   top_k={top_k}: Hit Rate={results['hit_rate']:.1%}, "
              f"MRR={results['mrr']:.3f}, "
              f"Content Match={results['content_match_rate']:.1%}")


# ── Main ──────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("STEP 7: RAG Evaluation")
    print("="*60 + "\n")
    
    # Initialize retriever
    retriever = Pigment CompanyRetriever(use_reranker=False)
    
    # Run evaluation
    results = evaluate_retrieval(retriever, top_k=5)
    
    # Print report
    print_eval_report(results)
    
    # Save results
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📁 Results saved to data/eval_results.json")


if __name__ == "__main__":
    main()
