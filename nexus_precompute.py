#!/usr/bin/env python3
"""
nexus_precompute.py — NEXUS Offline Precomputation

Builds semantic embeddings using sentence-transformers (all-MiniLM-L6-v2).
Falls back to TF-IDF if sentence-transformers is unavailable.

Produces artifacts/nexus/:
  candidate_embeddings.npy  — (N, dim) float32 semantic vectors
  candidate_ids.json         — ordered list of N candidate IDs
  jd_embedding.npy           — (1, dim) JD vector
  jd_analysis.json           — parsed JD with skill cluster weights
  tfidf_vectorizer.pkl       — TF-IDF vectorizer (always built as fallback)
  tfidf_matrix.npz           — sparse TF-IDF matrix
  tfidf_jd_vec.npz           — JD TF-IDF vector

Usage:
  python nexus_precompute.py --candidates ./candidates.jsonl
  python nexus_precompute.py --candidates ./sample_candidates.json --fast
"""

import argparse, gzip, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "nexus"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# JD text and analysis
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI production embeddings retrieval systems
sentence transformers dense retrieval semantic search vector database FAISS Pinecone
Weaviate Qdrant Milvus OpenSearch Elasticsearch hybrid search BM25 reranking cross encoder
Python ranking evaluation NDCG MRR MAP offline evaluation online AB testing
LLM large language model RAG retrieval augmented generation fine tuning LoRA QLoRA PEFT
learning to rank XGBoost neural ranking information retrieval NLP natural language processing
recommendation system production deployment real users scale product company startup
applied machine learning MLOps inference optimization distributed systems
PyTorch transformers HuggingFace Weights Biases MLflow experiment tracking
embedding drift index refresh retrieval quality regression vector index
5 to 9 years experience Pune Noida Delhi NCR Mumbai Bangalore Hyderabad India
founding team startup early stage series A talent intelligence platform
shipped production system real users embeddings at scale search quality
"""

JD_ANALYSIS = {
    "role": "Senior AI Engineer — Founding Team @ Redrob AI",
    "yoe_range": [5, 9],
    "preferred_locations": ["pune", "noida", "delhi", "ncr", "mumbai", "bangalore", "hyderabad"],
    "company_type": "product_startup",
    "skill_cluster_emphasis": {
        "vector_retrieval": 0.28,
        "llm_rag": 0.24,
        "ml_training": 0.20,
        "ml_production": 0.18,
        "python_data": 0.10,
    },
    "hard_disqualifiers": [
        "consulting_only_career", "cv_speech_robotics_specialist",
        "management_track_no_code", "pure_research_no_prod"
    ],
    "strong_signals": [
        "shipped_ranking_search_to_prod", "vector_db_in_prod",
        "founding_team_startup", "active_github", "product_company_ml"
    ],
}


# ---------------------------------------------------------------------------
# Candidate text builder
# ---------------------------------------------------------------------------

def build_candidate_text(c: dict) -> str:
    p = c.get("profile", {})
    parts = []

    title = p.get("current_title", "")
    company = p.get("current_company", "")
    industry = p.get("current_industry", "")
    # Double title for TF-IDF weight
    parts.append(f"{title} {title} {company} {industry}")
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))

    for job in c.get("career_history", [])[:5]:
        jt = job.get("title", "")
        jc = job.get("company", "")
        ji = job.get("industry", "")
        dur = job.get("duration_months", 0)
        desc = job.get("description", "")
        if dur >= 24:
            parts.append(f"{jt} {jt} {jc} {ji}")
        else:
            parts.append(f"{jt} {jc} {ji}")
        if desc:
            parts.append(desc[:500])

    skill_tokens = []
    for sk in c.get("skills", []):
        name = sk.get("name", "")
        prof = sk.get("proficiency", "intermediate")
        dur = sk.get("duration_months", 0)
        endrs = sk.get("endorsements", 0)
        reps = 1
        if prof in ("advanced", "expert"):
            reps += 1
        if dur >= 24:
            reps += 1
        if endrs >= 20:
            reps += 1
        skill_tokens.extend([name] * reps)
    if skill_tokens:
        parts.append(" ".join(skill_tokens))

    for edu in c.get("education", [])[:2]:
        parts.append(f"{edu.get('degree','')} {edu.get('field_of_study','')} {edu.get('institution','')}")

    for cert in c.get("certifications", [])[:3]:
        parts.append(f"{cert.get('name','')} {cert.get('issuer','')}")

    return " ".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    candidates = []
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        candidates = data if isinstance(data, list) else [data]
    elif p.suffix == ".gz":
        with gzip.open(p, "rt") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
                if (i + 1) % 10000 == 0:
                    print(f"  Loaded {i+1:,}...", flush=True)
    else:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    return candidates


# ---------------------------------------------------------------------------
# Sentence-transformer encoding
# ---------------------------------------------------------------------------

def encode_with_st(texts: list[str], jd_text: str, batch_size: int = 512):
    """Returns (candidate_matrix np.ndarray, jd_vector np.ndarray) or None."""
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Encoding JD...")
        jd_vec = model.encode([jd_text], batch_size=1, show_progress_bar=False,
                               normalize_embeddings=True)
        print(f"  Encoding {len(texts):,} candidates (batch={batch_size})...")
        cand_vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True)
        return cand_vecs.astype(np.float32), jd_vec.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] sentence-transformers failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# TF-IDF (always built as fallback)
# ---------------------------------------------------------------------------

def build_tfidf(texts: list[str], jd_text: str, max_features: int = 50000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-\.]{1,}\b",
    )
    all_texts = [jd_text] + texts
    all_matrix = vectorizer.fit_transform(all_texts)
    jd_vec = all_matrix[0]
    cand_matrix = all_matrix[1:]
    return vectorizer, cand_matrix, jd_vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NEXUS Precomputation")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fast", action="store_true",
                        help="Skip sentence-transformers, TF-IDF only (for testing)")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("NEXUS RANKER — Precomputation")
    print("=" * 60)

    # Save JD analysis
    print("\n[1/5] Saving JD analysis...")
    with open(ARTIFACTS_DIR / "jd_analysis.json", "w") as f:
        json.dump(JD_ANALYSIS, f, indent=2)

    # Load candidates
    print(f"\n[2/5] Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # Build texts
    print("\n[3/5] Building candidate text representations...")
    t3 = time.time()
    texts, ids = [], []
    for c in candidates:
        texts.append(build_candidate_text(c))
        ids.append(c["candidate_id"])
    with open(ARTIFACTS_DIR / "candidate_ids.json", "w") as f:
        json.dump(ids, f)
    print(f"  Built {len(texts):,} texts in {time.time()-t3:.1f}s")

    # Semantic embeddings
    if not args.fast:
        print("\n[4/5] Building semantic embeddings (sentence-transformers)...")
        t4 = time.time()
        cand_vecs, jd_vec_st = encode_with_st(texts, JD_TEXT, batch_size=args.batch_size)
        if cand_vecs is not None:
            np.save(ARTIFACTS_DIR / "candidate_embeddings.npy", cand_vecs)
            np.save(ARTIFACTS_DIR / "jd_embedding.npy", jd_vec_st)
            print(f"  Saved embeddings {cand_vecs.shape} in {time.time()-t4:.1f}s")

            # Sanity check
            sims = (cand_vecs @ jd_vec_st.T).flatten()
            top_idx = np.argsort(sims)[::-1][:5]
            cmap = {c["candidate_id"]: c for c in candidates}
            print("  Top-5 by semantic similarity:")
            for i, idx in enumerate(top_idx, 1):
                cid = ids[idx]
                cp = cmap.get(cid, {}).get("profile", {})
                print(f"    #{i} {cid} sim={sims[idx]:.4f}  "
                      f"{cp.get('current_title','?')[:28]} @ {cp.get('current_company','?')[:20]}")
        else:
            print("  Falling back to TF-IDF only.")
    else:
        print("\n[4/5] Skipping embeddings (--fast mode)")

    # TF-IDF (always)
    print(f"\n[5/5] Building TF-IDF fallback (max_features={args.max_features:,})...")
    t5 = time.time()
    vectorizer, cand_matrix_tf, jd_vec_tf = build_tfidf(texts, JD_TEXT, args.max_features)
    sp.save_npz(ARTIFACTS_DIR / "tfidf_matrix.npz", cand_matrix_tf)
    sp.save_npz(ARTIFACTS_DIR / "tfidf_jd_vec.npz", jd_vec_tf)
    with open(ARTIFACTS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f, protocol=4)
    print(f"  TF-IDF matrix {cand_matrix_tf.shape}, vocab {len(vectorizer.vocabulary_):,}  "
          f"({time.time()-t5:.1f}s)")

    print(f"\n✓ Precomputation complete in {time.time()-t0:.1f}s")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print("  Run: python nexus_ranker.py --candidates ./candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
