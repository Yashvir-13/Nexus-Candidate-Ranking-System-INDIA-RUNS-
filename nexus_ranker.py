#!/usr/bin/env python3
"""
nexus_ranker.py — NEXUS Main Ranking Pipeline

Usage:
  python nexus_ranker.py --candidates ./candidates.jsonl --out ./submission.csv
  python nexus_ranker.py --candidates ./sample_candidates.json --out ./submission.csv --diag
"""

import argparse, csv, gzip, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from nexus_scorer import compute_nexus_score
from nexus_reasoning import generate_nexus_reasoning
from nexus_precompute import build_candidate_text

ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "nexus"

# Fixed reference ceilings for similarity normalization (NOT computed from
# the current batch). Using batch min/max meant the same candidate's semantic
# score changed depending on who else happened to be scored alongside them --
# unstable and not reproducible if you ever score a subset. A fixed ceiling
# based on typical "strong match" magnitude for each method keeps scores
# stable and comparable run-to-run.
TFIDF_SIM_CEILING = 0.15    # observed strong-match raw cosine ~0.10-0.12
ST_EMBEDDING_SIM_CEILING = 0.55  # typical strong-match cosine for MiniLM-class models


def normalize_similarity(sims: np.ndarray, ceiling: float) -> np.ndarray:
    """
    Fixed-scale normalization: clip to [0, ceiling], scale to [0, 1], then
    apply a mild sqrt to spread out the low/mid range (raw cosine similarities
    for this domain cluster tightly near zero, so a linear scale would flatten
    almost every candidate into an indistinguishable low band).
    """
    scaled = np.clip(sims, 0.0, None) / ceiling
    return np.clip(np.sqrt(np.clip(scaled, 0.0, 1.0)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
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
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    else:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    return candidates


# ---------------------------------------------------------------------------
# Semantic score computation
# ---------------------------------------------------------------------------

def compute_semantic_scores(candidates: list[dict]) -> dict[str, float]:
    """
    Try sentence-transformer embeddings first, fall back to TF-IDF.
    Returns {candidate_id: normalized_score}.
    """
    ids = [c["candidate_id"] for c in candidates]
    id_set = set(ids)

    # --- Try sentence-transformer embeddings ---
    emb_path = ARTIFACTS_DIR / "candidate_embeddings.npy"
    jd_emb_path = ARTIFACTS_DIR / "jd_embedding.npy"
    id_path = ARTIFACTS_DIR / "candidate_ids.json"

    if emb_path.exists() and jd_emb_path.exists() and id_path.exists():
        try:
            print("  Loading sentence-transformer embeddings...")
            cand_vecs = np.load(str(emb_path))           # (N, dim)
            jd_vec = np.load(str(jd_emb_path))           # (1, dim)
            with open(id_path) as f:
                all_ids = json.load(f)
            id_to_row = {cid: i for i, cid in enumerate(all_ids)}

            # Cosine similarity (embeddings are L2-normalized → dot product = cosine)
            sims = (cand_vecs @ jd_vec.T).flatten()

            # Normalize using a fixed reference scale (not batch-dependent)
            sims_norm = normalize_similarity(sims, ST_EMBEDDING_SIM_CEILING)

            result = {}
            missing = []
            for c in candidates:
                cid = c["candidate_id"]
                row = id_to_row.get(cid)
                if row is not None:
                    result[cid] = float(np.clip(sims_norm[row], 0.0, 1.0))
                else:
                    missing.append(c)

            # On-the-fly for missing candidates
            if missing:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    texts = [build_candidate_text(c) for c in missing]
                    fly_vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                                            show_progress_bar=False)
                    fly_sims = (fly_vecs @ jd_vec.T).flatten()
                    fly_norm = normalize_similarity(fly_sims, ST_EMBEDDING_SIM_CEILING)
                    for c, score in zip(missing, fly_norm):
                        result[c["candidate_id"]] = float(score)
                except Exception:
                    for c in missing:
                        result[c["candidate_id"]] = 0.0

            print(f"  Semantic (ST embeddings): {len(result)} candidates scored")
            return result
        except Exception as e:
            print(f"  [WARN] Embedding load failed: {e} — trying TF-IDF")

    # --- Fall back to TF-IDF ---
    vect_path = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
    mat_path  = ARTIFACTS_DIR / "tfidf_matrix.npz"
    jd_path   = ARTIFACTS_DIR / "tfidf_jd_vec.npz"
    id_path   = ARTIFACTS_DIR / "candidate_ids.json"

    if vect_path.exists() and mat_path.exists() and jd_path.exists() and id_path.exists():
        try:
            with open(vect_path, "rb") as f:
                vectorizer = pickle.load(f)
            cand_matrix = sp.load_npz(str(mat_path))
            jd_vec_tf   = sp.load_npz(str(jd_path))
            with open(id_path) as f:
                all_ids = json.load(f)
            id_to_row = {cid: i for i, cid in enumerate(all_ids)}

            sims = cosine_similarity(jd_vec_tf, cand_matrix).flatten()
            sims_norm = normalize_similarity(sims, TFIDF_SIM_CEILING)

            result = {}
            missing = []
            for c in candidates:
                cid = c["candidate_id"]
                row = id_to_row.get(cid)
                if row is not None:
                    result[cid] = float(np.clip(sims_norm[row], 0.0, 1.0))
                else:
                    missing.append(c)

            if missing:
                texts = [build_candidate_text(c) for c in missing]
                fly_m = vectorizer.transform(texts)
                fly_s = cosine_similarity(jd_vec_tf, fly_m).flatten()
                fly_n = normalize_similarity(fly_s, TFIDF_SIM_CEILING)
                for c, score in zip(missing, fly_n):
                    result[c["candidate_id"]] = float(score)

            print(f"  Semantic (TF-IDF): {len(result)} candidates scored")
            return result
        except Exception as e:
            print(f"  [WARN] TF-IDF fallback failed: {e}")

    print("  [WARN] No precomputed artifacts found — semantic=0 for all candidates")
    print("         Run: python nexus_precompute.py --candidates <candidates_file>")
    return {c["candidate_id"]: 0.0 for c in candidates}


# ---------------------------------------------------------------------------
# Ranking pipeline
# ---------------------------------------------------------------------------

def rank_candidates(candidates: list[dict], semantic_scores: dict[str, float]) -> list[dict]:
    results = []
    hp_count = 0
    for c in candidates:
        cid = c["candidate_id"]
        sem = semantic_scores.get(cid, 0.0)
        scores = compute_nexus_score(c, semantic_score=sem)
        if scores["is_honeypot"]:
            hp_count += 1
        results.append({
            "candidate_id": cid,
            "candidate":    c,
            "scores":       scores,
            "final_score":  scores["final_score"],
        })

    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    print(f"  Anti-patterns detected: {hp_count}")
    if results:
        print(f"  Score range: {results[0]['final_score']:.4f} → {results[-1]['final_score']:.6f}")
    return results


def ensure_monotone(rows: list[dict]) -> list[dict]:
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i - 1]["score"]:
            rows[i]["score"] = round(rows[i - 1]["score"] - 1e-7, 7)
    return rows


def build_submission(ranked: list[dict], top_n: int = 100) -> list[dict]:
    top_n = min(top_n, len(ranked))
    rows = []
    for rank, item in enumerate(ranked[:top_n], 1):
        c      = item["candidate"]
        scores = item["scores"]
        reason = generate_nexus_reasoning(c, scores, rank)
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank":         rank,
            "score":        item["final_score"],
            "reasoning":    reason,
        })
    return ensure_monotone(rows)


def write_csv(rows: list[dict], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "rank":         row["rank"],
                "score":        f"{row['score']:.6f}",
                "reasoning":    row["reasoning"],
            })
    print(f"  Wrote {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(ranked: list[dict], top_n: int = 20) -> None:
    hdr = (f"{'Rk':>3} {'CandidateID':<14} {'Score':>7} "
           f"{'Arc':>5} {'Skl':>5} {'Sem':>5} {'Exp':>5} {'Loc':>5} "
           f"{'BM':>5} {'Sus':>4}  Title @ Company")
    print("\n" + hdr)
    print("-" * 120)
    for i, item in enumerate(ranked[:top_n], 1):
        c = item["candidate"]
        s = item["scores"]
        p = c.get("profile", {})
        sus = f"{s['suspicion_score']:.2f}"
        title   = p.get("current_title", "?")[:26]
        company = p.get("current_company", "?")[:20]
        print(
            f"{i:>3} {c['candidate_id']:<14} {s['final_score']:>7.4f}"
            f"  {s['career_arc']:>5.3f} {s['skill_clusters']:>5.3f}"
            f"  {s['semantic']:>5.3f} {s['experience']:>5.3f}"
            f"  {s['location']:>5.3f} {s['behavioral_multiplier']:>5.3f}"
            f"  {sus:>4}  {title} @ {company}"
        )

    scores_top = [x["final_score"] for x in ranked[:min(100, len(ranked))]]
    if len(scores_top) >= 10:
        ss = sorted(scores_top, reverse=True)
        print(f"\n  Score percentiles — "
              f"P10={ss[min(9,len(ss)-1)]:.4f}  "
              f"P25={ss[min(24,len(ss)-1)]:.4f}  "
              f"P50={ss[min(49,len(ss)-1)]:.4f}  "
              f"P100={ss[-1]:.4f}")

    hp_in_top = [x for x in ranked[:100] if x["scores"]["is_honeypot"]]
    print(f"\n  Anti-pattern candidates in top-100: {len(hp_in_top)} "
          f"({'⚠ RISK' if len(hp_in_top) > 10 else '✓ OK'})")

    # Cluster coverage of top-10
    print("\n  Skill cluster coverage — Top 10 avg:")
    from nexus_scorer import SKILL_CLUSTERS
    for cname, cl in SKILL_CLUSTERS.items():
        avg = np.mean([x["scores"]["cluster_scores"].get(cname, 0) for x in ranked[:10]])
        print(f"    {cl['label']:28s}: {avg:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NEXUS Intelligent Candidate Ranker")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("NEXUS RANKER — Neural + Expert Cross-signal Unified System")
    print("=" * 60)

    print(f"\n[1/4] Loading candidates...")
    candidates = load_candidates(args.candidates)
    print(f"  {len(candidates):,} candidates loaded  ({time.time()-t_start:.1f}s)")

    print("\n[2/4] Computing semantic scores...")
    t2 = time.time()
    sem_scores = compute_semantic_scores(candidates)
    print(f"  Semantic scores ready  ({time.time()-t2:.1f}s)")

    print(f"\n[3/4] Scoring {len(candidates):,} candidates (NEXUS multi-layer)...")
    t3 = time.time()
    ranked = rank_candidates(candidates, sem_scores)
    print(f"  Scoring complete  ({time.time()-t3:.1f}s)")

    print(f"\n[4/4] Building top-{args.top} submission...")
    rows = build_submission(ranked, top_n=args.top)
    write_csv(rows, args.out)

    diag_n = 100 if args.diag else 25
    print_diagnostics(ranked, top_n=diag_n)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"✓  Finished in {elapsed:.1f}s  |  Output → {args.out}")
    if elapsed > 300:
        print(f"⚠  Exceeded 5-minute budget ({elapsed:.0f}s)!")
    print(f"{'='*60}")
    print(f"\n  Validate: python India_runs_data_and_ai_challenge/validate_submission.py {args.out}")


if __name__ == "__main__":
    main()
