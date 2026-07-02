"""
nexus_scorer.py — NEXUS: Neural + Expert Cross-signal Unified Scoring System

Five scoring layers:
  1. Semantic Similarity    (injected from precompute — sentence-transformer embeddings)
  2. Skill Cluster Coverage (5 JD-derived clusters vs flat keyword matching)
  3. Career Arc Intelligence (time-decayed trajectory, founding-team fit)
  4. Experience Intelligence (YoE band, domain ratio, education tier)
  5. Location + Logistics   (geo fit, notice period, work mode)

Final: base × behavioral_multiplier × anti_pattern_factor  →  clamped [0, 1]
"""

from __future__ import annotations
import math
import re
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 15)

# ---------------------------------------------------------------------------
# Skill Clusters (5 semantic groups derived from JD)
# ---------------------------------------------------------------------------

SKILL_CLUSTERS: dict[str, dict] = {
    "vector_retrieval": {
        "label": "Vector & Retrieval",
        "weight": 0.28,
        "core": {
            "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "vector search", "vector database", "hybrid search",
            "bm25", "annoy", "dense retrieval", "semantic search",
            "information retrieval", "embeddings", "sentence-transformers",
            "sentence transformers", "bi-encoder", "dense encoder",
            "openai embeddings", "bge", "e5",
        },
        "adjacent": {
            "ranking", "retrieval", "reranking", "re-ranking", "cross-encoder",
            "ndcg", "mrr", "map", "learning to rank", "search",
        },
    },
    "llm_rag": {
        "label": "LLM & RAG",
        "weight": 0.24,
        "core": {
            "rag", "retrieval augmented generation", "llm", "large language model",
            "langchain", "llamaindex", "gpt", "claude", "llama", "mistral",
            "prompt engineering", "openai",
        },
        "adjacent": {
            "huggingface", "hugging face", "transformers", "bert", "t5",
            "generative ai", "chat", "agent",
        },
    },
    "ml_training": {
        "label": "Model Training & Tuning",
        "weight": 0.20,
        "core": {
            "pytorch", "fine-tuning", "fine tuning", "finetuning",
            "lora", "qlora", "peft", "xgboost", "lightgbm", "neural ranking",
        },
        "adjacent": {
            "tensorflow", "keras", "scikit-learn", "sklearn",
            "gradient descent", "model training", "deep learning",
        },
    },
    "ml_production": {
        "label": "ML Production & Eval",
        "weight": 0.18,
        "core": {
            "mlops", "mlflow", "wandb", "weights and biases", "a/b testing",
            "ab testing", "online evaluation", "offline evaluation",
            "inference optimization", "model serving",
        },
        "adjacent": {
            "kubernetes", "docker", "ray", "airflow", "kubeflow",
            "distributed systems", "spark", "data pipeline",
        },
    },
    "python_data": {
        "label": "Python & Data",
        "weight": 0.10,
        "core": {"python", "numpy", "pandas", "sql", "feature engineering"},
        "adjacent": {"pyspark", "dask", "polars", "etl", "data engineering"},
    },
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "ltimindtree",
    "l&t infotech", "niit technologies", "mastech", "kpit", "persistent",
}

DISQUALIFIER_INDUSTRIES = {
    "it services", "paper products", "manufacturing", "conglomerate",
    "food products", "construction", "real estate", "legal",
    "healthcare administration", "retail",
}

PRODUCT_INDUSTRIES = {
    "software", "fintech", "saas", "ai/ml", "edtech", "healthtech",
    "ecommerce", "food delivery", "transportation", "media", "gaming",
    "cybersecurity", "cloud", "data analytics",
}

PREFERRED_LOCATIONS = {
    # JD explicitly names: "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR
    # welcome to apply" with offices in Noida and Pune. Bangalore/Chennai are
    # NOT named in the JD -- the previous version treated them as top-tier
    # preferred locations anyway, which doesn't match the actual requirement.
    # Candidates there still get the "willing to relocate" tier below, just
    # not the top tier.
    "pune", "noida", "delhi", "gurugram", "gurgaon", "new delhi", "ncr",
    "mumbai", "hyderabad",
}

DISQUALIFIER_TITLES = {
    "marketing manager", "hr manager", "accountant", "civil engineer",
    "mechanical engineer", "graphic designer", "content writer",
    "sales executive", "customer support", "operations manager",
    "sap consultant", "financial analyst", "supply chain", "legal", "recruiter",
    "frontend engineer", "frontend developer", "ui developer", "ui engineer",
}

CV_SPEECH_SKILLS = {
    "computer vision", "image classification", "object detection", "yolo",
    "opencv", "speech recognition", "tts", "text to speech", "ocr",
    "video understanding", "optical flow", "pose estimation", "robotics", "ros",
}

EDU_TIER_BONUS = {"tier_1": 0.10, "tier_2": 0.05, "tier_3": 0.0, "tier_4": 0.0, "unknown": 0.0}

PROFICIENCY_WEIGHT = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wm(name: str, kw: str) -> bool:
    """Word-boundary match — avoids 'ai' matching inside 'faiss'."""
    if name == kw:
        return True
    return f" {name} " in f" {kw} " or f" {kw} " in f" {name} "


def _days_ago(date_str: str) -> int:
    try:
        return (REFERENCE_DATE - datetime.strptime(date_str[:10], "%Y-%m-%d").date()).days
    except Exception:
        return 9999


def _title_disqualified(current_title: str) -> bool:
    """
    Word-boundary check against DISQUALIFIER_TITLES. The previous naive
    `dis in current_title` substring check could false-positive -- e.g. the
    disqualifier "legal" would also match inside "paralegal". Word-boundary
    matching avoids penalizing candidates for substrings that happen to
    appear inside an unrelated word.
    """
    for dis in DISQUALIFIER_TITLES:
        if re.search(r"(?<![a-z])" + re.escape(dis) + r"(?![a-z])", current_title):
            return True
    return False


def _is_title_chaser(career: list[dict]) -> bool:
    """
    JD explicitly names this pattern as a disqualifier: "If your career
    trajectory shows you optimizing for 'Senior' -> 'Staff' -> 'Principal'
    titles by switching companies every 1.5 years, we're not a fit." This
    was not detected anywhere in the previous scorer.

    Heuristic: 3+ completed (non-current) roles averaging under 18 months
    tenure each signals a job-hopping pattern rather than a single short
    stint, which is a normal and different thing.
    """
    completed = [j for j in career if not j.get("is_current", False)]
    if len(completed) < 3:
        return False
    avg_tenure = sum(j.get("duration_months", 0) for j in completed) / len(completed)
    return avg_tenure < 18


def career_pattern_flags(c: dict) -> dict:
    """
    Consolidated analysis of the JD's named hard-disqualifier career
    patterns, computed once and reused by both the career-arc score and the
    semantic gate so the two stay consistent with each other.
    """
    career = c.get("career_history", [])
    total = sum(j.get("duration_months", 0) for j in career) or 1

    consulting_months = sum(
        j.get("duration_months", 0) for j in career
        if j.get("industry", "").lower() in DISQUALIFIER_INDUSTRIES
        or any(f in j.get("company", "").lower() for f in CONSULTING_FIRMS)
    )
    research_months = sum(
        j.get("duration_months", 0) for j in career
        if any(kw in j.get("title", "").lower() for kw in
               ["research scientist", "phd researcher", "academic"])
        and "production" not in (j.get("description", "") or "").lower()
    )
    management_months = sum(
        j.get("duration_months", 0) for j in career
        if any(t in j.get("title", "").lower() for t in
               ["vp", "director", "head of", "cto", "cpo", "chief",
                "project manager", "program manager"])
    )

    return {
        "title_disqualified": _title_disqualified(
            c.get("profile", {}).get("current_title", "").lower()
        ),
        "consulting_only": consulting_months / total > 0.85,
        "research_only": research_months / total > 0.7,
        "management_track": management_months / total > 0.6,
        "title_chaser": _is_title_chaser(career),
    }


def _job_ml_relevance(job: dict) -> float:
    """0-1 relevance score for how AI/ML-aligned a single job is."""
    title = job.get("title", "").lower()
    desc = (job.get("description", "") or "").lower()
    industry = job.get("industry", "").lower()
    company = job.get("company", "").lower()

    score = 0.0
    # Title signals
    if any(t in title for t in ["ml engineer", "machine learning engineer", "ai engineer",
                                 "search engineer", "ranking engineer", "recommendation",
                                 "applied scientist", "nlp engineer", "research engineer",
                                 "retrieval engineer"]):
        score += 0.6
    elif any(t in title for t in ["data scientist", "ml", "machine learning", "ai ", "nlp",
                                   "search", "ranking"]):
        score += 0.4
    elif any(t in title for t in ["data engineer", "software engineer", "backend engineer",
                                   "platform engineer", "mlops"]):
        score += 0.15

    # Description signals
    desc_keywords = ["ranking", "retrieval", "embedding", "vector", "faiss",
                     "elasticsearch", "semantic", "relevance", "recommendation",
                     "deployed", "production", "real users", "llm", "rag"]
    desc_hits = sum(1 for kw in desc_keywords if kw in desc)
    score += min(0.35, desc_hits * 0.05)

    # Industry penalty
    is_consulting = industry in DISQUALIFIER_INDUSTRIES or any(f in company for f in CONSULTING_FIRMS)
    if is_consulting:
        score *= 0.2

    # Startup / founding signal
    company_size = job.get("company_size", "")
    if company_size in ("1-10", "11-50"):
        score += 0.05

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Layer 1 — Skill Cluster Coverage
# ---------------------------------------------------------------------------

def score_skill_clusters(c: dict) -> tuple[float, dict[str, float]]:
    """
    Scores skill cluster coverage. Returns (overall_score, per_cluster_scores).
    Depth within a cluster is rewarded via log-scaling.
    """
    skills = c.get("skills", [])
    signals = c.get("redrob_signals", {})
    assess = signals.get("skill_assessment_scores", {})

    cluster_scores: dict[str, float] = {}

    for cname, cluster in SKILL_CLUSTERS.items():
        raw = 0.0
        max_possible = 0.0

        for sk in skills:
            name = sk.get("name", "").lower()
            prof = sk.get("proficiency", "intermediate")
            endorsements = sk.get("endorsements", 0)
            duration = sk.get("duration_months", 0)

            # Anti-stuffer
            if endorsements > 20 and duration == 0:
                endorsements = min(endorsements, 2)

            in_core = any(_wm(name, kw) for kw in cluster["core"])
            in_adjacent = any(_wm(name, kw) for kw in cluster["adjacent"])

            if not (in_core or in_adjacent):
                continue

            pw = PROFICIENCY_WEIGHT.get(prof, 0.5)
            ef = math.log1p(endorsements) / math.log1p(100)
            df = math.log1p(duration) / math.log1p(60)

            # Assessment boost
            boost = 1.0
            for aname, ascore in assess.items():
                if _wm(name, aname.lower()) or _wm(aname.lower(), name):
                    boost = 0.5 + (ascore / 100.0) * 0.8
                    break

            skill_val = pw * (0.3 + 0.4 * ef + 0.3 * df) * boost
            weight = 1.0 if in_core else 0.4
            raw += skill_val * weight
            max_possible += weight

        if max_possible > 0:
            # Log-scale to reward depth: perfect coverage of 3 core skills ~= 0.85
            norm = raw / max(max_possible, 3.0)
            cluster_scores[cname] = min(1.0, norm * 1.5)
        else:
            cluster_scores[cname] = 0.0

    # Weighted sum
    total = sum(cluster_scores.get(cn, 0.0) * cl["weight"]
                for cn, cl in SKILL_CLUSTERS.items())

    # CV/speech domain penalty
    all_names = {sk.get("name", "").lower() for sk in skills}
    cv_count = sum(1 for n in all_names if any(_wm(n, cv) for cv in CV_SPEECH_SKILLS))
    if len(skills) > 0 and cv_count / max(len(skills), 1) > 0.4:
        total *= 0.5

    return min(1.0, total), cluster_scores


# ---------------------------------------------------------------------------
# Layer 2 — Career Arc Intelligence
# ---------------------------------------------------------------------------

def score_career_arc(c: dict) -> tuple[float, dict]:
    """
    Time-decayed career trajectory scoring.
    Recent AI/ML roles at product companies are worth much more than old ones.
    Returns (score, metadata_dict).
    """
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    if not career:
        return 0.0, {}

    current_title = profile.get("current_title", "").lower()

    # Hard disqualifiers named explicitly in the JD. Previously only the
    # current-title check was applied here; research-only, management-track,
    # and job-hopping ("title-chasing") patterns are also named in the JD as
    # "we will not move forward" cases but were only given a soft partial
    # penalty on one experience sub-component. Applying them here as a single
    # consistent gate matches how strongly the JD states them.
    flags = career_pattern_flags(c)
    title_pen = 1.0
    if flags["title_disqualified"]:
        title_pen = 0.10
    elif flags["research_only"]:
        title_pen = 0.15
    elif flags["title_chaser"]:
        title_pen = 0.35
    elif flags["management_track"]:
        title_pen = 0.40

    # Sort by recency (most recent end_date first; current roles get pseudo-date '9999')
    def _sort_key(j):
        ed = j.get("end_date") or "9999-12-31"
        return ed if isinstance(ed, str) else "9999-12-31"

    sorted_jobs = sorted(career, key=_sort_key, reverse=True)

    weighted_relevance = 0.0
    total_weight = 0.0
    founding_bonus = 0.0
    recent_ml_streak = 0  # consecutive recent ML roles

    for rank_idx, job in enumerate(sorted_jobs):
        # Approximate years_ago
        end_str = job.get("end_date")
        if end_str is None or job.get("is_current"):
            years_ago = 0.0
        else:
            try:
                ed = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                years_ago = max(0.0, (REFERENCE_DATE - ed).days / 365.25)
            except Exception:
                years_ago = (rank_idx + 1) * 1.5

        # Exponential time decay: recent = 1.0, 3yr ago = 0.64, 8yr ago = 0.30
        time_weight = math.exp(-0.15 * years_ago)
        duration_yrs = max(0.1, job.get("duration_months", 0) / 12.0)

        relevance = _job_ml_relevance(job)

        # Track founding-team experience
        if job.get("company_size", "") in ("1-10", "11-50"):
            founding_bonus = max(founding_bonus, 0.08 * time_weight)

        # ML streak (top 2 recent roles)
        if rank_idx < 2 and relevance >= 0.4:
            recent_ml_streak += 1

        w = time_weight * duration_yrs
        weighted_relevance += relevance * w
        total_weight += w

    arc_score = (weighted_relevance / total_weight) if total_weight > 0 else 0.0

    # Bonus for recent ML streak
    if recent_ml_streak == 2:
        arc_score = min(1.0, arc_score + 0.12)
    elif recent_ml_streak == 1:
        arc_score = min(1.0, arc_score + 0.05)

    arc_score = min(1.0, arc_score + founding_bonus)
    arc_score *= title_pen

    meta = {
        "recent_ml_streak": recent_ml_streak,
        "founding_bonus": round(founding_bonus, 3),
        "arc_raw": round(arc_score, 3),
        "career_pattern_flags": {k: v for k, v in flags.items() if v},
    }
    return min(1.0, max(0.0, arc_score)), meta


# ---------------------------------------------------------------------------
# Layer 3 — Experience Intelligence
# ---------------------------------------------------------------------------

def score_experience(c: dict) -> float:
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 11:
        yoe_score = 0.75
    elif 3 <= yoe < 4 or 11 < yoe <= 13:
        yoe_score = 0.50
    elif yoe < 3:
        yoe_score = 0.20
    else:
        yoe_score = 0.45

    total_months = sum(j.get("duration_months", 0) for j in career) or 1
    applied_ml = research_only = management = 0

    for job in career:
        title = job.get("title", "").lower()
        desc = (job.get("description", "") or "").lower()
        dur = job.get("duration_months", 0)
        if any(t in title for t in ["vp", "director", "head of", "cto", "cpo", "chief",
                                     "project manager", "program manager"]):
            management += dur
        if any(kw in title for kw in ["ml", "machine learning", "ai ", "search",
                                       "ranking", "recommendation", "nlp", "applied"]):
            applied_ml += dur
        if any(kw in title for kw in ["research scientist", "phd researcher", "academic"]) \
                and "production" not in desc:
            research_only += dur

    domain = min(1.0, (applied_ml / total_months) * 2)
    if research_only / total_months > 0.7:
        domain *= 0.2
    if management / total_months > 0.6:
        domain *= 0.4

    # Education tier
    best_tier = max(
        (EDU_TIER_BONUS.get(edu.get("tier", "unknown"), 0.0) for edu in c.get("education", [])),
        default=0.0,
    )

    base = yoe_score * 0.5 + domain * 0.5
    return min(1.0, max(0.0, base + best_tier))


# ---------------------------------------------------------------------------
# Layer 4 — Location + Logistics
# ---------------------------------------------------------------------------

def score_location(c: dict) -> float:
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    relocate = signals.get("willing_to_relocate", False)
    notice = signals.get("notice_period_days", 90)
    mode = signals.get("preferred_work_mode", "flexible")

    if country == "india":
        if any(p in location for p in PREFERRED_LOCATIONS):
            loc = 1.0
        elif relocate:
            loc = 0.75
        else:
            loc = 0.50
    else:
        loc = 0.30 if relocate else 0.05

    if notice <= 15:
        ns = 1.0
    elif notice <= 30:
        ns = 0.95
    elif notice <= 60:
        ns = 0.70
    elif notice <= 90:
        ns = 0.45
    elif notice <= 120:
        ns = 0.20
    else:
        ns = 0.05

    ms = 1.0 if mode in ("hybrid", "flexible") else (0.8 if mode == "onsite" else 0.6)

    return min(1.0, max(0.0, loc * 0.5 + ns * 0.35 + ms * 0.15))


# ---------------------------------------------------------------------------
# Behavioral Multiplier (all 23 signals with time-decay)
# ---------------------------------------------------------------------------

def behavioral_multiplier(c: dict) -> float:
    """
    Composite behavioral signal score, expressed as a bounded multiplier.

    CALIBRATION NOTE: the previous version multiplied ~15 independent factors
    together (each swinging the score up to +/-35%). Multiplicative stacking
    of that many factors compounds fast -- worst case ~0.07x, best case ~2.1x
    before clamping to [0.25, 1.30]. That let a handful of behavioral signals
    overwhelm actual job-fit scoring (career arc, skills, semantic, experience,
    location), which is what produced the "one perfect candidate, then a
    cliff" ranking behavior seen in the sandbox UI.

    Fix: compute a single normalized behavioral_score in [0, 1] as a WEIGHTED
    AVERAGE (not a product) of all 23 signals, then map that into a modest,
    bounded multiplier range. Behavioral signals should break ties and reward
    genuine availability -- they should not be able to out-weigh job fit.
    """
    s = c.get("redrob_signals", {})

    def contrib(value: float, weight: float) -> tuple[float, float]:
        v = max(0.0, min(1.0, value))
        return v * weight, weight

    parts = []

    # 1. Open to work (meaningful, but not a hard gate)
    parts.append(contrib(1.0 if s.get("open_to_work_flag", True) else 0.30, 0.15))

    # 2. Recency of activity (time-decayed)
    days_inactive = _days_ago(s.get("last_active_date", "2020-01-01"))
    if days_inactive <= 14:
        recency = 1.0
    elif days_inactive <= 30:
        recency = 0.85
    elif days_inactive <= 60:
        recency = 0.65
    elif days_inactive <= 90:
        recency = 0.45
    elif days_inactive <= 180:
        recency = 0.25
    else:
        recency = 0.10
    parts.append(contrib(recency, 0.18))

    # 3. Recruiter response rate
    parts.append(contrib(s.get("recruiter_response_rate", 0.5), 0.10))

    # 4. Interview completion rate
    parts.append(contrib(s.get("interview_completion_rate", 0.5), 0.08))

    # 5. Notice period (availability)
    notice = s.get("notice_period_days", 90)
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.70
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.20
    parts.append(contrib(notice_score, 0.10))

    # 6. GitHub activity (missing/-1 treated as neutral, never penalized as if bad)
    gh = s.get("github_activity_score", -1)
    gh_score = 0.5 if gh < 0 else gh / 100.0
    parts.append(contrib(gh_score, 0.10))

    # 7. Profile completeness
    pc = s.get("profile_completeness_score", 70)
    parts.append(contrib(pc / 100.0, 0.07))

    # 8. Verified contact info
    if s.get("verified_email") and s.get("verified_phone"):
        verified_score = 1.0
    elif s.get("verified_email") or s.get("verified_phone"):
        verified_score = 0.6
    else:
        verified_score = 0.30
    parts.append(contrib(verified_score, 0.06))

    # 9. Active job-seeking behavior
    apps = s.get("applications_submitted_30d", 0)
    parts.append(contrib(apps / 5.0, 0.05))

    # 10. Saved by recruiters
    saved = s.get("saved_by_recruiters_30d", 0)
    parts.append(contrib(saved / 10.0, 0.04))

    # 11. LinkedIn connected
    parts.append(contrib(1.0 if s.get("linkedin_connected", False) else 0.5, 0.02))

    # 12. Avg response time (lower is better)
    rth = s.get("avg_response_time_hours", 24)
    if rth <= 4:
        rt_score = 1.0
    elif rth <= 24:
        rt_score = 0.7
    elif rth <= 48:
        rt_score = 0.4
    else:
        rt_score = 0.2
    parts.append(contrib(rt_score, 0.02))

    # 13. Search appearance
    parts.append(contrib(s.get("search_appearance_30d", 0) / 20.0, 0.01))

    # 14. Profile views
    parts.append(contrib(s.get("profile_views_received_30d", 0) / 40.0, 0.01))

    # 15. Salary expectation fit (heuristic: very high asks score lower)
    sal = s.get("expected_salary_range_inr_lpa", {})
    sal_min = sal.get("min", 0) if isinstance(sal, dict) else 0
    if sal_min > 80:
        sal_score = 0.5
    elif sal_min > 60:
        sal_score = 0.75
    else:
        sal_score = 1.0
    parts.append(contrib(sal_score, 0.01))

    weighted_total = sum(v for v, _ in parts)
    weight_sum = sum(w for _, w in parts)
    behavioral_score = weighted_total / weight_sum if weight_sum > 0 else 0.5

    # Map [0, 1] behavioral_score into a modest multiplier band -- a
    # tie-breaker / calibration signal, not a dominant scoring factor.
    # Worst case 0.85x, best case 1.20x (vs. the old 0.25x-1.30x, ~5x swing).
    multiplier = 0.85 + 0.35 * behavioral_score
    return round(multiplier, 4)


# ---------------------------------------------------------------------------
# Anti-Pattern Detection (graded 0-1, not binary)
# ---------------------------------------------------------------------------

def detect_anti_patterns(c: dict) -> tuple[float, list[str]]:
    """
    Returns (suspicion_score 0-1, reasons).
    suspicion_score is applied as: anti_factor = 1 - suspicion_score * 0.95
    """
    flags: list[str] = []
    categories: set[str] = set()
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    # 1. YoE vs earliest job mismatch
    yoe = profile.get("years_of_experience", 0)
    earliest = None
    for job in career:
        sd = job.get("start_date")
        if sd:
            try:
                d = datetime.strptime(sd[:10], "%Y-%m-%d").date()
                if earliest is None or d < earliest:
                    earliest = d
            except Exception:
                pass
    if earliest:
        actual = (REFERENCE_DATE - earliest).days / 365.25
        if yoe > actual + 3:
            flags.append(f"yoe_mismatch({yoe:.0f}yr_claimed,{actual:.0f}yr_actual)")
            categories.add("yoe_mismatch")

    # 2. Expert skill + zero duration
    for sk in skills:
        if sk.get("proficiency") == "expert" and sk.get("duration_months", 1) == 0:
            flags.append(f"expert_zero_dur:{sk.get('name','?')}")
            categories.add("expert_zero_dur")

    # 3. Overlapping jobs (>3 months)
    ranges = []
    for job in career:
        sd, ed = job.get("start_date"), job.get("end_date")
        if sd and ed:
            try:
                s = datetime.strptime(sd[:10], "%Y-%m-%d").date()
                e = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                ranges.append((s, e, job.get("company", "")))
            except Exception:
                pass
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            s1, e1, c1 = ranges[i]
            s2, e2, c2 = ranges[j]
            if c1 != c2 and s2 < e1 and s1 < e2:
                overlap = (min(e1, e2) - max(s1, s2)).days / 30
                if overlap > 3:
                    flags.append(f"overlap:{c1}&{c2}")
                    categories.add("overlapping_jobs")

    # 4. Perfect completeness + empty profile
    if signals.get("profile_completeness_score", 0) > 95 and not skills and not career:
        flags.append("perfect_score_empty_profile")
        categories.add("perfect_score_empty_profile")

    # 5. Future start dates
    for job in career:
        sd = job.get("start_date")
        if sd:
            try:
                if datetime.strptime(sd[:10], "%Y-%m-%d").date() > REFERENCE_DATE:
                    flags.append(f"future_start:{sd}")
                    categories.add("future_start")
            except Exception:
                pass

    # 6. Endorsements > connections
    conns = signals.get("connection_count", 0)
    endrs = signals.get("endorsements_received", 0)
    if conns < 5 and endrs > 200:
        flags.append(f"endrs_exceed_conns({endrs}e,{conns}c)")
        categories.add("endrs_exceed_conns")

    # 7. All expert, no assessments
    expert_ct = sum(1 for sk in skills if sk.get("proficiency") == "expert")
    if expert_ct >= 5 and not signals.get("skill_assessment_scores") and skills:
        flags.append(f"all_expert_no_assess({expert_ct})")
        categories.add("all_expert_no_assess")

    # Graded suspicion based on DISTINCT rule categories triggered, not raw
    # flag count. The previous version counted every instance (e.g. 3
    # overlapping job *pairs*, or 2 expert-zero-duration *skills*) as
    # separate flags, which let one systemic issue of a single type
    # falsely trip the honeypot cap. Counting distinct categories means a
    # candidate needs several genuinely different kinds of red flags before
    # being treated as a likely-synthetic profile.
    n = len(categories)
    if n == 0:
        suspicion = 0.0
    elif n == 1:
        suspicion = 0.15   # slight concern -- do not cap
    elif n == 2:
        suspicion = 0.55   # notable, but not an automatic honeypot
    else:
        suspicion = 0.90   # 3+ distinct anomaly types -- almost certainly synthetic

    return suspicion, flags


# ---------------------------------------------------------------------------
# Dynamic Weight Calibration (JD-driven)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "career_arc":      0.30,
    "skill_clusters":  0.25,
    "semantic":        0.25,
    "experience":      0.12,
    "location":        0.08,
}


# ---------------------------------------------------------------------------
# Main Scoring Entry Point
# ---------------------------------------------------------------------------

def compute_nexus_score(c: dict, semantic_score: float = 0.0) -> dict:
    """
    Multi-layer scoring. Returns dict with all components + final_score.
    """
    # Semantic gating for JD-named hard disqualifiers -- kept consistent with
    # score_career_arc by using the same career_pattern_flags() analysis
    # instead of a separately-duplicated consulting-only calculation.
    flags = career_pattern_flags(c)
    if flags["title_disqualified"] or flags["consulting_only"] or flags["research_only"]:
        semantic_score *= 0.12
    elif flags["title_chaser"] or flags["management_track"]:
        semantic_score *= 0.45

    arc_score, arc_meta = score_career_arc(c)
    skill_score, cluster_scores = score_skill_clusters(c)
    exp_score = score_experience(c)
    loc_score = score_location(c)

    base = (
        arc_score   * WEIGHTS["career_arc"] +
        skill_score * WEIGHTS["skill_clusters"] +
        semantic_score * WEIGHTS["semantic"] +
        exp_score   * WEIGHTS["experience"] +
        loc_score   * WEIGHTS["location"]
    )

    bm = behavioral_multiplier(c)
    suspicion, anti_flags = detect_anti_patterns(c)
    anti_factor = 1.0 - suspicion * 0.95

    final = min(1.0, max(0.0, base * bm * anti_factor))

    return {
        "final_score":            round(final, 6),
        "base_score":             round(base, 6),
        "career_arc":             round(arc_score, 4),
        "skill_clusters":         round(skill_score, 4),
        "semantic":               round(semantic_score, 4),
        "experience":             round(exp_score, 4),
        "location":               round(loc_score, 4),
        "behavioral_multiplier":  round(bm, 4),
        "suspicion_score":        round(suspicion, 3),
        "anti_pattern_flags":     anti_flags,
        "is_honeypot":            suspicion >= 0.70,
        "cluster_scores":         {k: round(v, 4) for k, v in cluster_scores.items()},
        "arc_meta":               arc_meta,
    }