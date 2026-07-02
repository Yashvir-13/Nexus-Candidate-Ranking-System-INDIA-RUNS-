"""
nexus_app.py — NEXUS Candidate Intelligence Sandbox UI

A Streamlit sandbox for the NEXUS candidate ranking system. Shows the ranked
shortlist for the Senior AI Engineer JD, with score breakdowns, behavioral
signals, and lightweight analytics.

Run:
  streamlit run nexus_app.py
"""

import gzip
import json
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from nexus_scorer import compute_nexus_score, SKILL_CLUSTERS, WEIGHTS
from nexus_reasoning import generate_nexus_reasoning
from nexus_ranker import normalize_similarity, TFIDF_SIM_CEILING, ST_EMBEDDING_SIM_CEILING

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NEXUS — Candidate Intelligence",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts" / "nexus"
REFERENCE_DATE = date(2026, 6, 15)

# ---------------------------------------------------------------------------
# Palette — deliberately restrained: one dominant neutral, one accent,
# three status colors used consistently and sparingly.
# ---------------------------------------------------------------------------
BG = "#12141a"
PANEL = "#1a1d26"
PANEL_ALT = "#20232e"
BORDER = "#2c303c"
TEXT = "#e6e8ec"
TEXT_MUTED = "#8b90a0"
ACCENT = "#4f8cff"        # single accent — used for the primary brand mark and links only
GOOD = "#3ecf8e"
WARN = "#e0a840"
BAD = "#e0596b"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
.mono {{ font-family: 'IBM Plex Mono', monospace; }}

.stApp {{ background: {BG}; }}

/* Header */
.app-header {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 24px 28px;
    margin-bottom: 20px;
    background: {PANEL};
}}
.app-title {{ color: {TEXT}; font-size: 22px; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
.app-subtitle {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 4px; }}
.app-tag {{
    display: inline-block;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 8px 6px 0 0;
    font-family: 'IBM Plex Mono', monospace;
}}

/* Metric tiles */
.metric-box {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
}}
.metric-num {{ font-size: 22px; font-weight: 700; color: {TEXT}; font-family: 'IBM Plex Mono', monospace; }}
.metric-lbl {{ font-size: 11px; color: {TEXT_MUTED}; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }}

/* Candidate card -- these target Streamlit's own container wrapper via the
   `key=` parameter (st.container(key=...)), NOT a manually-opened <div>.
   A <div> opened in one st.markdown() call and closed in a later one does
   NOT visually wrap the elements rendered in between -- each st.markdown()
   call is an independent HTML fragment in Streamlit, so that approach only
   produces a stray empty bordered box. st.container(key=...) is the correct
   way to group a variable number of Streamlit elements into one real,
   styleable wrapper. */
div[class*="st-key-nxcard"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-left: 3px solid {BORDER};
    border-radius: 6px;
    padding: 16px 20px 4px 20px;
    margin-bottom: 10px;
}}
div[class*="st-key-nxcardtop"] {{ border-left-color: {ACCENT}; }}
div[class*="st-key-nxcardflag"] {{ border-left-color: {BAD}; background: #1c1518; }}

.rank-label {{
    font-family: 'IBM Plex Mono', monospace;
    color: {TEXT_MUTED};
    font-size: 13px;
    font-weight: 600;
}}
.candidate-name {{ color: {TEXT}; font-size: 15px; font-weight: 600; }}
.candidate-meta {{ color: {TEXT_MUTED}; font-size: 12px; margin-top: 2px; }}

.score-pill {{
    text-align: center;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'IBM Plex Mono', monospace;
}}
.score-pill-num {{ font-size: 18px; font-weight: 700; }}
.score-pill-lbl {{ font-size: 9px; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.05em; }}

.reasoning-text {{ color: {TEXT_MUTED}; font-size: 13px; font-style: italic; margin: 10px 0; line-height: 1.5; }}

.badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    margin: 2px 4px 2px 0;
    border: 1px solid;
    font-family: 'IBM Plex Mono', monospace;
}}
.badge-good {{ color: {GOOD}; border-color: rgba(62,207,142,0.35); background: rgba(62,207,142,0.08); }}
.badge-warn {{ color: {WARN}; border-color: rgba(224,168,64,0.35); background: rgba(224,168,64,0.08); }}
.badge-bad {{ color: {BAD}; border-color: rgba(224,89,107,0.35); background: rgba(224,89,107,0.08); }}
.badge-neutral {{ color: {TEXT_MUTED}; border-color: {BORDER}; background: {PANEL_ALT}; }}

.bar-track {{ background: {PANEL_ALT}; border-radius: 3px; height: 5px; width: 100%; }}
.bar-fill {{ height: 5px; border-radius: 3px; }}
.component-row {{ font-size: 12px; color: {TEXT_MUTED}; display: flex; justify-content: space-between; margin-top: 2px; }}

.flag-banner {{
    border: 1px solid rgba(224,89,107,0.4);
    background: rgba(224,89,107,0.08);
    color: {BAD};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    margin-top: 8px;
}}

section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}

/* Streamlit's own chrome (top header/toolbar) defaults to light and was
   clashing with the dark body below it -- that's the "white bar" seen
   above the dark UI. config.toml sets [theme] base="dark" which handles
   most native widgets, but the header needs an explicit override too. */
header[data-testid="stHeader"] {{ background: {BG}; }}
div[data-testid="stToolbar"] {{ background: {BG}; }}
.stApp > header {{ background: {BG} !important; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999


def tier_color(v: float) -> str:
    if v >= 0.65:
        return GOOD
    if v >= 0.40:
        return WARN
    return BAD


def badge(text: str, level: str = "neutral") -> str:
    return f'<span class="badge badge-{level}">{text}</span>'


def component_bar(label: str, value: float, weight: float | None = None) -> str:
    pct = int(max(0.0, min(1.0, value)) * 100)
    color = tier_color(value)
    weight_str = f" &times;{weight:.0%}" if weight is not None else ""
    return f"""
    <div style="margin:6px 0">
      <div class="component-row">
        <span>{label}<span style="opacity:0.6">{weight_str}</span></span>
        <span class="mono" style="color:{color};font-weight:600">{value:.3f}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>"""


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _read_candidates_file(p: Path) -> list[dict]:
    """Supports .json (array), .jsonl (one object per line), and .jsonl.gz."""
    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if p.suffixes[-2:] == [".jsonl", ".gz"] or p.suffix == ".gz":
        rows = []
        with gzip.open(p, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    # .jsonl or any other plain-text line-delimited file
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@st.cache_resource
def load_candidates_cached():
    """
    Loads the real dataset if it's present, falling back to the small
    sample only if the real file can't be found. Checked in this order:
      1. India_runs_data_and_ai_challenge/candidates.jsonl (organizer layout)
      2. candidates.jsonl / candidates.json next to this app
      3. sample_candidates.json (fallback -- small demo set)
    Returns (candidates, source_label, source_path).
    """
    here = Path(__file__).parent
    candidates_dir = here / "India_runs_data_and_ai_challenge"

    search_order = [
        candidates_dir / "candidates.jsonl",
        candidates_dir / "candidates.jsonl.gz",
        candidates_dir / "candidates.json",
        here / "candidates.jsonl",
        here / "candidates.jsonl.gz",
        here / "candidates.json",
    ]
    for p in search_order:
        if p.exists():
            return _read_candidates_file(p), "full dataset", p.name

    sample_p = here / "sample_candidates.json"
    if sample_p.exists():
        with open(sample_p) as f:
            return json.load(f), "sample (50 candidates)", sample_p.name

    return [], "none", ""


@st.cache_data(show_spinner="Parsing uploaded dataset...")
def parse_uploaded_candidates(uploaded_file) -> list[dict]:
    """
    Parses a user-uploaded candidates file (.json / .jsonl / .jsonl.gz).
    Exists for deployments where the real dataset can't be bundled into the
    repo/sandbox (e.g. a 55MB+ candidates.jsonl exceeds what's practical to
    push or host) -- the user drags the file in at runtime instead.

    @st.cache_data hashes Streamlit's UploadedFile by its content, so
    re-running the app without changing the uploaded file won't re-parse it;
    uploading a genuinely different file correctly invalidates this cache.
    """
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".json"):
        data = json.loads(content)
        return data if isinstance(data, list) else [data]

    if name.endswith(".gz"):
        import io
        rows = []
        with gzip.open(io.BytesIO(content), "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # .jsonl or unrecognized extension -- assume line-delimited JSON
    rows = []
    for line in content.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@st.cache_resource
def load_semantic_scores_cached(candidate_ids_tuple):
    """
    Load precomputed semantic scores using the SAME fixed-scale normalization
    as the offline ranker (nexus_ranker.normalize_similarity), so scores here
    match what the actual submission.csv would contain — no batch-dependent
    drift between the sandbox and the real run.
    """
    candidate_ids = list(candidate_ids_tuple)

    emb_p = ARTIFACTS_DIR / "candidate_embeddings.npy"
    jd_p = ARTIFACTS_DIR / "jd_embedding.npy"
    id_p = ARTIFACTS_DIR / "candidate_ids.json"

    if emb_p.exists() and jd_p.exists() and id_p.exists():
        try:
            cand_vecs = np.load(str(emb_p))
            jd_vec = np.load(str(jd_p))
            with open(id_p) as f:
                all_ids = json.load(f)
            id_to_row = {cid: i for i, cid in enumerate(all_ids)}
            sims = (cand_vecs @ jd_vec.T).flatten()
            sims_norm = normalize_similarity(sims, ST_EMBEDDING_SIM_CEILING)
            result = {cid: float(sims_norm[id_to_row[cid]]) if cid in id_to_row else 0.0
                      for cid in candidate_ids}
            return result, "sentence-transformers"
        except Exception:
            pass

    vp = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
    mp = ARTIFACTS_DIR / "tfidf_matrix.npz"
    jp = ARTIFACTS_DIR / "tfidf_jd_vec.npz"
    ip = ARTIFACTS_DIR / "candidate_ids.json"

    if vp.exists() and mp.exists() and jp.exists() and ip.exists():
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            with open(vp, "rb") as f:
                vectorizer = pickle.load(f)
            cand_mat = sp.load_npz(str(mp))
            jd_vec = sp.load_npz(str(jp))
            with open(ip) as f:
                all_ids = json.load(f)
            id_to_row = {cid: i for i, cid in enumerate(all_ids)}
            sims = cosine_similarity(jd_vec, cand_mat).flatten()
            sims_norm = normalize_similarity(sims, TFIDF_SIM_CEILING)
            result = {cid: float(sims_norm[id_to_row[cid]]) if cid in id_to_row else 0.0
                      for cid in candidate_ids}
            return result, "TF-IDF (fixed-scale)"
        except Exception:
            pass

    return {cid: 0.0 for cid in candidate_ids}, "none"


@st.cache_data
def score_all_candidates(_candidates, _sem_scores, cache_key: str):
    """
    `cache_key` exists because `_candidates` and `_sem_scores` are both
    underscore-prefixed (Streamlit skips hashing them, since hashing a
    100K-row list on every rerun would be slow). Without SOME real,
    hashable argument to key on, this cache would silently return the
    FIRST dataset's scored results forever, even after switching to a
    different uploaded file -- there'd be nothing telling Streamlit the
    input actually changed. `cache_key` (built from filename + row count)
    is what makes cache invalidation actually work when the dataset changes.
    """
    results = []
    for c in _candidates:
        cid = c["candidate_id"]
        sem = _sem_scores.get(cid, 0.0)
        s = compute_nexus_score(c, semantic_score=sem)
        results.append({"candidate": c, "scores": s, "final_score": s["final_score"]})
    results.sort(key=lambda x: (-x["final_score"], x["candidate"]["candidate_id"]))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(f"""
    <div class="app-header">
      <div class="app-title">NEXUS &nbsp;/&nbsp; Candidate Intelligence</div>
      <div class="app-subtitle">Senior AI Engineer, Founding Team — Redrob AI &nbsp;·&nbsp; Neural + Expert Cross-signal Unified Scoring</div>
      <div>
        <span class="app-tag">SEMANTIC SIMILARITY</span>
        <span class="app-tag">CAREER ARC</span>
        <span class="app-tag">SKILL CLUSTERS</span>
        <span class="app-tag">BEHAVIORAL SIGNALS</span>
        <span class="app-tag">ANOMALY DETECTION</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("#### Dataset")
        uploaded_file = st.file_uploader(
            "Upload candidates file (.jsonl / .json / .jsonl.gz)",
            type=["jsonl", "json", "gz"],
            help="Use this if the real dataset isn't bundled with this deployment "
                 "(large files aren't pushed to the sandbox repo). Drag and drop, "
                 "or click to browse.",
        )
        st.markdown("---")

    if uploaded_file is not None:
        candidates = parse_uploaded_candidates(uploaded_file)
        data_source, data_filename = "uploaded file", uploaded_file.name
    else:
        candidates, data_source, data_filename = load_candidates_cached()

    if not candidates:
        st.error(
            "No candidate data found. Upload a candidates file above, or place "
            "candidates.jsonl (or sample_candidates.json) in "
            "India_runs_data_and_ai_challenge/ or alongside this app."
        )
        return

    st.caption(f"Dataset: {data_source} — {data_filename} ({len(candidates):,} candidates)")

    sem_scores, sem_method = load_semantic_scores_cached(tuple(c["candidate_id"] for c in candidates))
    score_cache_key = f"{data_filename}:{len(candidates)}"
    results = score_all_candidates(candidates, sem_scores, score_cache_key)

    # ---- Sidebar controls ----
    # Intentionally minimal: the system does the ranking (that's the point of
    # the challenge -- "delivering an expertly ranked shortlist", not a
    # user-tunable re-sort). Filters here only narrow the view of an
    # already-final ranking; they don't change how candidates are scored.
    with st.sidebar:
        st.markdown("#### Controls")
        search = st.text_input("Search title or company", "")
        view_mode = st.radio("View", ["Cards", "Table"], horizontal=True)
        top_n = st.slider("Show top N", 5, min(50, len(results)), 20)
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
        hide_flagged = st.checkbox("Hide flagged profiles", value=True)

        st.markdown("---")
        st.markdown("#### Scoring weights")
        for k, v in WEIGHTS.items():
            st.markdown(
                f'<div class="component-row"><span>{k.replace("_"," ").title()}</span>'
                f'<span class="mono">{v:.0%}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(f'<span class="badge badge-neutral">Semantic: {sem_method}</span>', unsafe_allow_html=True)
        if sem_method == "none":
            st.warning("Run nexus_precompute.py to enable semantic scoring.")

    # ---- Filter ----
    # `results` is already ranked by final_score (see score_all_candidates);
    # filtering below only narrows the view, it never re-ranks.
    filtered = [r for r in results if r["final_score"] >= min_score]
    if hide_flagged:
        filtered = [r for r in filtered if not r["scores"]["is_honeypot"]]
    if search.strip():
        q = search.strip().lower()
        filtered = [
            r for r in filtered
            if q in r["candidate"]["profile"].get("current_title", "").lower()
            or q in r["candidate"]["profile"].get("current_company", "").lower()
        ]
    filtered = filtered[:top_n]

    # ---- Metrics row ----
    hp_count = sum(1 for r in results if r["scores"]["is_honeypot"])
    top_score = results[0]["final_score"] if results else 0
    avg_top10 = float(np.mean([r["final_score"] for r in results[:10]])) if len(results) >= 10 else 0
    p10 = results[max(0, min(9, len(results) - 1))]["final_score"] if results else 0

    cols = st.columns(5)
    metrics = [
        ("Candidates scored", f"{len(candidates):,}"),
        ("Matching filters", str(len(filtered))),
        ("Flagged profiles", str(hp_count)),
        ("Top score", f"{top_score:.3f}"),
        ("Rank-10 score", f"{p10:.3f}"),
    ]
    for col, (lbl, val) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-num">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    tab_ranked, tab_analytics, tab_methodology = st.tabs(["Ranked candidates", "Analytics", "Methodology"])

    # ======================= RANKED CANDIDATES TAB =======================
    with tab_ranked:
        if view_mode == "Table":
            rows = []
            for rank, item in enumerate(filtered, 1):
                c, s = item["candidate"], item["scores"]
                p = c.get("profile", {})
                rows.append({
                    "Rank": rank,
                    "Candidate ID": c["candidate_id"],
                    "Title": p.get("current_title", "?"),
                    "Company": p.get("current_company", "?"),
                    "YoE": round(p.get("years_of_experience", 0), 1),
                    "Location": p.get("location", ""),
                    "Score": round(item["final_score"], 4),
                    "Career Arc": s["career_arc"],
                    "Skills": s["skill_clusters"],
                    "Semantic": s["semantic"],
                    "Flagged": "Yes" if s["is_honeypot"] else "",
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(700, 44 * (len(df) + 1)),
            )
        else:
            for rank, item in enumerate(filtered, 1):
                c = item["candidate"]
                s = item["scores"]
                p = c.get("profile", {})
                sig = c.get("redrob_signals", {})

                is_flagged = s["is_honeypot"]
                is_top = rank <= 10 and not is_flagged
                if is_flagged:
                    container_key = f"nxcardflag{rank}"
                elif is_top:
                    container_key = f"nxcardtop{rank}"
                else:
                    container_key = f"nxcard{rank}"

                card_ctx = st.container(key=container_key)
                card_ctx.__enter__()
                col_info, col_score = st.columns([5, 1])

                with col_info:
                    flag_tag = badge("FLAGGED", "bad") if is_flagged else ""
                    st.markdown(
                        f'<span class="rank-label">#{rank}</span> '
                        f'<span class="candidate-name">{c["candidate_id"]} — '
                        f'{p.get("current_title","?")} @ {p.get("current_company","?")}</span> {flag_tag}',
                        unsafe_allow_html=True,
                    )
                    yoe = p.get("years_of_experience", 0)
                    notice = sig.get("notice_period_days", "?")
                    st.markdown(
                        f'<div class="candidate-meta">{p.get("current_industry","?")} &middot; '
                        f'{yoe:.1f} yrs &middot; {p.get("location","")} {p.get("country","")} &middot; '
                        f'Notice: {notice}d</div>',
                        unsafe_allow_html=True,
                    )

                with col_score:
                    color = tier_color(item["final_score"])
                    st.markdown(
                        f'<div class="score-pill" style="background:{color}18;border:1px solid {color}55">'
                        f'<div class="score-pill-num" style="color:{color}">{item["final_score"]:.3f}</div>'
                        f'<div class="score-pill-lbl" style="color:{color}">score</div></div>',
                        unsafe_allow_html=True,
                    )

                reasoning = generate_nexus_reasoning(c, s, rank)
                st.markdown(f'<div class="reasoning-text">{reasoning}</div>', unsafe_allow_html=True)

                badges = []
                otw = sig.get("open_to_work_flag", False)
                badges.append(badge("Open to work", "good") if otw else badge("Passive candidate", "neutral"))

                di = _days_since(sig.get("last_active_date", "2020-01-01"))
                if di <= 14:
                    badges.append(badge(f"Active {di}d ago", "good"))
                elif di <= 60:
                    badges.append(badge(f"Active {di}d ago", "warn"))
                else:
                    badges.append(badge(f"Inactive {di}d", "bad"))

                gh = sig.get("github_activity_score", -1)
                if gh >= 0:
                    badges.append(badge(f"GitHub {gh:.0f}/100", "good" if gh >= 50 else "warn"))

                rrr = sig.get("recruiter_response_rate", 0)
                badges.append(badge(f"Response rate {rrr:.0%}", "good" if rrr >= 0.6 else "warn"))

                if sig.get("linkedin_connected"):
                    badges.append(badge("LinkedIn verified", "neutral"))

                sal = sig.get("expected_salary_range_inr_lpa", {})
                if isinstance(sal, dict) and sal.get("min"):
                    badges.append(badge(f"{sal['min']:.0f}-{sal['max']:.0f} LPA", "neutral"))

                sus = s["suspicion_score"]
                if sus > 0:
                    badges.append(badge(f"Suspicion {sus:.2f}", "bad" if sus >= 0.55 else "warn"))

                st.markdown("".join(badges), unsafe_allow_html=True)

                with st.expander(
                    f"Breakdown — arc {s['career_arc']:.2f} · skills {s['skill_clusters']:.2f} "
                    f"· semantic {s['semantic']:.2f} · behavioral x{s['behavioral_multiplier']:.2f}"
                ):
                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown("**Score components**")
                        st.markdown(component_bar("Career arc", s["career_arc"], WEIGHTS["career_arc"]), unsafe_allow_html=True)
                        st.markdown(component_bar("Skill clusters", s["skill_clusters"], WEIGHTS["skill_clusters"]), unsafe_allow_html=True)
                        st.markdown(component_bar("Semantic", s["semantic"], WEIGHTS["semantic"]), unsafe_allow_html=True)
                        st.markdown(component_bar("Experience", s["experience"], WEIGHTS["experience"]), unsafe_allow_html=True)
                        st.markdown(component_bar("Location", s["location"], WEIGHTS["location"]), unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="component-row" style="margin-top:8px">'
                            f'<span>Behavioral calibration</span>'
                            f'<span class="mono" style="font-weight:600">&times;{s["behavioral_multiplier"]:.3f}</span></div>'
                            f'<div class="component-row"><span>Base score</span>'
                            f'<span class="mono">{s["base_score"]:.4f}</span></div>',
                            unsafe_allow_html=True,
                        )
                    with b2:
                        st.markdown("**Skill cluster coverage**")
                        cs = s.get("cluster_scores", {})
                        for cname, cl in SKILL_CLUSTERS.items():
                            st.markdown(component_bar(cl["label"], cs.get(cname, 0.0)), unsafe_allow_html=True)

                if is_flagged and s["anti_pattern_flags"]:
                    st.markdown(
                        f'<div class="flag-banner">Anomaly flags: {"; ".join(s["anti_pattern_flags"][:3])}</div>',
                        unsafe_allow_html=True,
                    )

                card_ctx.__exit__(None, None, None)

    # ======================= ANALYTICS TAB =======================
    with tab_analytics:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Score distribution — top 100**")
            try:
                import plotly.graph_objects as go
                scores_list = [r["final_score"] for r in results[:min(100, len(results))]]
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=scores_list, nbinsx=20, marker_color=ACCENT, opacity=0.85))
                fig.update_layout(
                    paper_bgcolor=PANEL, plot_bgcolor=BG,
                    font=dict(color=TEXT_MUTED, family="Inter"),
                    margin=dict(l=20, r=20, t=10, b=30),
                    height=280,
                    xaxis_title="Final score", yaxis_title="Count",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for the score-distribution chart.")

        with col_b:
            st.markdown("**Average skill cluster coverage — top 10**")
            try:
                import plotly.graph_objects as go
                cluster_avgs = {
                    cl["label"]: float(np.mean([r["scores"]["cluster_scores"].get(cname, 0) for r in results[:10]]))
                    for cname, cl in SKILL_CLUSTERS.items()
                }
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=list(cluster_avgs.values()), y=list(cluster_avgs.keys()),
                    orientation="h", marker_color=ACCENT,
                ))
                fig2.update_layout(
                    paper_bgcolor=PANEL, plot_bgcolor=BG,
                    font=dict(color=TEXT_MUTED, family="Inter"),
                    margin=dict(l=10, r=20, t=10, b=30),
                    height=280,
                    xaxis=dict(range=[0, 1]),
                )
                st.plotly_chart(fig2, use_container_width=True)
            except ImportError:
                st.info("Install plotly for the cluster-coverage chart.")

        st.markdown("**Industry distribution — top 100**")
        ind_counts: dict[str, int] = {}
        for r in results[:100]:
            ind = r["candidate"].get("profile", {}).get("current_industry", "unknown")
            ind_counts[ind] = ind_counts.get(ind, 0) + 1
        ind_df = pd.DataFrame(sorted(ind_counts.items(), key=lambda x: -x[1]), columns=["Industry", "Count"])
        st.dataframe(ind_df, use_container_width=True, hide_index=True)

    # ======================= METHODOLOGY TAB =======================
    with tab_methodology:
        st.markdown("""
**Five scoring layers, combined as a weighted sum, then calibrated:**

| Layer | Weight | What it captures |
|---|---|---|
| Career Arc | 30% | Time-decayed trajectory — recent AI/ML roles at product companies count far more than roles from years ago |
| Skill Clusters | 25% | Five JD-derived skill groups, scored for depth (not flat keyword presence) |
| Semantic Similarity | 25% | Text similarity to the JD, normalized on a fixed scale so scores don't shift depending on which other candidates are scored alongside them |
| Experience | 12% | Years-of-experience band, applied-ML ratio, education tier |
| Location & Logistics | 8% | Geographic fit, notice period, work-mode preference |

**Behavioral calibration** — a weighted average of all 23 `redrob_signals`,
mapped to a bounded 0.85x–1.20x multiplier. This breaks ties in favor of
active, responsive, verified candidates without letting behavioral data
override actual job fit.

**Anomaly detection** — seven rule-based checks (experience/tenure
mismatches, impossible skill claims, overlapping employment, and similar).
Suspicion is graded by the number of *distinct* anomaly categories
triggered, so one systemic issue doesn't unfairly flag a candidate; three or
more distinct categories caps the score.
        """)

    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;color:{TEXT_MUTED};font-size:11px;padding:8px" class="mono">'
        "NEXUS — Career Arc 30% · Skill Clusters 25% · Semantic 25% · Experience 12% · Location 8% "
        "· behavioral calibration applied as a bounded multiplier"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()