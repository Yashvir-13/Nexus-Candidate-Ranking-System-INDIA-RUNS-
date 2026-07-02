"""
nexus_reasoning.py — Rich, grounded reasoning for NEXUS ranked candidates.

Generates 1-3 sentence reasoning that:
- References specific facts from the candidate profile
- Cites which JD skill clusters are matched
- Acknowledges gaps honestly
- Varies tone by rank position
"""

from __future__ import annotations
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 15)

CLUSTER_LABELS = {
    "vector_retrieval": "Vector/Retrieval",
    "llm_rag":          "LLM/RAG",
    "ml_training":      "Model Training",
    "ml_production":    "ML Production",
    "python_data":      "Python/Data",
}

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
}


def _days_since(date_str: str) -> int:
    try:
        return (REFERENCE_DATE - datetime.strptime(date_str[:10], "%Y-%m-%d").date()).days
    except Exception:
        return 9999


def _top_clusters(cluster_scores: dict, n: int = 2) -> list[str]:
    sorted_c = sorted(cluster_scores.items(), key=lambda x: -x[1])
    return [CLUSTER_LABELS.get(k, k) for k, v in sorted_c[:n] if v > 0.1]


def _best_product_role(c: dict) -> str:
    for job in sorted(c.get("career_history", []),
                      key=lambda j: j.get("duration_months", 0), reverse=True):
        ind = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        is_product = ind not in {"it services", "manufacturing", "conglomerate"} \
                     and not any(f in company for f in CONSULTING_FIRMS)
        if is_product:
            return f"{job.get('title','')} at {job.get('company','')}"
    return ""


def _is_consulting_heavy(c: dict) -> bool:
    career = c.get("career_history", [])
    total = sum(j.get("duration_months", 0) for j in career) or 1
    cons = sum(j.get("duration_months", 0) for j in career
               if j.get("industry", "").lower() in {"it services"}
               or any(f in j.get("company", "").lower() for f in CONSULTING_FIRMS))
    return cons / total > 0.6


def generate_nexus_reasoning(c: dict, scores: dict, rank: int) -> str:
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    yoe = profile.get("years_of_experience", 0)

    notice = signals.get("notice_period_days", 90)
    otw = signals.get("open_to_work_flag", False)
    last_active = _days_since(signals.get("last_active_date", "2020-01-01"))
    gh = signals.get("github_activity_score", -1)

    cluster_scores = scores.get("cluster_scores", {})
    top_clusters = _top_clusters(cluster_scores)
    best_role = _best_product_role(c)
    consulting = _is_consulting_heavy(c)
    arc = scores.get("career_arc", 0)
    suspicion = scores.get("suspicion_score", 0)
    flags = scores.get("anti_pattern_flags", [])

    fscore = scores.get("final_score", 0.0)

    # Honeypot / high suspicion
    if suspicion >= 0.70:
        return (
            f"{title} with {yoe:.1f}yr; profile flagged for data inconsistencies "
            f"({'; '.join(flags[:2])}). "
            f"Ranked out of contention — possible synthetic or misleading profile."
        )

    # Tone is keyed to the ACTUAL SCORE, not rank position. The previous
    # version used `if rank <= 10` etc., which meant that on a small or
    # bimodal candidate pool, ranks 4-10 could still say "strong JD
    # coverage" even when their real score was 0.19 and every component
    # was near zero -- rank alone doesn't tell you whether a candidate is
    # actually strong, only that they beat the others in this batch. Score
    # thresholds keep the language honest regardless of how the rest of
    # the pool happens to be distributed.

    # Genuinely strong fit
    if fscore >= 0.55:
        parts = []
        lead = best_role if best_role else f"{title} at {company}"
        parts.append(lead)
        parts.append(f"{yoe:.1f}yr experience")
        if top_clusters:
            parts.append(f"strong JD coverage in {' & '.join(top_clusters)}")
        behavioral = []
        if otw:
            behavioral.append("actively looking")
        if last_active <= 14:
            behavioral.append(f"active {last_active}d ago")
        if notice <= 30:
            behavioral.append(f"{notice}d notice")
        if gh >= 50:
            behavioral.append(f"GitHub {gh:.0f}/100")
        if behavioral:
            parts.append("; ".join(behavioral))
        loc_str = location if country == "India" else f"{location}, {country}"
        parts.append(f"based in {loc_str}")

        s1 = ". ".join(parts[:3]) + "."
        s2 = ("; ".join(parts[3:]) + ".") if len(parts) > 3 else ""
        return (s1 + " " + s2).strip()

    # Moderate fit with real strengths, but caveats worth naming
    elif fscore >= 0.35:
        strengths, concerns = [], []
        if top_clusters:
            strengths.append(f"JD-aligned skills in {', '.join(top_clusters)}")
        if best_role and not consulting:
            strengths.append(f"product-company track record")
        if 5 <= yoe <= 9:
            strengths.append(f"{yoe:.1f}yr in ideal experience band")
        if consulting:
            concerns.append("career predominantly at services/consulting firms")
        if last_active > 60:
            concerns.append(f"inactive for {last_active}d")
        if notice > 90:
            concerns.append(f"long notice ({notice}d)")
        s1 = f"{title} — {', '.join(strengths[:2]) if strengths else 'adjacent background'}."
        s2 = f"Note: {'; '.join(concerns[:2])}." if concerns else \
             f"Behavioral signals {'strong' if scores.get('behavioral_multiplier',1)>=1 else 'moderate'}."
        return f"{s1} {s2}"

    # Weak-to-moderate fit -- rank #{rank} noted for context, not as praise
    elif fscore >= 0.18:
        sk_note = f"covers {', '.join(top_clusters)}" if top_clusters else "limited JD skill coverage"
        career_note = "product-company background" if not consulting else "services/consulting background"
        avail = (f"active {last_active}d ago, {notice}d notice"
                 if last_active < 90 else f"inactive {last_active}d, {notice}d notice")
        return (f"{title} ({yoe:.1f}yr), ranked #{rank} — {sk_note}; {career_note}. "
                f"Availability: {avail}.")

    # Weak fit -- honest, brief, no inflated language
    else:
        gaps = []
        if scores.get("skill_clusters", 0) < 0.15:
            gaps.append("minimal JD skill cluster coverage")
        if consulting:
            gaps.append("consulting-only career")
        if last_active > 120:
            gaps.append(f"inactive {last_active}d")
        if arc < 0.2:
            gaps.append("career arc not aligned with AI/ML")
        if not gaps:
            gaps.append("below threshold on multiple scoring dimensions")
        return (f"{title} ({yoe:.1f}yr); ranked #{rank} — {'; '.join(gaps[:2])}. "
                f"Included as boundary candidate.")