"""
Error Analysis Script — LLM-as-a-Judge Diagnosis of Failed Queries

Loads per-query results from the optimal config (E1_L1_D2_S2), extracts
the bottom-15 queries with NDCG@5 = 0.000, and uses DeepSeek API to
classify each failure into one of six failure modes with a detailed
root-cause explanation.

Methodology follows Report Section 4.6.1.

Usage:
    python experiments/error_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root so we can import deepseek_api
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # AA_Assignment3_DongB
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from deepseek_api import DeepSeekClient  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_NAME = "E2_L1_D2_S2"
INPUT_FILE = Path(__file__).resolve().parent / "results" / f"{CONFIG_NAME}_deepseek-multi_per_query.json"
OUTPUT_FILE = Path(__file__).resolve().parent / "results" / f"{CONFIG_NAME}_error_analysis.json"
NUM_FAILED_TO_ANALYZE = 15

FAILURE_MODES = [
    "Genre Confusion",
    "Tag Blindness",
    "Semantic Ambiguity",
    "Review Dilution",
    "Cold Game Gap",
    "Other",
]

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert evaluator for a game-recommendation system.
Your task is to analyse a FAILED recommendation (NDCG@5 = 0.0 — none of the
top-5 recommended games matched the user's ground-truth preferences) and
diagnose WHY the system failed.

For each case you will receive:
- The user's natural-language query
- The top-5 recommended games (name, relevance score, genres, tags, short
  description)
- The ground-truth games (name, genres)

You must produce a JSON object with exactly three fields:
1. "relevance_rating" — integer 1-5: how relevant the top-5 recommendations
   COLLECTIVELY are to the query intent (1 = completely irrelevant,
   5 = perfect match)
2. "failure_mode" — string, exactly one of:
   "Genre Confusion"      — system returned wrong genres
   "Tag Blindness"        — system ignored important tag-level semantics
   "Semantic Ambiguity"   — query was ambiguous and system resolved it
                             incorrectly
   "Review Dilution"      — positive reviews misled the system about the
                             game's actual content
   "Cold Game Gap"        — query described a game concept that barely
                             exists in the catalogue
   "Other"                — failure does not fit the above categories
3. "root_cause_explanation" — a concise (1-3 sentence) human-readable
   analysis of the failure.

IMPORTANT — "Other" elaboration rule:
If you classify a failure as "Other", you MUST provide a fine-grained,
specific sub-type within the root_cause_explanation field.  Do NOT leave
it as a generic "Other" statement.  Examples of elaborated "Other" types:
  "Price Mismatch — recommended games are all AAA titles while the query
   implies a preference for indie/budget games"
  "Platform Mismatch — query seeks multiplayer but top picks are all
   single-player"
  "Tone Mismatch — query asks for light-hearted casual games but
   recommendations are dark/horror themed"
  "Content Filter Issue — recommendations share the query's genre but
   diverge in age rating or thematic content"
  "Niche Concept — the query describes a very specific hybrid concept
   that the catalogue lacks entirely"

Return ONLY a valid JSON object, no other text."""


def build_user_prompt(query_record: dict) -> str:
    """Build the evaluation prompt for a single failed query."""
    query_text = query_record["query"]
    top5 = query_record["top5"]
    gt_names = query_record.get("gt_names", [])
    gt_ids = query_record.get("gt_ids", [])

    # Format top-5 recommendations
    rec_lines = []
    for i, game in enumerate(top5, 1):
        name = game.get("name", "Unknown")
        score = game.get("score", 0.0)
        genres = ", ".join(game.get("genres", [])) or "(none)"
        tags = ", ".join(game.get("tags", [])) or "(none)"
        desc = game.get("short_description", "")[:200]
        price = game.get("price", "?")
        rec_lines.append(
            f"  #{i} | {name} | score={score:.2f} | ${price}\n"
            f"      Genres: {genres}\n"
            f"      Tags: {tags}\n"
            f"      Description: {desc}"
        )

    # Format ground truth
    gt_lines = []
    for i, name in enumerate(gt_names, 1):
        gt_lines.append(f"  {i}. {name}")

    prompt = f"""USER QUERY:
"{query_text}"

TOP-5 RECOMMENDED GAMES:
{chr(10).join(rec_lines)}

GROUND TRUTH GAMES (what the user actually wanted):
{chr(10).join(gt_lines) if gt_lines else '  (no ground truth provided)'}

Analyse the failure. Output ONLY a JSON object with fields:
relevance_rating, failure_mode, root_cause_explanation"""

    return prompt


def extract_json_from_response(text: str) -> dict | None:
    """Robust JSON extraction from LLM response."""
    import re

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def main():
    # ------------------------------------------------------------------
    # 1. Load per-query results
    # ------------------------------------------------------------------
    print(f"Loading per-query results from: {INPUT_FILE}")
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    print(f"  Loaded {len(all_queries)} queries.")

    # ------------------------------------------------------------------
    # 2. Identify failed queries (NDCG@5 = 0.0)
    # ------------------------------------------------------------------
    failed = [q for q in all_queries if q.get("ndcg@5", 1.0) == 0.0]
    # Sort by query_index for deterministic order, then take bottom-15
    failed.sort(key=lambda q: q["query_index"])
    failed = failed[:NUM_FAILED_TO_ANALYZE]

    print(f"  Found {len(failed)} queries with NDCG@5 = 0.000")
    print(f"  Analyzing bottom-{NUM_FAILED_TO_ANALYZE} failed queries.\n")

    if not failed:
        print("No failed queries to analyse. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Initialize DeepSeek client
    # ------------------------------------------------------------------
    print("Initializing DeepSeek client...")
    try:
        client = DeepSeekClient()
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Please create a .env file at the project root with your DEEPSEEK_API_KEY.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Run LLM-as-a-Judge on each failed query
    # ------------------------------------------------------------------
    per_query_results = []
    failure_counts = {mode: 0 for mode in FAILURE_MODES}
    other_elaborations = []
    total_relevance = 0.0

    for idx, record in enumerate(failed):
        query_text = record["query"]
        query_index = record["query_index"]
        print(f"[{idx + 1}/{len(failed)}] Query #{query_index}: {query_text[:100]}...")

        prompt = build_user_prompt(record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = client.chat(messages, temperature=0.2, max_tokens=1024)
        except Exception as e:
            print(f"  ERROR calling DeepSeek API: {e}")
            response = ""

        result = extract_json_from_response(response)

        if result is None:
            print(f"  WARNING: Could not parse JSON. Raw response: {response[:200]}")
            result = {
                "relevance_rating": 1,
                "failure_mode": "Other",
                "root_cause_explanation": f"LLM response parsing failed. Raw: {response[:300]}",
            }

        # Validate and normalise
        rating = result.get("relevance_rating", 1)
        try:
            rating = int(rating)
            rating = max(1, min(5, rating))
        except (ValueError, TypeError):
            rating = 1

        fm = result.get("failure_mode", "Other")
        if fm not in FAILURE_MODES:
            # Try fuzzy match
            fm_lower = fm.lower()
            matched = None
            for valid_fm in FAILURE_MODES:
                if valid_fm.lower() in fm_lower or fm_lower in valid_fm.lower():
                    matched = valid_fm
                    break
            fm = matched if matched else "Other"
            print(f"  (normalised failure_mode '{result.get('failure_mode')}' → '{fm}')")

        rce = result.get("root_cause_explanation", "No explanation provided.")

        # Handle "Other" elaboration
        if fm == "Other":
            other_elaborations.append({
                "query_index": query_index,
                "query": query_text[:120],
                "sub_type": rce,
            })

        failure_counts[fm] += 1
        total_relevance += rating

        per_query_results.append({
            "query_index": query_index,
            "query": query_text,
            "relevance_rating": rating,
            "failure_mode": fm,
            "root_cause_explanation": rce,
            "top5_recommendations": [
                {
                    "name": g.get("name"),
                    "score": g.get("score"),
                    "genres": g.get("genres", []),
                    "tags": g.get("tags", []),
                }
                for g in record.get("top5", [])
            ],
            "ground_truth": [
                {"name": name}
                for name in record.get("gt_names", [])
            ],
        })

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # 5. Compute summary statistics
    # ------------------------------------------------------------------
    mean_relevance = total_relevance / len(per_query_results) if per_query_results else 0.0

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nConfig: {CONFIG_NAME}")
    print(f"Failed queries analysed: {len(per_query_results)}")
    print(f"Mean relevance rating: {mean_relevance:.2f} / 5.0")
    print(f"\nFailure Mode Distribution:")
    for mode in FAILURE_MODES:
        count = failure_counts[mode]
        pct = count / len(per_query_results) * 100 if per_query_results else 0
        bar = "█" * count
        print(f"  {mode:<25s} {count:>2d} ({pct:5.1f}%)  {bar}")

    if other_elaborations:
        print(f"\n'Other' elaborations:")
        for elab in other_elaborations:
            print(f"  Q#{elab['query_index']}: {elab['sub_type']}")

    print(f"\n{'=' * 70}")
    print("Per-query details:")
    print(f"{'=' * 70}")
    for r in per_query_results:
        print(f"  Q#{r['query_index']:>2d} | rating={r['relevance_rating']} | "
              f"mode={r['failure_mode']}")
        print(f"       Root cause: {r['root_cause_explanation']}")
        print(f"       Top-5: {', '.join(g['name'] for g in r['top5_recommendations'])}")
        print()

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    output_data = {
        "config": CONFIG_NAME,
        "num_failed_queries_analyzed": len(per_query_results),
        "mean_relevance_rating": round(mean_relevance, 2),
        "failure_mode_distribution": failure_counts,
        "other_elaborations": other_elaborations,
        "per_query": per_query_results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()