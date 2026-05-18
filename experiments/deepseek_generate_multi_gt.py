"""
Step 2: Generate multi-ground-truth labels using DeepSeek's own judgment.

Critical design principle: NO circular dependency.
  - Round 1: DeepSeek scans 5000 game names → selects ~100 candidates (names only)
  - Round 2: DeepSeek reviews candidates' full details → judges which are relevant
  - DeepSeek is NOT part of our RAG pipeline (not nomic/all-MiniLM/gemma/phi)
  - No vector search, no tag matching — purely DeepSeek's semantic understanding

Process:
  1. Load the 100 DeepSeek-generated queries (from Step 1)
  2. Load 5000 game names → build catalog for Round 1
     NOTE: Uses the EXACT same load_games_from_sqlite('top_reviewed')
     as Step 1, ensuring both steps see the SAME 5000 games.
  3. For each query:
     Round 1: DeepSeek picks ~100 potentially relevant games from 5000 names
     Round 2: DeepSeek judges which of ~100 are truly relevant → multi-GT
  4. Save to raglooker/data/synthetic_queries_deepseek_multi_gt.json

Usage:
  python deepseek_generate_multi_gt.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from experiments.deepseek_api import DeepSeekClient
from raglooker.steam_sqlite import load_games_from_sqlite

# Paths
DB_PATH = BASE_DIR / "raglooker" / "steam_games_reviews_25.sqlite"
QUERIES_PATH = BASE_DIR / "raglooker" / "data" / "synthetic_queries_deepseek.json"
OUTPUT_PATH = BASE_DIR / "raglooker" / "data" / "synthetic_queries_deepseek_multi_gt.json"
MAX_GAMES = 5000

# Round 1: number of candidates to select
ROUND1_CANDIDATE_COUNT = 100
# Round 2: candidate detail format
INCLUDE_DESCRIPTION = True


def load_game_catalog(max_games: int = 5000) -> dict:
    """
    Load all games from DB with their details.
    Uses the EXACT SAME load_games_from_sqlite('top_reviewed') as
    deepseek_generate_queries.py (Step 1), ensuring both steps
    see the same 5000 games → no more "not in catalog" errors.

    Returns:
      catalog: {app_id: {name, short_description, genres_str, tags_str}}
      name_list: [(app_id, name), ...] for Round 1 catalog text
    """
    # Use the same query as Step 1 — top_reviewed mode
    games = load_games_from_sqlite(DB_PATH, max_games, select_mode='top_reviewed')

    catalog = {}
    name_list = []

    for app_id, game in games:
        name = game.get("name", "Unknown")

        # Genres (already parsed by steam_sqlite)
        raw_genres = game.get("genres", [])
        if raw_genres and isinstance(raw_genres[0], dict):
            genres_str = ", ".join(g["description"] for g in raw_genres[:5])
        else:
            genres_str = ", ".join(str(g) for g in raw_genres[:5])

        # Tags (already parsed by steam_sqlite)
        raw_tags = game.get("tags", {})
        if isinstance(raw_tags, dict):
            top_tags = sorted(raw_tags.items(), key=lambda x: -x[1])[:8]
            tags_str = ", ".join(t[0] for t in top_tags)
        else:
            tags_str = ", ".join(str(t) for t in raw_tags[:8])

        catalog[app_id] = {
            "name": name,
            "short_description": game.get("short_description", "") or "",
            "genres_str": genres_str,
            "tags_str": tags_str,
        }
        name_list.append((app_id, name))

    print(f"Loaded {len(catalog)} games into catalog (top_reviewed, same as Step 1).")
    return catalog, name_list


def build_catalog_text(name_list: list[tuple[str, str]]) -> str:
    """Build a numbered list of game names for Round 1 prompt."""
    lines = []
    for i, (app_id, name) in enumerate(name_list, 1):
        lines.append(f"{i}. {name}")
    return "\n".join(lines)


def build_app_id_lookup(name_list: list[tuple[str, str]]) -> dict[int, str]:
    """Build a mapping: position (1-based) -> app_id"""
    return {i: app_id for i, (app_id, _) in enumerate(name_list, 1)}


def round1_select_candidates(
    client: DeepSeekClient,
    query: str,
    catalog_text: str,
    lookup: dict[int, str],
    top_n: int = 100,
) -> list[str]:
    """
    Round 1: DeepSeek scans all 5000 game names and selects ~top_n candidates.

    Only game names are provided — no descriptions, no genres, no tags.
    This is purely a name-based relevance filter by DeepSeek's understanding.
    """
    prompt = (
        f"A Steam user is searching for:\n\"{query}\"\n\n"
        f"Below is a numbered list of 5000 Steam games (names only).\n"
        f"From these, select the {top_n} games that are MOST LIKELY to match this search.\n"
        f"Think about genre keywords, game themes, and naming conventions.\n\n"
        f"{catalog_text}\n\n"
        f"Return ONLY a JSON array of the selected line numbers, like:\n"
        f"[12, 45, 78, 103, ...]\n"
        f"Select exactly {top_n} candidates."
    )

    try:
        response = client.chat([
            {"role": "system", "content": "You are a Steam game expert selecting candidate games. Output only JSON numbers."},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=2048)

        indices = client.extract_json_array(response)
        if not indices:
            print("  ⚠ Round 1: empty response, keeping original GT only")
            return []

        # Convert indices to app_ids
        candidate_ids = []
        for idx in indices:
            if isinstance(idx, int) and idx in lookup:
                candidate_ids.append(lookup[idx])
            elif isinstance(idx, (str, float)):
                try:
                    int_idx = int(idx)
                    if int_idx in lookup:
                        candidate_ids.append(lookup[int_idx])
                except (ValueError, TypeError):
                    continue

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for aid in candidate_ids:
            if aid not in seen:
                seen.add(aid)
                unique.append(aid)

        print(f"  Round 1: selected {len(unique)} candidates from 5000", end="")
        return unique[:top_n]

    except Exception as e:
        print(f"  ⚠ Round 1 error: {e}")
        return []


def round2_judge_relevance(
    client: DeepSeekClient,
    query: str,
    candidate_ids: list[str],
    catalog: dict,
    original_gt_id: str,
) -> list[str]:
    """
    Round 2: DeepSeek reviews candidate details and judges relevance.

    Args:
        candidate_ids: list of app_ids from Round 1 (or original GT if Round 1 failed)
        catalog: full game details lookup
        original_gt_id: the original ground truth app_id

    Returns:
        list of app_ids judged relevant (including original GT)
    """
    if not candidate_ids:
        return [original_gt_id]

    # Build candidate details
    lines = []
    for i, app_id in enumerate(candidate_ids, 1):
        info = catalog.get(app_id, {})
        name = info.get("name", "Unknown")
        desc = info.get("short_description", "")[:150] if INCLUDE_DESCRIPTION else ""
        genres = info.get("genres_str", "")
        tags = info.get("tags_str", "")

        if INCLUDE_DESCRIPTION and desc:
            lines.append(
                f"{i}. \"{name}\"\n"
                f"   Genres: {genres}\n"
                f"   Tags: {tags}\n"
                f"   Description: {desc}"
            )
        else:
            lines.append(
                f"{i}. \"{name}\" — {genres}"
            )

    candidates_text = "\n\n".join(lines)
    total = len(candidate_ids)

    prompt = (
        f"A Steam user is looking for games matching this search:\n"
        f"\"{query}\"\n\n"
        f"Below are {total} candidate games. For EACH game, decide:\n"
        f"- YES: This game is GENUINELY a good match for the search query\n"
        f"- NO: This game is NOT a relevant match\n\n"
        f"Be strict but fair. A game is relevant if it shares the core genre, "
        f"theme, gameplay mechanics, or vibe described in the query.\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        f"Return ONLY a JSON array of the line numbers marked YES:\n"
        f"[1, 3, 7, 12, ...]\n"
        f"Include at least 1 game, at most {total}."
    )

    try:
        response = client.chat([
            {"role": "system", "content": "You judge game relevance strictly. Output only JSON numbers."},
            {"role": "user", "content": prompt},
        ], temperature=0.2, max_tokens=2048)

        indices = client.extract_json_array(response)
        if not indices:
            print(f"  ⚠ Round 2: empty response, using original GT only")
            return [original_gt_id]

        # Convert indices to app_ids
        relevant_ids = set()
        for idx in indices:
            if isinstance(idx, int) and 1 <= idx <= len(candidate_ids):
                relevant_ids.add(candidate_ids[idx - 1])
            elif isinstance(idx, (str, float)):
                try:
                    int_idx = int(idx)
                    if 1 <= int_idx <= len(candidate_ids):
                        relevant_ids.add(candidate_ids[int_idx - 1])
                except (ValueError, TypeError):
                    continue

        # Always include original GT
        if original_gt_id not in relevant_ids:
            relevant_ids.add(original_gt_id)

        result = list(relevant_ids)
        print(f"  Round 2: judged {len(result)}/{total} relevant", end="")
        return result

    except Exception as e:
        print(f"  ⚠ Round 2 error: {e}")
        return [original_gt_id]


def get_gt_candidate_names(
    ground_truth_ids: list[str],
    catalog: dict,
) -> list[str]:
    """Get readable names for ground truth IDs."""
    names = []
    for aid in ground_truth_ids:
        info = catalog.get(aid)
        if info:
            names.append(info.get("name", aid))
        else:
            names.append(aid)
    return names


def main():
    print("=" * 70)
    print("DEEPSEEK MULTI-GT GENERATOR — Step 2")
    print("Two-Round Expansion: Names → Candidates → Relevance Judgment")
    print("=" * 70)

    # Initialize DeepSeek client
    print("\nInitializing DeepSeek client...")
    try:
        client = DeepSeekClient()
        print(f"  Model: {client.model}")
    except ValueError as e:
        print(f"\n❌ {e}")
        return

    # Step 1: Load queries
    print(f"\nStep 1: Loading DeepSeek-generated queries...")
    if not QUERIES_PATH.exists():
        print(f"❌ Queries file not found: {QUERIES_PATH}")
        print("   Run deepseek_generate_queries.py first.")
        return

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  Loaded {len(queries)} queries from {QUERIES_PATH}")

    # Step 2: Load game catalog
    print(f"\nStep 2: Loading game catalog ({MAX_GAMES} games)...")
    catalog, name_list = load_game_catalog(MAX_GAMES)

    # Build catalog text for Round 1
    catalog_text = build_catalog_text(name_list)
    lookup = build_app_id_lookup(name_list)
    catalog_token_est = len(catalog_text) // 4
    print(f"  Catalog text: ~{catalog_token_est}K estimated tokens")

    # Step 3: For each query, run Round 1 + Round 2
    print(f"\nStep 3: Generating multi-GT for {len(queries)} queries...")
    print(f"  Round 1: DeepSeek selects ~{ROUND1_CANDIDATE_COUNT} from 5000 names")
    print(f"  Round 2: DeepSeek judges relevance from details\n")

    new_queries = []
    total_gt_count = 0
    skip_count = 0

    for i, item in enumerate(queries):
        query = item["query"]
        original_gt = item["ground_truth_ids"]
        original_name = item.get("source_game_name", "?")
        original_app_id = original_gt[0] if original_gt else None

        # Check if original GT app_id is in catalog.
        # Since we now use the SAME loading logic as Step 1,
        # this should almost never happen.
        if not original_app_id or original_app_id not in catalog:
            print(f"  [{i+1:3d}/{len(queries):3d}] ⚠ SKIP: {original_name[:40]} (not in catalog)")
            skip_count += 1
            continue

        print(f"\n  [{i+1:3d}/{len(queries):3d}] \"{query[:60]}...\"")
        print(f"       Original GT: \"{original_name}\"")
        sys.stdout.flush()

        # ---- Round 1: Select candidates ----
        candidate_ids = round1_select_candidates(
            client, query, catalog_text, lookup, ROUND1_CANDIDATE_COUNT
        )

        if not candidate_ids:
            # Fallback: use original GT + nearest-name matches
            candidate_ids = [original_app_id]
            # Try adding games with similar names (simple keyword overlap)
            query_tokens = set(query.lower().split())
            for aid, info in catalog.items():
                if aid == original_app_id:
                    continue
                name_tokens = set(info["name"].lower().split())
                overlap = len(query_tokens & name_tokens)
                if overlap >= 2 and len(candidate_ids) < 10:
                    candidate_ids.append(aid)
            print(f"  Round 1: fallback to {len(candidate_ids)} keyword candidates")

        # ---- Round 2: Judge relevance ----
        relevant_ids = round2_judge_relevance(
            client, query, candidate_ids, catalog, original_app_id
        )

        # Build result entry
        gt_names = get_gt_candidate_names(relevant_ids, catalog)
        new_item = {
            "query": query,
            "ground_truth_ids": relevant_ids,
            "source_game_name": original_name,
            "original_gt_id": original_app_id,
            "total_gt_count": len(relevant_ids),
            "candidate_count": len(candidate_ids),
            "gt_names": gt_names,
        }
        new_queries.append(new_item)
        total_gt_count += len(relevant_ids)

        print(f"       → {len(relevant_ids)} GT games: {', '.join(gt_names[:5])}{'...' if len(gt_names) > 5 else ''}")
        sys.stdout.flush()

        # Rate limiting
        time.sleep(1.5)

    # Summary
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Queries processed:   {len(new_queries)}")
    print(f"  Queries skipped:     {skip_count}")
    print(f"  Total GT entries:    {total_gt_count}")
    avg_gt = total_gt_count / max(len(new_queries), 1)
    print(f"  Avg GT per query:    {avg_gt:.1f}")
    gt_gt1 = sum(1 for q in new_queries if q["total_gt_count"] > 1)
    print(f"  Queries with >1 GT:  {gt_gt1}/{len(new_queries)}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(new_queries, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {OUTPUT_PATH}")

    # Show samples
    print(f"\n--- Sample expanded queries ---")
    for item in new_queries[:5]:
        if item["total_gt_count"] > 1:
            print(f'  Query: "{item["query"][:60]}..."')
            print(f"    Original: {item['source_game_name']}")
            print(f"    Multi-GT ({item['total_gt_count']} games):")
            for name in item["gt_names"][:5]:
                print(f"      - {name}")
            if item["total_gt_count"] > 5:
                print(f"      ... and {item['total_gt_count'] - 5} more")
            print()

    print("=" * 70)
    print("NEXT: Update run_experiments.py to use the new test set")
    print("  Then: python experiments/run_experiments.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
