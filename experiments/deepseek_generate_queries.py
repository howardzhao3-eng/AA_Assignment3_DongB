"""
Step 1: Generate better test queries using DeepSeek v4 Pro.

Replaces the old generate_queries.py (which used gemma2:2b).
DeepSeek generates more realistic, diverse, and natural user search queries.

Process:
  1. Load top-5000 most-reviewed games from SQLite
  2. Sample ~100 diverse games (stratified by genre)
  3. For each game, DeepSeek reverse-generates a realistic user query
  4. Save to raglooker/data/synthetic_queries_deepseek.json

Usage:
  cd experiments
  python deepseek_generate_queries.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from raglooker.steam_sqlite import load_games_from_sqlite
from experiments.deepseek_api import DeepSeekClient

# Paths
DB_PATH = BASE_DIR / "raglooker" / "steam_games_reviews_25.sqlite"
OUTPUT_PATH = BASE_DIR / "raglooker" / "data" / "synthetic_queries_deepseek.json"
TARGET_COUNT = 100
MAX_GAMES = 5000


def load_sample_games(max_games: int = 5000, target: int = 100) -> list[tuple[str, dict]]:
    """Load games and try to sample diverse ones across genres."""
    print(f"Loading up to {max_games} games from SQLite (top_reviewed)...")
    games = load_games_from_sqlite(DB_PATH, max_games, select_mode='top_reviewed')
    print(f"Loaded {len(games)} games.")

    # Group by primary genre
    genre_buckets: dict[str, list] = {}
    for app_id, game in games:
        raw_genres = game.get("genres", []) or []
        if raw_genres and isinstance(raw_genres[0], dict):
            primary = raw_genres[0].get("description", "Other")
        elif raw_genres and isinstance(raw_genres[0], str):
            primary = raw_genres[0]
        else:
            primary = "Other"
        genre_buckets.setdefault(primary, []).append((app_id, game))

    # Balanced sampling: at most 10 per genre, fill rest randomly
    import random
    random.shuffle(games)
    sampled = []
    seen_ids = set()

    # Do one pass per genre to ensure diversity
    for genre, bucket in sorted(genre_buckets.items(), key=lambda x: -len(x[1])):
        random.shuffle(bucket)
        per_genre = max(1, target // len(genre_buckets))
        count = 0
        for item in bucket:
            if len(sampled) >= target:
                break
            if item[0] not in seen_ids:
                sampled.append(item)
                seen_ids.add(item[0])
                count += 1
            if count >= per_genre:
                break
        if len(sampled) >= target:
            break

    # Fill remaining with random games
    if len(sampled) < target:
        remaining = [g for g in games if g[0] not in seen_ids]
        random.shuffle(remaining)
        for item in remaining:
            if len(sampled) >= target:
                break
            sampled.append(item)
            seen_ids.add(item[0])

    print(f"Sampled {len(sampled)} diverse games across {len(genre_buckets)} genres.")
    return sampled


def generate_query_with_deepseek(client: DeepSeekClient, app_id: str, game: dict) -> dict | None:
    """
    Ask DeepSeek to reverse-generate a realistic search query for this game.

    Returns: {"query": str, "ground_truth_ids": [app_id], "source_game_name": str}
    """
    name = game.get("name", "Unknown")
    desc = game.get("short_description", "")[:300]
    genres_raw = game.get("genres", [])
    if genres_raw and isinstance(genres_raw[0], dict):
        genres_str = ", ".join(g["description"] for g in genres_raw[:5])
    else:
        genres_str = ", ".join(genres_raw[:5])
    tags_raw = game.get("tags", {})
    if isinstance(tags_raw, dict):
        tags_str = ", ".join(list(tags_raw.keys())[:8])
    else:
        tags_str = ", ".join(tags_raw[:8])

    prompt = (
        f"Look at this Steam game:\n"
        f"Name: {name}\n"
        f"Short Description: {desc}\n"
        f"Genres: {genres_str}\n"
        f"Tags: {tags_str}\n\n"
        f"Write a short, natural language search query (1-2 sentences) "
        f"that a REAL Steam user might type to find this game.\n\n"
        f"Guidelines:\n"
        f"- Be creative and varied — don't just paraphrase the description\n"
        f"- Write what the USER wants to play, not a list of features\n"
        f"- Sound natural, like an actual person browsing Steam\n"
        f"- Don't include the game name in the query\n"
        f"- Make each query unique and realistic\n\n"
        f"Return ONLY a valid JSON object (no other text):\n"
        f'{{"query": "...", "quality": 5}}'
    )

    try:
        response = client.chat([
            {"role": "system", "content": "You are helping create a game recommendation test set. Respond only with valid JSON."},
            {"role": "user", "content": prompt}
        ])

        # Try parsing response as JSON object directly (most common case)
        import re
        obj_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if obj_match:
            try:
                import json as j
                obj = j.loads(obj_match.group())
                query = obj.get("query", "").strip()
                quality = int(obj.get("quality", 0))
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

        if not query or len(query) < 10:
            print(f"  [SKIP] Query too short for {name}")
            return None

        return {
            "query": query,
            "ground_truth_ids": [app_id],
            "source_game_name": name,
            "quality_score": quality,
        }

    except Exception as e:
        print(f"  [ERROR] Failed for {name}: {e}")
        return None


def main():
    print("=" * 70)
    print("DEEPSEEK QUERY GENERATOR — Step 1")
    print("=" * 70)

    # Initialize DeepSeek client
    print("\nInitializing DeepSeek client...")
    try:
        client = DeepSeekClient()
        print(f"  Model: {client.model}")
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nTo fix this:")
        print("  1. Open the .env file")
        print("  2. Set DEEPSEEK_API_KEY=your_actual_api_key")
        print("  3. Re-run this script")
        return

    # Load and sample games
    print("\nStep 1: Loading and sampling games...")
    games = load_sample_games(MAX_GAMES, TARGET_COUNT)

    # Generate queries
    print(f"\nStep 2: Generating queries for {len(games)} games using DeepSeek...\n")

    results = []
    for i, (app_id, game) in enumerate(games, 1):
        name = game.get("name", "?")
        print(f"[{i:3d}/{len(games):3d}] {name[:50]:50s}...", end=" ")
        sys.stdout.flush()

        result = generate_query_with_deepseek(client, app_id, game)
        if result:
            results.append(result)
            print(f"✓ (q={result['quality_score']})")
        else:
            print("✗ SKIP")

        # Rate limiting: respect DeepSeek API limits
        time.sleep(1)

    # Filter to keep only quality >= 3
    filtered = [r for r in results if r["quality_score"] >= 3]
    print(f"\n{'='*70}")
    print(f"Generation complete: {len(results)} generated, {len(filtered)} after quality filter (>= 3).")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(filtered)} queries to {OUTPUT_PATH}")

    # Show samples
    print("\n--- Sample queries ---")
    for item in filtered[:5]:
        print(f"  [Q={item['quality_score']}] \"{item['query'][:80]}...\"")
        print(f"    -> {item['source_game_name']}")
        print()

    print("=" * 70)
    print("NEXT: Run deepseek_generate_multi_gt.py for multi-GT expansion")
    print("=" * 70)


if __name__ == "__main__":
    main()
