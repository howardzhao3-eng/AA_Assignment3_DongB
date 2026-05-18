"""
Demo: Formatted Game Recommendation Output
===========================================
Standalone script — does NOT modify any existing code.
Uses a custom prompt to generate recommendations in a senior game editor style.

Usage:
  python experiments/demo_formatted_output.py
  python experiments/demo_formatted_output.py --query "I want a cozy card game"
  python experiments/demo_formatted_output.py --exp E2_L1_D2_S2 --query "survival game for co-op"
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path so we can import raglooker
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from raglooker.recommender_factory import create_search_engine
from raglooker.config import get_experiment_config, FINAL_CONFIG


def build_context(matches: list[dict]) -> str:
    """Build LLM context from search() matches."""
    context_list = []
    for i, match in enumerate(matches[:5], 1):
        name = match.get("name", "Unknown")
        desc = match.get("short_description", "No description")
        genres = ", ".join(match.get("genres", [])[:3]) or "Unknown genre"
        tags_list = match.get("tags", [])[:6]
        tags_str = ", ".join(tags_list) if tags_list else "No tags"
        price = match.get("price", "Unknown")
        release = match.get("release_date", "Unknown")
        platforms = match.get("platforms", {})
        platform_str = "/".join(
            p for p, enabled in platforms.items() if enabled
        ) or "Unknown platform"

        # Build rating string from review counts, fall back to Metacritic
        positive = match.get("positive", 0)
        negative = match.get("negative", 0)
        total = positive + negative
        if total > 0:
            rating_str = f"{positive / total * 100:.0f}% positive (from {total:,} reviews)"
        elif metacritic_raw := match.get("metacritic"):
            rating_str = f"Metacritic {metacritic_raw}"
        else:
            rating_str = "User ratings N/A"

        context_list.append(
            f"[Game {i}]\n"
            f"Name: {name}\n"
            f"Description: {desc}\n"
            f"Genres: {genres}\n"
            f"Tags: {tags_str}\n"
            f"Price: {price}\n"
            f"Release Date: {release}\n"
            f"Platforms: {platform_str}\n"
            f"Rating: {rating_str}\n"
        )
    return "\n---\n".join(context_list)


def generate_curated_recommendation(
    query: str,
    matches: list[dict],
    llm_model: str = "gemma2:2b",
) -> str:
    """
    Generate a recommendation using a custom prompt (senior game editor style).
    Does NOT modify recommender.py's original logic.
    """
    if not matches:
        return "No matching games found."

    context_text = build_context(matches)

    prompt = (
        "You are a friend who has played these games for 100 hours. "
        "Talk like you're texting a buddy — punchy, genuine, specific. "
        "Use contractions and short sentences. Never use marketing fluff.\n\n"
        "Output exactly three game cards in the format below. "
        "No intro, no outro, no commentary — just the cards.\n\n"
        "FORMAT:\n"
        "🎮 Recommended Game: [Title] ([English name if different])\n"
        "• Genre: ...\n"
        "• Platform: ...\n"
        "• Release Year: ...\n"
        "• Rating: (copy the Rating field EXACTLY from the candidate data — do NOT write N/A if the data contains a rating)\n"
        "• Why You'll Love It: (MUST give exactly 2-3 bullet points)\n"
        "  1. Describe a concrete mechanic that matches one of the Notable tags. "
        "Say exactly what you can do in the game.\n"
        "  2. Quote or closely paraphrase the Player quote to ground it in a real player's words. "
        "If there's no quote, describe a vivid sensory detail (visual style, music, atmosphere).\n"
        "  3. (Optional third point) Add another hook — humor, challenge curve, or a standout moment.\n"
        "  Avoid empty words like 'fun gameplay' or 'amazing experience'.\n"
        "• Perfect For: 1–2 specific player types "
        "(e.g., 'fans of physics sandboxes who love short chaotic sessions').\n"
        "• If You Like: 2–3 games, each with a note in parentheses that names "
        "ONE shared mechanic or emotional tone. Use the formula "
        "'[Game] (both about [X])'. Be specific, avoid vague terms like 'exploration'.\n"
        "• One-Liner: An evocative sentence that captures the heart of the game "
        "in a personal, memorable way.\n\n"
        "EXAMPLE:\n"
        "🎮 Recommended Game: Dave the Diver (DAVE THE DIVER)\n"
        "• Genre: Adventure, Simulation, RPG\n"
        "• Platform: Windows, Mac\n"
        "• Release Year: 2023\n"
        "• Rating: Metacritic 90\n"
        "• Why You'll Love It:\n"
        "  1. You can dive into a procedurally generated ocean by day and run "
        "a sushi restaurant by night — the loop is addictive because every catch "
        "literally becomes a menu item.\n"
        "  2. One Steam reviewer says it's 'the perfect mix of exploration and "
        "management,' and they're right — there's always a new recipe or upgrade waiting.\n"
        "  3. The hand-drawn pixel art makes every fish feel handcrafted, and the "
        "jazzy soundtrack during dinner service is pure vibes.\n"
        "• Perfect For: Stardew Valley fans who want more underwater exploration, "
        "and anyone who likes switching between action and zen management.\n"
        "• If You Like: Stardew Valley (both about farming-life loop with exploration), "
        "Moonlighter (both about fight by day, sell by night), "
        "Subnautica (both about oceanic immersion and discovery).\n"
        "• One-Liner: When your sushi bar needs the freshest catch, you're the diver "
        "who brings up the sea's best-kept secrets.\n\n"
        "Now recommend games for: \"" + query + "\"\n"
        "Candidate data:\n" + context_text
    )

    try:
        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a formatting engine that only outputs game cards. "
                        "You have the voice of a trusted gamer friend. "
                        "Never write introductions, summaries, or markdown outside the cards. "
                        "Never invent ratings — copy them exactly from the data."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating recommendation: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Demo: formatted game recommendation output (senior editor style)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Game description query (if omitted, prompts interactively)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=FINAL_CONFIG,
        help=f"Experiment config ID (default: {FINAL_CONFIG})",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM model name (default: use the one from experiment config)",
    )
    args = parser.parse_args()

    # Get query
    query = args.query
    if not query:
        query = input("Enter a game description:\n> ").strip()
    if not query:
        print("Query cannot be empty.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"🔍 Query: \"{query}\"")
    print(f"⚙️  Config: {args.exp}")
    print(f"{'='*60}\n")

    # Create search engine
    print("Loading game data and building index...")
    config = get_experiment_config(args.exp)
    engine = create_search_engine(config)
    print("Index ready.\n")

    # Run search
    print("Searching for matching games...")
    result = engine.search(query)
    matches = result.get("matches", [])
    meta = result.get("meta", {})

    print(f"Found {len(matches)} candidate games")
    print(f"Retrieval mode: {meta.get('retrieval_mode', 'N/A')}")
    print(f"Embedding model: {meta.get('embed_model', 'N/A')}")
    print(f"LLM model: {meta.get('llm_model', 'N/A')}")
    timing = meta.get("timing_ms", {})
    print(f"Total time: {timing.get('total_ms', 'N/A')}ms\n")

    # Generate formatted recommendation
    llm_model = args.llm or config["llm_model"]["name"]
    print("Generating recommendation...")
    answer = generate_curated_recommendation(query, matches, llm_model)

    # Output
    print(f"\n{'='*60}")
    print("📝 Recommendation")
    print(f"{'='*60}\n")
    print(answer)
    print(f"\n{'='*60}")
    print("✨ Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
