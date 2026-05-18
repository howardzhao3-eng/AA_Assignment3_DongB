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

        context_list.append(
            f"[Game {i}]\n"
            f"Name: {name}\n"
            f"Description: {desc}\n"
            f"Genres: {genres}\n"
            f"Tags: {tags_str}\n"
            f"Price: {price}\n"
            f"Release Date: {release}\n"
            f"Platforms: {platform_str}\n"
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
        f"You are a senior game editor. Recommend the best-matching Steam games based on the user's request.\n\n"
        f"User request: '{query}'\n\n"
        f"Candidate games:\n{context_text}\n\n"
        f"Select the top 1~3 most relevant games and output in the following format:\n\n"
        f"🎮 Recommended Game: 《Game Name》\n"
        f"• Genre / Platform / Release Year / Rating\n"
        f"• Why we recommend it: 3~4 bullet points, each focusing on a different dimension "
        f"(gameplay, narrative, art, innovation, emotion, etc.), with specific details.\n"
        f"• Best for: Clearly state which types of players will enjoy this game.\n"
        f"• Similar games: 2~3 titles with brief explanations of similarities.\n"
        f"• One-liner: A short sentence that sparks curiosity or emotional resonance.\n\n"
        f"Formatting guidelines:\n"
        f"- Use emojis tastefully, don't overdo it\n"
        f"- Separate sections with blank lines\n"
        f"- Keep total length around 300 words\n"
        f"- Tone: professional, enthusiastic, engaging — not hype-ish\n\n"
        f"Note: If some info (release year, rating) is missing from the candidate data, "
        f"you may reasonably infer it or omit that field."
    )

    try:
        response = ollama.chat(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior game editor with deep knowledge of Steam games. "
                        "Your recommendation style is professional, warm, and engaging — never hype-ish. "
                        "You excel at using concise language and specific details to captivate readers."
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
