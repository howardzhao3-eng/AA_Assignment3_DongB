"""
STEAM GAME RECOMMENDER SYSTEM - RAG PIPELINE
---------------------------------------------
RAG Pipeline:
  1. retrieve_candidates()  -> Cosine similarity vector search (top-K)
  2. rank_candidates()      -> LLM re-ranking (S2) or raw scores (S1)
  3. generate_answer()      -> LLM generates natural language recommendation

Config-driven: uses config.py for model/strategy selection.
Supports experiment IDs: E1_L1_D2_S2, E2_L1_D2_S2, E1_L2_D2_S2, etc.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from raglooker.steam_sqlite import load_games_with_reviews

# Configuration and Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))


@dataclass
class GameRecord:
    """Represents a single game record enriched with metadata and reviews."""
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        """Converts the record to a dictionary format for frontend display."""
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        """Ensures tags are returned as a list of strings (max 8)."""
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


def _fix_and_parse_json(text: str) -> list[dict]:
    """
    Robust JSON extraction from LLM output.
    Handles:
      - Markdown code fences (```json ... ```)
      - Extra text before/after JSON
      - Single quotes instead of double quotes
      - Trailing commas before ] or }
      - Missing quotes on keys
      - Invalid escape sequences like \\' or \\"
    """
    import re
    
    # Parse a candidate string with multiple fallback strategies
    def _try_parse(s: str) -> list[dict] | None:
        """Try to parse s as JSON, return list of dicts or None."""
        try:
            result = json.loads(s)
            if isinstance(result, list):
                return result
            if isinstance(result, dict):
                return [result]
            return None
        except json.JSONDecodeError:
            return None
    
    # Common JSON fixes
    def _apply_fixes(s: str) -> str:
        """Apply common LLM JSON formatting fixes."""
        # Strip markdown code fences
        s = re.sub(r'```(?:json)?\s*', '', s.strip())
        # Remove trailing commas before ] or }
        s = re.sub(r',\s*([\]}])', r'\1', s)
        # Fix unquoted keys (e.g., {app_id: "123"} -> {"app_id": "123"})
        # Use capture group approach to avoid variable-width lookbehind
        s = re.sub(r'([{,])\s*(\w+)(\s*:)', r'\1"\2"\3', s)
        # Remove invalid escape sequences like \' (keep only valid JSON escapes)
        s = re.sub(r'\\([^"\\/bfnrtu])', r'\1', s)
        # Fix double-escaped quotes
        s = s.replace('\\"', '"')
        return s
    
    # Strategy 1: Try parsing the whole text (fixed)
    fixed = _apply_fixes(text)
    result = _try_parse(fixed)
    if result:
        return result
    
    # Strategy 2: Extract JSON array [...] 
    start = fixed.find('[')
    end = fixed.rfind(']') + 1
    if start >= 0 and end > start:
        candidate = fixed[start:end]
        result = _try_parse(candidate)
        if result:
            return result
    
    # Strategy 3: Extract single JSON object {...}
    start = fixed.find('{')
    end = fixed.rfind('}') + 1
    if start >= 0 and end > start:
        candidate = fixed[start:end]
        result = _try_parse(candidate)
        if result:
            return result
    
    # Strategy 4: Fix invalid escape chars specifically
    # Some LLMs output \" for every quote inside strings
    escaped = fixed.replace('\\"', '"').replace("\\'", "'").replace('\\ ', ' ')
    result = _try_parse(escaped)
    if result:
        return result
    
    # Strategy 5: Regex extract individual objects
    for candidate_text in [fixed, escaped]:
        objects = re.findall(r'\{[^}]+\}', candidate_text)
        if objects:
            results = []
            for obj in objects:
                try:
                    obj_fixed = re.sub(r',\s*\}', '}', obj)
                    obj_fixed = re.sub(r'(\w+)\s*:', r'"\1":', obj_fixed)
                    obj_fixed = re.sub(r"'", '"', obj_fixed)
                    obj_fixed = re.sub(r'\\([^"\\/bfnrtu])', r'\1', obj_fixed)
                    results.append(json.loads(obj_fixed))
                except json.JSONDecodeError:
                    continue
            if results:
                return results
    
    return []


class GameSearchEngine:
    """
    Configurable RAG-based Game Recommender.

    Supports embedding models: nomic-embed-text (Ollama), all-MiniLM-L6-v2 (SentenceTransformers).
    Supports LLM models: gemma2:2b, phi3.5 (via Ollama).
    Supports document strategies: D1 (metadata only), D2 (metadata + top reviews).
    Supports retrieval strategies: S1 (pure vector), S2 (vector + LLM re-ranking).
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize engine: load config, load data, build docs, embed with caching."""
        if config is None:
            from raglooker.config import get_experiment_config
            config = get_experiment_config("E1_L1_D2_S2")
        self.config = config

        # Step 1: Load games with reviews (top-reviewed by default)
        max_games = int(os.environ.get("RAGLOOKER_MAX_GAMES", 5000))
        print(f"Loading up to {max_games} games with reviews...")
        self.games_data = load_games_with_reviews(DB_PATH, max_games, 3)
        print(f"Loaded {len(self.games_data)} games.")

        # Step 2: Build documents per game
        self.documents = []
        for app_id, game, reviews in self.games_data:
            doc = self.build_game_document(game, reviews)
            self.documents.append(doc)
        print(f"Built {len(self.documents)} game documents (strategy: {config['doc_strategy']['id']}).")

        # Step 3: Initialize embedding model
        embed_config = config['embed_model']
        if embed_config['source'] == 'sentence-transformers':
            self.encoder = SentenceTransformer(embed_config['name'])
        else:
            # Ollama embedding model — no local encoder needed
            self.encoder = None

        # Step 4: Embed with caching
        cache_dir = BASE_DIR / "data" / "embeddings_cache"
        os.makedirs(cache_dir, exist_ok=True)
        embed_name = embed_config['name'].replace(":", "_")
        doc_strategy = config['doc_strategy']['id']
        cache_path = cache_dir / f"{embed_name}_{doc_strategy}_{max_games}.npy"

        if cache_path.exists():
            print(f"Loading embeddings from cache: {cache_path.name}")
            self.game_vectors = np.load(cache_path)
        else:
            print(f"Computing embeddings... this may take a minute.")
            self.game_vectors = self._embed_all(self.documents)
            np.save(cache_path, self.game_vectors)
            print(f"Saved embeddings to cache: {cache_path.name} ({len(self.game_vectors)} vectors)")

        # Step 5: Build records — inject formatted reviews into raw for LLM prompt use
        self.records = []
        for app_id, game, reviews in self.games_data:
            # Format top reviews with vote counts (like Howard's approach)
            review_lines = []
            for r in reviews[:3]:
                review_text = r.get('review', '')[:200]
                votes = r.get('votes_up', 0)
                if review_text:
                    review_lines.append(f"{review_text} (Votes: {votes})")
            game['top_reviews'] = " | ".join(review_lines)
            self.records.append(GameRecord(app_id=app_id, raw=game))
        print(f"Ready: {len(self.records)} records indexed.")

    def build_game_document(self, game: dict, reviews: list[dict]) -> str:
        """
        Build a text document for a game based on the configured document strategy.
        
        Name is repeated 3x at the front to boost keyword matching in embedding space.
        (Fixes: 40% of misses were due to game name not being prominent enough)

        D1 (metadata only): name + description + genres + tags.
        D2 (metadata + positive reviews): same as D1 + top-voted positive reviews.
        """
        name = game.get("name", "")
        strategy = self.config['doc_strategy']
        # Repeat name 3x so the embedding gives more weight to the game's title
        parts = [
            f"{name} {name} {name}",
            game.get("short_description", ""),
        ]
        genres = game.get("genres", [])
        if isinstance(genres, list):
            parts.append(f"Genres: {', '.join(g['description'] if isinstance(g, dict) else g for g in genres[:5])}")
        tags = game.get("tags", {})
        if isinstance(tags, dict):
            parts.append(f"Tags: {', '.join(list(tags.keys())[:10])}")
        elif isinstance(tags, list):
            parts.append(f"Tags: {', '.join(t[:10] if isinstance(t, str) else str(t) for t in tags[:10])}")

        if strategy.get('use_reviews', False):
            # Sort reviews by votes_up descending to get the most helpful ones
            sorted_reviews = sorted(
                [r for r in reviews if r.get('voted_up')],
                key=lambda r: r.get('votes_up', 0), reverse=True
            )[:strategy.get('pos_count', 3)]
            for r in sorted_reviews:
                review_text = r.get('review', '')[:200]
                parts.append(f"Positive review: {review_text}")

        return ". ".join(parts)

    def _embed_all(self, documents: list[str]) -> np.ndarray:
        """Embed all game documents into vectors using the configured model."""
        embed_config = self.config['embed_model']
        if embed_config['source'] == 'sentence-transformers':
            return self.encoder.encode(documents, show_progress_bar=True)

        # Ollama embedding: sequential API calls
        vectors = []
        total = len(documents)
        for i, doc in enumerate(documents):
            resp = ollama.embeddings(model=embed_config['name'], prompt=doc)
            vectors.append(resp['embedding'])
            if (i + 1) % 500 == 0:
                print(f"Embedding progress: {i + 1}/{total}")
        return np.array(vectors, dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        embed_config = self.config['embed_model']
        if embed_config['source'] == 'sentence-transformers':
            return self.encoder.encode([query]).flatten()

        resp = ollama.embeddings(model=embed_config['name'], prompt=query)
        return np.array(resp['embedding'], dtype=np.float32)

    def retrieve_candidates(self, query: str) -> tuple[list[GameRecord], np.ndarray]:
        """
        Hybrid retrieval: vector search + keyword name matching.

        1) Cosine similarity on embeddings → top-N candidates.
        2) Extract meaningful keywords from the query, find exact name matches.
        3) Merge results: exact matches are promoted to the front.
        """
        if not self.records:
            return [], np.array([])

        candidate_count = self.config['retrieval_strategy']['candidate_count']
        query_vec = self._embed_query(query)

        # —— 1. Vector search ——
        similarities = np.dot(self.game_vectors, query_vec) / (
            np.linalg.norm(self.game_vectors, axis=1) * np.linalg.norm(query_vec)
        )
        top_k_indices = np.argsort(similarities)[-candidate_count:][::-1]
        vector_candidates = [self.records[i] for i in top_k_indices]
        vector_scores = similarities[top_k_indices]

        # —— 2. Keyword boost ——
        # Extract meaningful tokens from query (≥3 chars, excluding common words)
        import re
        stop_words = {'the', 'and', 'for', 'with', 'like', 'that', 'this',
                      'from', 'have', 'you', 'not', 'are', 'but', 'all',
                      'game', 'games', 'looking', 'want', 'find', 'some'}
        tokens = [
            t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)
            if len(t) >= 3 and t.lower() not in stop_words
        ]

        # —— 2. Keyword boost ——
        # For each keyword token, find games whose name contains it.
        # A match is valid even if it covers only a small part of a long name.
        keyword_matches = {}
        for record in self.records:
            name_lower = record.name.lower()
            matched_tokens = [t for t in tokens if t in name_lower]
            if matched_tokens:
                # Score: count of unique tokens matched (normalized)
                score = len(set(matched_tokens)) / max(len(tokens), 1)
                keyword_matches[record.app_id] = (record, score)

        # Keep the 3 best keyword matches
        exact_matches = sorted(keyword_matches.values(), key=lambda x: x[1], reverse=True)[:3]
        exact_ids = {record.app_id for record, _ in exact_matches}

        # —— 3. Merge: exact matches first, then vector candidates (dedup) ——
        merged = []
        seen_ids = set()

        # First: exact name matches (boosted)
        for record, _ in exact_matches:
            if record.app_id not in seen_ids:
                merged.append(record)
                seen_ids.add(record.app_id)

        # Then: vector candidates (skip if already in merged)
        for record in vector_candidates:
            if record.app_id not in seen_ids:
                merged.append(record)
                seen_ids.add(record.app_id)

        # Build combined score array
        merged_scores = np.array([
            1.0 if r.app_id in exact_ids else 0.5
            for r in merged
        ])

        # Trim to candidate_count
        merged = merged[:candidate_count]
        merged_scores = merged_scores[:candidate_count]

        if exact_matches:
            print(f"  [Hybrid] Keyword matched {len(exact_matches)} games from query tokens: {tokens}")

        return merged, merged_scores

    def rank_candidates(self, query: str,
                        candidates: list[GameRecord]) -> list[tuple[GameRecord, float]]:
        """
        Rank candidates.

        S2 (default): Use LLM to re-rank top candidates.
        S1 (ablation): Use similarity scores directly, skip LLM.
        """
        use_llm = self.config['retrieval_strategy']['use_llm_rank']

        if not use_llm:
            # S1: Pure vector — just return top 5 with decayed scores
            return [(cand, 1.0 - i / len(candidates)) for i, cand in enumerate(candidates[:5])]

        if not candidates:
            return []

        # S2: LLM re-ranking
        llm_model = self.config['llm_model']['name']

        ######################################################################
        # NEW: Use line numbers in the candidate list to avoid app_id
        # formatting mismatches. The LLM returns the index (1-20) as "id",
        # which we then use to look up the actual game record.
        ######################################################################
        lines = []
        for i, record in enumerate(candidates[:20], 1):
            desc = record.short_description[:100] if record.short_description else "No description"
            lines.append(f'{i}. {record.name} — {desc}')

        prompt = (
            f"You are a Steam game expert. A user is looking for: \"{query}\"\n"
            f"Here are 20 candidate Steam games:\n"
            + "\n".join(lines) +
            "\n\nSelect the 5 most relevant games."
            "\nReturn ONLY a valid JSON array (no other text):"
            '\n[{"id": 1, "score": 0.95, "reason": "..."}]'
            "\n\nIMPORTANT: 'id' is the line number (1-20) from the list above, NOT the app_id."
        )

        try:
            response = ollama.chat(model=llm_model, messages=[
                {"role": "system", "content": "You are a Steam game expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ])
            content = response['message']['content']

            # Use robust JSON parser
            results = _fix_and_parse_json(content)

            ranked = []
            for r in results:
                idx = r.get('id') or r.get('index') or r.get('app_id')
                if isinstance(idx, int) and 1 <= idx <= len(candidates):
                    ranked.append((candidates[idx - 1], float(r.get('score', 0.5))))
                elif isinstance(idx, str) and idx.isdigit():
                    idx_num = int(idx)
                    if 1 <= idx_num <= len(candidates):
                        ranked.append((candidates[idx_num - 1], float(r.get('score', 0.5))))
                elif isinstance(idx, str):
                    # fallback: match by app_id
                    for record in candidates:
                        if record.app_id == idx:
                            ranked.append((record, float(r.get('score', 0.5))))
                            break

            # Fallback: if LLM returned empty/unparseable results, use vector scores
            if not ranked:
                print("  [LLM fallback] Using vector scores (LLM returned empty)")
                return [(cand, 1.0 - i / len(candidates)) for i, cand in enumerate(candidates[:5])]

            return ranked[:5]

        except Exception as e:
            print(f"LLM ranking error: {e}")
            return [(cand, 1.0 - i / len(candidates)) for i, cand in enumerate(candidates[:5])]

    def generate_answer(self, query: str,
                        matches: list[tuple[GameRecord, float]]) -> str:
        """
        Generate a natural language recommendation using LLM.

        Only runs when retrieval strategy uses LLM (S2).
        Returns empty string for S1 (pure vector).
        """
        if not self.config['retrieval_strategy']['use_llm_rank']:
            return ""

        if not matches:
            return "No matching games were found for your request."

        llm_model = self.config['llm_model']['name']

        # Build context from top 5 matches — top_reviews was injected in __init__
        context_list = []
        for record, score in matches[:5]:
            reviews = record.raw.get('top_reviews', 'No player reviews available.')
            context_list.append(
                f"GAME: {record.name}\n"
                f"MATCH SCORE: {score:.4f}\n"
                f"DESCRIPTION: {record.short_description}\n"
                f"REVIEWS: {reviews}"
            )
        context_text = "\n---\n".join(context_list)

        prompt = (
            f"You are a professional Steam game curator.\n"
            f"User request: '{query}'\n\n"
            f"Candidate Games Dataset:\n{context_text}\n\n"
            f"Instructions:\n"
            f"1. Select the top 3 best matching games.\n"
            f"2. For each game, explain why it matches the user's request.\n"
            f"3. Include specific features, genres, or mechanics.\n"
            f"4. Be conversational but informative.\n\n"
            f"Respond in paragraphs."
        )

        try:
            response = ollama.chat(model=llm_model, messages=[
                {"role": "system", "content": "You are a helpful Steam game curator."},
                {"role": "user", "content": prompt}
            ])
            return response['message']['content']
        except Exception as e:
            print(f"LLM generation error: {e}")
            return f"The recommendation engine encountered an error: {str(e)}"

    def search(self, query: str) -> dict[str, Any]:
        """
        Main entry point called by Flask.

        Returns JSON with matches, answer, and timing metadata.
        """
        times = {}

        t0 = time.time()
        candidates, _ = self.retrieve_candidates(query)
        times['retrieval_ms'] = round((time.time() - t0) * 1000, 1)

        t0 = time.time()
        ranked = self.rank_candidates(query, candidates)
        times['ranking_ms'] = round((time.time() - t0) * 1000, 1)

        results = [record.to_result(score) for record, score in ranked[:5]]

        t0 = time.time()
        answer = self.generate_answer(query, ranked[:5])
        times['generation_ms'] = round((time.time() - t0) * 1000, 1)

        times['total_ms'] = times['retrieval_ms'] + times['ranking_ms'] + times['generation_ms']

        return {
            "matches": results,
            "answer": answer,
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "vector+llm" if self.config['retrieval_strategy']['use_llm_rank'] else "vector",
                "embed_model": self.config['embed_model']['name'],
                "llm_model": self.config['llm_model']['name'],
                "doc_strategy": self.config['doc_strategy']['id'],
                "timing_ms": times,
            },
        }


def create_search_engine(config: dict | None = None) -> GameSearchEngine:
    """
    Factory function for backward compatibility with app.py.

    If config is None, uses default (E1_L1_D2_S2).
    Otherwise creates engine with specified config.
    """
    if config is None:
        from raglooker.config import get_experiment_config
        config = get_experiment_config("E1_L1_D2_S2")
    return GameSearchEngine(config=config)
