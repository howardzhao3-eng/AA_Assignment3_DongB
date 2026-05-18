from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def load_games_from_sqlite(db_path: Path, limit: int = 5000, select_mode: str = 'top_reviewed') -> list[tuple[str, dict[str, Any]]]:
    """
    Load games from SQLite with different selection strategies.

    select_mode options:
    - 'top_reviewed': Pick games with the most positive reviews (default, recommended).
    - 'sequential': Original behavior, ORDER BY appid LIMIT ?.
    - 'random': ORDER BY RANDOM() LIMIT ?.
    """
    if select_mode == 'top_reviewed':
        query = """
            SELECT g.* FROM games g
            WHERE g.name IS NOT NULL AND TRIM(g.name) != ''
            AND g.appid IN (
                SELECT r.appid FROM reviews r 
                WHERE r.voted_up = 1 
                GROUP BY r.appid 
                ORDER BY COUNT(*) DESC 
                LIMIT ?
            )
            ORDER BY appid
        """
    elif select_mode == 'random':
        query = """
            SELECT * FROM games
            WHERE name IS NOT NULL AND TRIM(name) != ''
            ORDER BY RANDOM() LIMIT ?
        """
    else:
        # sequential (fallback)
        query = """
            SELECT * FROM games
            WHERE name IS NOT NULL AND TRIM(name) != ''
            ORDER BY appid LIMIT ?
        """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (limit,)).fetchall()

    records: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        raw = {
            "name": row["name"] or "Unknown title",
            "short_description": row["short_description"] or "",
            "about_the_game": row["about_the_game"] or "",
            "detailed_description": row["detailed_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            "positive": row["positive"] if row["positive"] is not None else 0,
            "negative": row["negative"] if row["negative"] is not None else 0,
            "developers": _load_json_value(row["developers_json"], []),
            "publishers": _load_json_value(row["publishers_json"], []),
            "categories": _load_json_value(row["categories_json"], []),
            "genres": _load_json_value(row["genres_json"], []),
            "tags": _load_json_value(row["tags_json"], {}),
        }
        records.append((str(row["appid"]), raw))

    return records


def _load_json_value(payload: str | None, default: Any) -> Any:
    if not payload:
        return default

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return default


def load_reviews_for_games(db_path: Path, app_ids: list[str],
                           max_reviews_per_game: int = 5) -> dict[str, list[dict]]:
    """
    Fetch top-voted positive reviews for a list of game app_ids.

    Uses ORDER BY votes_up DESC to get the most helpful reviews,
    rather than the most recent ones (timestamp_created).
    Batches queries in chunks of 500 to avoid SQLite IN clause limits.
    """
    result: dict[str, list[dict]] = {}
    chunk_size = 500

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for i in range(0, len(app_ids), chunk_size):
            chunk = app_ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            query = f"""
                SELECT appid, review, voted_up, votes_up, timestamp_created
                FROM reviews
                WHERE appid IN ({placeholders}) AND voted_up = 1
                ORDER BY votes_up DESC
            """
            rows = conn.execute(query, chunk).fetchall()
            for row in rows:
                aid = str(row["appid"])
                if aid not in result:
                    result[aid] = []
                if len(result[aid]) < max_reviews_per_game:
                    result[aid].append({
                        "review": row["review"],
                        "voted_up": bool(row["voted_up"]),
                        "votes_up": row["votes_up"],
                        "timestamp_created": row["timestamp_created"],
                    })

    return result


def load_games_with_reviews(db_path: Path, game_limit: int,
                            reviews_per_game: int = 3) -> list[tuple[str, dict, list[dict]]]:
    """
    High-level loader: gets games (top-reviewed) + their top reviews.

    Returns: [(app_id, game_dict, [review_dict, ...]), ...]
    This is the function that recommender.py will call at startup.
    """
    base_data = load_games_from_sqlite(db_path, game_limit, select_mode='top_reviewed')
    app_ids = [aid for aid, _ in base_data]
    reviews_map = load_reviews_for_games(db_path, app_ids, reviews_per_game)

    result = []
    for app_id, raw in base_data:
        reviews = reviews_map.get(app_id, [])
        result.append((app_id, raw, reviews))

    return result
