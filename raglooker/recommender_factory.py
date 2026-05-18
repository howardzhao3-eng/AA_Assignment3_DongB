"""
Factory function for creating search engine instances from config.

Selects embedding model source, document strategy, and retrieval strategy.
"""

from __future__ import annotations

from raglooker.recommender import GameSearchEngine
from raglooker.config import get_experiment_config, FINAL_CONFIG


def create_search_engine(config: dict | None = None) -> GameSearchEngine:
    """
    Create a configured GameSearchEngine instance.

    If config is None, use the final recommended configuration
    (E2_L1_D2_S2 = all-MiniLM + gemma2:2b + D2 + S2).
    """
    if config is None:
        config = get_experiment_config(FINAL_CONFIG)

    return GameSearchEngine(config=config)
