"""
Central experiment configuration management.

Usage:
  from config import get_experiment_config
  cfg = get_experiment_config("E1_L1_D2_S2")
  engine = GameSearchEngine(config=cfg)
"""

import os

# Available embedding models
EMBEDDING_MODELS = {
    "E1": {"name": "nomic-embed-text", "dim": 768, "source": "ollama"},
    "E2": {"name": "all-MiniLM-L6-v2", "dim": 384, "source": "sentence-transformers"},
}

# Available LLM models
LLM_MODELS = {
    "L1": {"name": "gemma2:2b", "params": "2B", "source": "ollama"},
    "L2": {"name": "phi3.5", "params": "3.8B", "source": "ollama"},
}

# Document strategies
DOCUMENT_STRATEGIES = {
    "D1": {"id": "D1", "desc": "metadata only", "use_reviews": False, "pos_count": 0, "neg_count": 0},
    "D2": {"id": "D2", "desc": "metadata + top-3 positive reviews", "use_reviews": True, "pos_count": 3, "neg_count": 0},
}

# Retrieval strategies
RETRIEVAL_STRATEGIES = {
    "S1": {"id": "S1", "desc": "pure vector search, skip LLM", "candidate_count": 5, "use_llm_rank": False},
    "S2": {"id": "S2", "desc": "vector search + LLM re-ranking", "candidate_count": 20, "use_llm_rank": True},
}


def get_experiment_config(exp_id: str) -> dict:
    """
    Parse experiment ID and return config dict.

    Format: E{embed}_L{llm}_D{doc}_S{retrieval}
    Example: "E1_L1_D2_S2" -> nomic-embed-text + gemma2:2b + D2 + S2
    """
    parts = exp_id.split("_")
    e_id, l_id, d_id, s_id = parts

    return {
        "embed_model": EMBEDDING_MODELS[e_id],
        "llm_model": LLM_MODELS[l_id],
        "doc_strategy": DOCUMENT_STRATEGIES[d_id],
        "retrieval_strategy": RETRIEVAL_STRATEGIES[s_id],
        "max_games": int(os.environ.get("RAGLOOKER_MAX_GAMES", "5000")),
    }


# Final recommended configuration: all-MiniLM-L6-v2 + gemma2:2b + D2 + S2
FINAL_CONFIG = "E2_L1_D2_S2"

# All 5 experiments (no duplicates, no E3/L3/D3/S3)
ALL_EXPERIMENTS = [
    # Embedding comparison (fixed: L1 + D2 + S2)
    "E1_L1_D2_S2",  # baseline: nomic + gemma2:2b
    "E2_L1_D2_S2",  # all-MiniLM-L6-v2 ⭐ FINAL

    # LLM comparison (fixed: E1 + D2 + S2)
    "E1_L2_D2_S2",  # phi3.5

    # Document strategy comparison (fixed: E1 + L1 + S2)
    "E1_L1_D1_S2",  # metadata only

    # Retrieval strategy comparison (fixed: E1 + L1 + D2)
    "E1_L1_D2_S1",  # pure vector (no LLM) = ablation
]
