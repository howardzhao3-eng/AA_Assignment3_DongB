"""
Evaluation module: run the system on a test set and compute metrics.

Metric hierarchy (for multi-GT test sets):
  PRIMARY:   NDCG@k  — normalized for varying GT counts, fair across queries
  AUXILIARY: Precision@k, Recall@k, MRR

NDCG is the most appropriate metric when ground_truth_ids vary in size (1–92):
  - Recall penalises queries with many GT games (max recall@5 = 5/92 ≈ 0.05)
  - Precision is fair but ignores ranking order
  - NDCG normalises by IDCG, so GT=1 and GT=92 queries contribute equally

Usage:
  engine = create_search_engine(config)
  queries = load_test_queries("synthetic_queries.json")
  metrics = evaluate(engine, queries, ks=[5, 10, 20])
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np


# ── Genre lookup (optional, for ablation analysis) ─────────────────────────

def load_genre_lookup(db_path: str | Path) -> dict[str, set[str]]:
    """
    Load genre information from the database.
    Returns a dict: app_id -> {genre_name_1, genre_name_2, ...}
    """
    conn = sqlite3.connect(str(db_path))
    lookup = {}
    cursor = conn.execute(
        "SELECT appid, genres_json FROM games WHERE name IS NOT NULL"
    )
    for row in cursor:
        app_id = str(row[0])
        genres = set()
        if row[1]:
            try:
                parsed = json.loads(row[1])
                for g in parsed:
                    if isinstance(g, dict) and "description" in g:
                        genres.add(g["description"])
            except (json.JSONDecodeError, TypeError):
                pass
        lookup[app_id] = genres
    conn.close()
    return lookup


# ── Binary relevance metrics ──────────────────────────────────────────────

def precision_at_k(matches: list[dict], ground_truth: list[str], k: int = 5) -> float:
    """Fraction of top-K results that are relevant. [0, 1]"""
    matched = sum(1 for m in matches[:k] if m['app_id'] in ground_truth)
    return matched / k


def recall_at_k(matches: list[dict], ground_truth: list[str], k: int = 5) -> float:
    """Fraction of ground-truth games found in top-K. [0, 1]"""
    matched = sum(1 for m in matches[:k] if m['app_id'] in ground_truth)
    return matched / len(ground_truth) if ground_truth else 0.0


def mrr(matches: list[dict], ground_truth: list[str]) -> float:
    """Mean Reciprocal Rank: how early does the first correct answer appear?"""
    for i, m in enumerate(matches):
        if m['app_id'] in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


# ── NDCG (primary metric for multi-GT evaluation) ─────────────────────────

def ndcg_at_k(matches: list[dict], ground_truth: list[str], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain.

    NDCG is the PRIMARY metric because it naturally normalises for
    varying ground-truth sizes via IDCG:
      - GT=1  → IDCG@5 = 1.0 (only one perfect match possible)
      - GT=15 → IDCG@5 = 2.95 (can get full credit for all 5 slots)

    This means every query contributes equally to the final average,
    regardless of how many relevant games exist.
    """
    # DCG: actual ranking
    dcg = 0.0
    for i, m in enumerate(matches[:k]):
        if m['app_id'] in ground_truth:
            dcg += 1.0 / np.log2(i + 2)

    # IDCG: perfect ranking (all relevant games at top, capped at k)
    ideal_relevant = min(k, len(ground_truth))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_ks(matches: list[dict], ground_truth: list[str],
               ks: list[int] = [5, 10, 20]) -> dict:
    """Compute NDCG at multiple cut-offs in one pass."""
    # Precompute relevance for each position
    max_k = max(ks)
    gains = np.zeros(max_k)
    for i, m in enumerate(matches[:max_k]):
        if m['app_id'] in ground_truth:
            gains[i] = 1.0

    # DCG at each position
    positions = np.arange(1, max_k + 1)
    discounts = 1.0 / np.log2(positions + 1)
    dcg_cum = np.cumsum(gains * discounts)

    # IDCG at each position
    ideal_gains = np.zeros(max_k)
    ideal_gains[:min(max_k, len(ground_truth))] = 1.0
    idcg_cum = np.cumsum(ideal_gains * discounts)

    return {
        f"ndcg@{k}": float(dcg_cum[k-1] / idcg_cum[k-1]) if idcg_cum[k-1] > 0 else 0.0
        for k in ks
    }


# ── Main evaluation entry point ──────────────────────────────────────────

def evaluate(engine, test_queries: list[dict],
             ks: list[int] | None = None) -> dict:
    """
    Run engine.search() on every test query and aggregate metrics.

    Args:
        engine: An object with .search(query) returning {matches: [...], meta: ...}
        test_queries: List of {"query": str, "ground_truth_ids": [str, ...]}
        ks: List of k values for NDCG (default [5, 10, 20])

    Returns:
        dict with averaged metrics across all queries:
          Primary:  ndcg@5, ndcg@10, ndcg@20
          Auxiliary: precision@5, recall@5, mrr
          Timing:   retrieval_ms, ranking_ms, generation_ms, total_ms
    """
    if ks is None:
        ks = [5, 10, 20]

    # Primary metrics
    metrics = {
        "precision@5": [], "recall@5": [], "mrr": [],
        "retrieval_ms": [], "ranking_ms": [], "generation_ms": [], "total_ms": [],
    }
    for k in ks:
        metrics[f"ndcg@{k}"] = []

    total = len(test_queries)
    for idx, item in enumerate(test_queries, 1):
        query = item["query"][:60]
        gt = item["ground_truth_ids"]

        print(f"  Query [{idx}/{total}]: \"{query}...\"", end=" ")
        sys.stdout.flush()

        result = engine.search(item["query"])
        matches = result.get("matches", [])
        timing = result.get("meta", {}).get("timing_ms", {})

        # NDCG at multiple ks
        ndcg_results = ndcg_at_ks(matches, gt, ks)
        for k_name, k_val in zip([f"ndcg@{k}" for k in ks], ndcg_results.values()):
            metrics[k_name].append(k_val)

        # Auxiliary metrics at k=5
        metrics["precision@5"].append(precision_at_k(matches, gt, 5))
        metrics["recall@5"].append(recall_at_k(matches, gt, 5))
        metrics["mrr"].append(mrr(matches, gt))

        # Timing
        metrics["retrieval_ms"].append(timing.get("retrieval_ms", 0))
        metrics["ranking_ms"].append(timing.get("ranking_ms", 0))
        metrics["generation_ms"].append(timing.get("generation_ms", 0))
        metrics["total_ms"].append(timing.get("total_ms", 0))

        # Print status
        ndcg5 = ndcg_results.get("ndcg@5", 0)
        ndcg_icon = "✓" if ndcg5 > 0.5 else ("▸" if ndcg5 > 0 else "✗")
        p5 = matches[:5]
        matched = sum(1 for m in p5 if m['app_id'] in gt)
        print(f"{ndcg_icon} NDCG@5={ndcg5:.3f} P@5={matched}/5")

    return {k: round(np.mean(v), 4) if isinstance(v, list) else v
            for k, v in metrics.items()}
