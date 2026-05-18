"""
Run a single experiment and cache the result to experiments/results/{exp_id}_{test_label}.json.

Also saves per-query details (full metadata, answer, metrics for EVERY query) to
experiments/results/{exp_id}_{test_label}_per_query.json for downstream analyses
(error analysis, case studies, ablation deep-dives, etc.).

Usage:
  python experiments/run_single.py E1_L1_D2_S2 --deepseek-multi
  python experiments/run_single.py E2_L1_D2_S2 --deepseek-multi --force   # re-run
  python experiments/run_single.py --run-all --deepseek-multi --force     # re-run all 5
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raglooker.config import get_experiment_config
from raglooker.recommender_factory import create_search_engine
from raglooker.evaluation import ndcg_at_ks, precision_at_k, recall_at_k, mrr


# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"

TEST_SETS = {
    "legacy": (
        BASE_DIR / "raglooker" / "data" / "synthetic_queries.json",
        "legacy",
    ),
    "deepseek": (
        BASE_DIR / "raglooker" / "data" / "synthetic_queries_deepseek.json",
        "deepseek",
    ),
    "deepseek-multi": (
        BASE_DIR / "raglooker" / "data" / "synthetic_queries_deepseek_multi_gt.json",
        "deepseek-multi",
    ),
}


def result_path(exp_id: str, test_label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{exp_id}_{test_label}.json"


def per_query_path(exp_id: str, test_label: str) -> Path:
    return RESULTS_DIR / f"{exp_id}_{test_label}_per_query.json"


def safe_game_detail(engine, match: dict) -> dict:
    """Enrich a match with full metadata from engine.records. JSON-safe."""
    app_id = match["app_id"]
    for record in engine.records:
        if record.app_id == app_id:
            genres = record.raw.get("genres", [])
            safe_genres = []
            for g in genres:
                if isinstance(g, dict):
                    safe_genres.append({"id": g.get("id", ""), "description": g.get("description", "")})
                else:
                    safe_genres.append(g)
            return {
                "app_id": app_id,
                "name": record.name,
                "score": match.get("score", 0),
                "short_description": (record.short_description or "")[:500],
                "genres": safe_genres,
                "tags": record._normalize_tags(record.raw.get("tags", [])),
                "price": record.raw.get("price"),
            }
    return {
        "app_id": app_id,
        "name": match.get("name", "?"),
        "score": match.get("score", 0),
        "short_description": "",
        "genres": [],
        "tags": [],
    }


def run_and_cache(exp_id: str, test_label: str, force: bool = False) -> dict:
    """Run experiment, save aggregated + per-query results. Skip if cached."""
    out_path = result_path(exp_id, test_label)
    pq_path = per_query_path(exp_id, test_label)

    if out_path.exists() and pq_path.exists() and not force:
        with open(out_path, encoding="utf-8") as f:
            cached = json.load(f)
        print(f"  [{exp_id}] Using cached result: {out_path.name}")
        return cached

    test_path, _ = TEST_SETS[test_label]
    with open(test_path, encoding="utf-8-sig") as f:
        test_queries = json.load(f)

    print(f"  [{exp_id}] Running on {test_label} ({len(test_queries)} queries)...")
    config = get_experiment_config(exp_id)
    engine = create_search_engine(config)

    ks = [5, 10, 20]
    buckets = {f"ndcg@{k}": [] for k in ks}
    buckets.update({"precision@5": [], "recall@5": [], "mrr": [],
                    "retrieval_ms": [], "ranking_ms": [], "generation_ms": [], "total_ms": []})
    per_query = []

    total = len(test_queries)
    for idx, item in enumerate(test_queries, 1):
        query_text = item["query"]
        gt = item["ground_truth_ids"]
        print(f"  Query [{idx:3d}/{total:3d}]: \"{query_text[:60]}...\"", end=" ")
        sys.stdout.flush()

        result = engine.search(query_text)
        matches = result.get("matches", [])
        answer = result.get("answer", "")
        timing = result.get("meta", {}).get("timing_ms", {})

        ndcg_r = ndcg_at_ks(matches, gt, ks)
        p5 = precision_at_k(matches, gt, 5)
        r5 = recall_at_k(matches, gt, 5)
        mr = mrr(matches, gt)

        for k in ks:
            buckets[f"ndcg@{k}"].append(ndcg_r.get(f"ndcg@{k}", 0))
        buckets["precision@5"].append(p5)
        buckets["recall@5"].append(r5)
        buckets["mrr"].append(mr)
        buckets["retrieval_ms"].append(timing.get("retrieval_ms", 0))
        buckets["ranking_ms"].append(timing.get("ranking_ms", 0))
        buckets["generation_ms"].append(timing.get("generation_ms", 0))
        buckets["total_ms"].append(timing.get("total_ms", 0))

        matched_gt_ids = [m["app_id"] for m in matches[:5] if m["app_id"] in gt]
        top5 = [safe_game_detail(engine, m) for m in matches[:5]]

        per_query.append({
            "query_index": idx - 1,
            "query": query_text,
            "gt_ids": gt,
            "gt_names": item.get("gt_names", []),
            "gt_count": len(gt),
            "matched_in_top5": matched_gt_ids,
            "ndcg@5": ndcg_r.get("ndcg@5", 0),
            "ndcg@10": ndcg_r.get("ndcg@10", 0),
            "ndcg@20": ndcg_r.get("ndcg@20", 0),
            "precision@5": p5,
            "recall@5": r5,
            "mrr": mr,
            "timing_ms": timing,
            "top5": top5,
            "top5_answer": answer if len(answer) > 10 else "",
        })

        ndcg5 = per_query[-1]["ndcg@5"]
        icon = "✓" if ndcg5 > 0.5 else ("▸" if ndcg5 > 0 else "✗")
        print(f"{icon} NDCG@5={ndcg5:.3f} P@5={p5:.3f} T={timing.get('total_ms', 0):.0f}ms")

    import numpy as np
    metrics = {k: round(np.mean(v), 4) for k, v in buckets.items()}
    metrics.update({
        "embed_model": config["embed_model"]["name"],
        "llm_model": config["llm_model"]["name"],
        "doc_strategy": config["doc_strategy"]["desc"],
        "retrieval_strategy": config["retrieval_strategy"]["desc"],
        "config_desc": f"{config['embed_model']['name']} + {config['llm_model']['name']} + {config['doc_strategy']['desc']} + {config['retrieval_strategy']['desc']}",
        "num_queries": total,
    })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(pq_path, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)
    print(f"  [{exp_id}] Saved aggregated → {out_path.name}, per-query → {pq_path.name}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("exp_id", nargs="?", help="Experiment ID, e.g. E1_L1_D2_S2")
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--deepseek-multi", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-all", action="store_true", help="Run all 5 experiments")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.deepseek_multi:
        test_label = "deepseek-multi"
    elif args.deepseek:
        test_label = "deepseek"
    else:
        test_label = "legacy"

    if args.run_all:
        from raglooker.config import ALL_EXPERIMENTS
        for eid in ALL_EXPERIMENTS:
            print(f"\n{'='*60}")
            run_and_cache(eid, test_label, force=args.force)
        print(f"\n{'='*60}\nAll experiments complete.")
        return

    if not args.exp_id:
        parser.error("the following arguments are required: exp_id (or use --run-all)")

    metrics = run_and_cache(args.exp_id, test_label, force=args.force)
    if not args.quiet:
        print()
        print(f"  -> NDCG@5:      {metrics.get('ndcg@5', '?')}")
        print(f"  -> NDCG@10:     {metrics.get('ndcg@10', '?')}")
        print(f"  -> NDCG@20:     {metrics.get('ndcg@20', '?')}")
        print(f"  -> Precision@5: {metrics.get('precision@5', '?')}")
        print(f"  -> Recall@5:    {metrics.get('recall@5', '?')}")
        print(f"  -> MRR:         {metrics.get('mrr', '?')}")
        print(f"  -> Total Time:  {metrics.get('total_ms', '?')}ms")


if __name__ == "__main__":
    main()
