"""
Run all 5 experiments, skipping completed ones. Generate results_summary.md.

Usage:
  python experiments/run_experiments.py                         # Legacy test set
  python experiments/run_experiments.py --deepseek              # DeepSeek single-GT
  python experiments/run_experiments.py --deepseek-multi        # DeepSeek multi-GT (recommended!)
  python experiments/run_experiments.py --deepseek-multi --force  # Force re-run all

Results cached individually in experiments/results/{exp_id}_{test_label}.json
Summary saved to experiments/results_summary.md
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raglooker.config import ALL_EXPERIMENTS

# Reuse the caching logic from run_single.py
from experiments.run_single import run_and_cache, result_path, TEST_SETS


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def select_test_set(args) -> tuple[str, str]:
    """Resolve CLI args to (test_label, display_name)."""
    if args.deepseek_multi:
        return "deepseek-multi", "DeepSeek multi-GT"
    elif args.deepseek:
        return "deepseek", "DeepSeek single-GT"
    else:
        return "legacy", "Legacy (gemma2)"


def fmt(v, suffix=""):
    if isinstance(v, str):
        return str(v) + suffix
    return f"{v:.4f}" + suffix


def save_markdown(results: dict, test_label_display: str):
    """Format all results as markdown and write to results_summary.md."""
    lines = [
        f"# Experiment Results Summary\n",
        f"Test set: {test_label_display}\n",
        "",
        "## 1. Embedding Model Comparison (E1 vs E2)\n",
        "| Embedding Model | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Recall@5 | MRR | Total Time | Note |",
        "|----------------|:--------:|:-------:|:-------:|:-----------:|:--------:|:---:|:----------:|------|",
    ]

    for exp_id in ["E1_L1_D2_S2", "E2_L1_D2_S2"]:
        m = results.get(exp_id, {})
        note = "Baseline" if "E1" in exp_id else ""
        embed_name = m.get('embed_model', exp_id)
        lines.append(
            f"| {embed_name} | "
            f"{fmt(m.get('ndcg@5', '?'))} | "
            f"{fmt(m.get('ndcg@10', '?'))} | "
            f"{fmt(m.get('ndcg@20', '?'))} | "
            f"{fmt(m.get('precision@5', '?'))} | "
            f"{fmt(m.get('recall@5', '?'))} | "
            f"{fmt(m.get('mrr', '?'))} | "
            f"{fmt(m.get('total_ms', '?'), 'ms')} | {note} |"
        )

    lines += [
        "",
        "## 2. LLM Model Comparison (L1 vs L2)\n",
        "| LLM Model | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |",
        "|-----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|",
    ]

    for exp_id in ["E1_L1_D2_S2", "E1_L2_D2_S2"]:
        m = results.get(exp_id, {})
        note = "Baseline" if "L1" in exp_id else ""
        llm_name = m.get('llm_model', exp_id)
        lines.append(
            f"| {llm_name} | "
            f"{fmt(m.get('ndcg@5', '?'))} | "
            f"{fmt(m.get('ndcg@10', '?'))} | "
            f"{fmt(m.get('ndcg@20', '?'))} | "
            f"{fmt(m.get('precision@5', '?'))} | "
            f"{fmt(m.get('total_ms', '?'), 'ms')} | {note} |"
        )

    lines += [
        "",
        "## 3. Document Strategy Comparison (D1 vs D2)\n",
        "| Strategy | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |",
        "|----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|",
    ]

    for exp_id in ["E1_L1_D1_S2", "E1_L1_D2_S2"]:
        m = results.get(exp_id, {})
        note = "Baseline" if "D2" in exp_id else "Ablation (no reviews)"
        doc_desc = m.get('doc_strategy', exp_id)
        if isinstance(doc_desc, dict):
            doc_desc = doc_desc.get('desc', str(doc_desc))
        lines.append(
            f"| {doc_desc} | "
            f"{fmt(m.get('ndcg@5', '?'))} | "
            f"{fmt(m.get('ndcg@10', '?'))} | "
            f"{fmt(m.get('ndcg@20', '?'))} | "
            f"{fmt(m.get('precision@5', '?'))} | "
            f"{fmt(m.get('total_ms', '?'), 'ms')} | {note} |"
        )

    lines += [
        "",
        "## 4. Retrieval Strategy Comparison (S1 vs S2)\n",
        "| Strategy | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |",
        "|----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|",
    ]

    for exp_id in ["E1_L1_D2_S1", "E1_L1_D2_S2"]:
        m = results.get(exp_id, {})
        note = "Ablation (no LLM)" if "S1" in exp_id else "Complete RAG"
        ret_desc = m.get('retrieval_strategy', exp_id)
        if isinstance(ret_desc, dict):
            ret_desc = ret_desc.get('desc', str(ret_desc))
        lines.append(
            f"| {ret_desc} | "
            f"{fmt(m.get('ndcg@5', '?'))} | "
            f"{fmt(m.get('ndcg@10', '?'))} | "
            f"{fmt(m.get('ndcg@20', '?'))} | "
            f"{fmt(m.get('precision@5', '?'))} | "
            f"{fmt(m.get('total_ms', '?'), 'ms')} | {note} |"
        )

    lines += [
        "",
        "## 5. Overall Summary\n",
        "",
        "| Experiment | Config | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Recall@5 | MRR | Total Time |",
        "|-----------|--------|:--------:|:-------:|:-------:|:-----------:|:--------:|:---:|:----------:|",
    ]

    for exp_id in ALL_EXPERIMENTS:
        m = results.get(exp_id, {})
        include_timing = True
        if m.get('total_ms') in ('N/A', 'ERROR', '?'):
            include_timing = False
        lines.append(
            f"| {exp_id} | "
            f"{m.get('config_desc', exp_id)} | "
            f"{fmt(m.get('ndcg@5', '?'))} | "
            f"{fmt(m.get('ndcg@10', '?'))} | "
            f"{fmt(m.get('ndcg@20', '?'))} | "
            f"{fmt(m.get('precision@5', '?'))} | "
            f"{fmt(m.get('recall@5', '?'))} | "
            f"{fmt(m.get('mrr', '?'))} | "
            f"{fmt(m.get('total_ms', '?'), 'ms') if include_timing else 'N/A'} |"
        )

    # ── Metric interpretation section ──
    lines += [
        "",
        "---",
        "",
        "## Metric Interpretation",
        "",
        "| Metric | What it measures | Suited for multi-GT? |",
        "|--------|-----------------|:-------------------:|",
        "| **NDCG@k** (Primary) | Ranking quality: how early are relevant items? Normalised by IDCG so queries with 1 or 92 GT contribute equally. | ✅ Yes — the standard metric for graded/judged relevance |",
        "| Precision@5 | Fraction of top-5 results that are relevant. Simple to understand. | ✅ Yes — denominator fixed at k=5 |",
        "| Recall@5 | Fraction of all GT games found in top-5. | ⚠️ Biased against queries with many GT (max recall@5 = 5/92 = 0.05) |",
        "| MRR | How early does the *first* correct answer appear? | ✅ Yes — unaffected by GT count |",
        "",
        "**Why NDCG is our primary metric:**",
        "- NDCG's IDCG normalisation makes GT=1 and GT=92 queries equally weighted in the average",
        "- It rewards ranking quality, not just raw hit count",
        "- Discount factor (logarithmic) correctly values top positions over lower ones",
        "- Widely used in information retrieval and recommendation system evaluation",
        "",
    ]

    output_path = BASE_DIR / "experiments" / "results_summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments (skip completed)")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek single-GT test set")
    parser.add_argument("--deepseek-multi", action="store_true", help="Use DeepSeek multi-GT test set (recommended)")
    parser.add_argument("--force", action="store_true", help="Re-run all experiments even if cached")
    args = parser.parse_args()

    test_label, test_label_display = select_test_set(args)

    # Validate test set exists
    test_path, _ = TEST_SETS[test_label]
    if not test_path.exists():
        print(f"Error: Test set not found at {test_path}")
        print("Hint: Run experiments/deepseek_generate_multi_gt.py first if needed.")
        return

    print(f"\n{'='*60}")
    print(f"Test set: {test_label_display}")
    print(f"  {len(ALL_EXPERIMENTS)} experiments to check")
    print(f"{'='*60}")

    results = {}
    for exp_id in ALL_EXPERIMENTS:
        print(f"\n{'─'*50}")

        out_path = result_path(exp_id, test_label)
        if out_path.exists() and not args.force:
            print(f"  [{exp_id}] Loading cached result: {out_path.name}")
            with open(out_path, encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            try:
                metrics = run_and_cache(exp_id, test_label, force=args.force)
            except Exception as e:
                print(f"  [{exp_id}] !! Failed: {e}")
                metrics = {
                    "ndcg@5": "ERROR", "ndcg@10": "ERROR", "ndcg@20": "ERROR",
                    "precision@5": "ERROR", "recall@5": "ERROR", "mrr": "ERROR",
                    "retrieval_ms": "ERROR", "ranking_ms": "ERROR",
                    "generation_ms": "ERROR", "total_ms": "ERROR",
                    "embed_model": "ERROR", "llm_model": "ERROR",
                    "doc_strategy": "ERROR", "retrieval_strategy": "ERROR",
                    "config_desc": "ERROR",
                }

        results[exp_id] = metrics

        if metrics.get("total_ms") not in ("N/A", "ERROR", "?"):
            print(f"  -> NDCG@5: {metrics.get('ndcg@5', '?'):.4f}  |  "
                  f"P@5: {metrics.get('precision@5', '?'):.4f}  |  "
                  f"MRR: {metrics.get('mrr', '?'):.4f}  |  "
                  f"{metrics.get('total_ms', '?'):.1f}ms")
        else:
            print(f"  -> {metrics.get('ndcg@5', '?')} (skipped/failed)")

    save_markdown(results, test_label_display)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
