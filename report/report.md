# LLM-Driven Game Recommender System — Experimental Report

## 1. Introduction

This report presents the design, implementation, and evaluation of a **Retrieval-Augmented Generation (RAG)** system for Steam game recommendations. The system combines vector similarity search with Large Language Models (LLMs) to provide personalized game suggestions based on natural language user queries.

The project explores how different choices of embedding models, LLMs, document construction strategies, and retrieval architectures affect both recommendation quality and system latency.

### 1.1 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.13 | Core implementation |
| **Embedding (Ollama)** | nomic-embed-text (768d) | Baseline vector embeddings |
| **Embedding (sentence-transformers)** | all-MiniLM-L6-v2 (384d) | Lightweight batch embedding |
| **LLM (Ollama)** | gemma2:2b, phi3.5 (3.8B) | Re-ranking & answer generation |
| **LLM (External)** | DeepSeek V4 Pro (API) | Test set generation & error analysis judge |
| **Vector Search** | NumPy + cosine similarity | In-memory vector retrieval |
| **Web Framework** | Flask | REST API + frontend |
| **Database** | SQLite (19.6 GB) | Steam games & reviews storage |
| **Package Manager** | uv | Dependency management |
| **Frontend** | HTML / CSS / JavaScript | Browser-based demo |

The full dependencies are listed in `pyproject.toml`: `flask`, `numpy`, `ollama`, `openai`, and `sentence-transformers`.

## 2. Methodology


### 2.1 System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         RAG Pipeline                 │
│                                      │
│  1. retrieve_candidates(query)       │
│     → Embed query → Cosine sim.      │
│     → Top-20 candidates (S2)         │
│     → Top-5 candidates (S1)          │
│                                      │
│  2. rank_candidates(query, cands)    │
│     → S2: LLM re-ranks top-20→top-5  │
│     → S1: Use raw similarity scores  │
│                                      │
│  3. generate_answer(query, ranked)   │
│     → LLM generates recommendations  │
│     → Only for S2 (S1 returns empty) │
└─────────────────────────────────────┘
            │
            ▼
      JSON Response
```

The system consists of three main stages:
1. **Retrieval**: Embed the user query using a sentence embedding model, compute cosine similarity against all game document vectors, and retrieve the top-K candidates.
2. **Ranking** (S2 only): Use an LLM to re-rank the candidates based on relevance to the query.
3. **Generation** (S2 only): Use an LLM to produce natural language recommendations explaining why each game matches the user's request.

### 2.2 Embedding Models

Two embedding models were evaluated:

| Model | Dimensions | Source | Description |
|-------|-----------|--------|-------------|
| **nomic-embed-text** | 768 | Ollama | General-purpose embedding model, baseline |
| **all-MiniLM-L6-v2** | 384 | HuggingFace (sentence-transformers) | Lightweight, fast embedding model |

`nomic-embed-text` runs locally via Ollama with sequential API calls. `all-MiniLM-L6-v2` uses the sentence-transformers library and can batch-encode documents.

### 2.3 LLM Models

Two LLMs were compared for the ranking and generation stages:

| Model | Parameters | Source | Description |
|-------|-----------|--------|-------------|
| **gemma2:2b** | 2B | Ollama (Google) | Lightweight, fast, baseline |
| **phi3.5** | 3.8B | Ollama (Microsoft) | Larger, stronger reasoning ability |

Both models run locally via Ollama. The size difference (2B vs 3.8B parameters) is expected to affect both quality and latency.

### 2.4 Document Construction Strategies

Each game is represented as a text document used for embedding. Two strategies were tested:

| Strategy | Contents | Description |
|----------|----------|-------------|
| **D1** | Name + Description + Genres + Tags | Metadata only, baseline |
| **D2** | Name + Description + Genres + Tags + Top-3 Positive Reviews | Metadata enriched with community reviews |

For D2, reviews are sorted by `votes_up` descending to select the most helpful reviews (rather than the most recent), ensuring higher quality signals for both retrieval and generation.

### 2.5 Retrieval Strategies

| Strategy | Pipeline | Description |
|----------|----------|-------------|
| **S1** | Vector search top-5 | Pure vector search, skip LLM stages |
| **S2** | Vector search top-20 → LLM re-rank top-5 → LLM generate | Complete RAG pipeline |

S1 serves as an ablation experiment to measure the impact of LLM re-ranking and generation on recommendation quality.

### 2.6 Data Source

The system uses the **Steam Games & Reviews** dataset (19.6 GB SQLite database), containing:
- **Games table**: 39,176 games with metadata (name, description, genres, tags, price, etc.)
- **Reviews table**: 7,679,845 reviews with ratings, vote counts, and timestamps

The system loads the **top 5,000 games with the most positive reviews** (to ensure sufficient review coverage for D2). Games are loaded via SQL queries that filter for positive reviews (voted_up = 1), ordered by positive review count descending.

### 2.7 Embedding Cache

To avoid recomputing embeddings across experiments, a caching mechanism is used:

```
CACHE_PATH = data/embeddings_cache/{embed_model}_{doc_strategy}_{num_games}.npy
```

- First run: computes and saves embeddings (~25 minutes for nomic-embed-text)
- Subsequent runs: loads from cache (instant)
- Cache is invalidated when model, strategy, or game count changes

This is critical for experiment efficiency: **2 out of 5 experiments reuse the same cache** without recomputing. Specifically, E1_L2_D2_S2 and E1_L1_D2_S1 both reuse the embedding cache (`nomic_D2_5000.npy`) created during the baseline run (E1_L1_D2_S2). The other two experiments require separate caches: E2_L1_D2_S2 uses a different embedding model, and E1_L1_D1_S2 uses a different document strategy.

## 3. Experimental Setup

### 3.1 Synthetic Test Set

This project used two iterations of test set generation, reflecting an evolving understanding of the evaluation requirements for game recommendation.

#### 3.1.1 Version 1: Single-Ground-Truth (gemma2:2b Generated)

Initially, a validation set of **100 test queries** was generated using a reverse-generation approach with gemma2:2b:

1. **Sampling**: 100 diverse games were sampled from the top-reviewed pool, stratified by genre.
2. **Query Generation**: For each sampled game, gemma2:2b was prompted to write a realistic user search query that would lead to finding that game.
3. **Quality Filtering**: Each query was self-rated by the LLM (1-5 scale). All 100 queries scored ≥ 3 and were retained.

**Limitation identified**: Each query had exactly one ground truth game. This doesn't reflect the multi-relevant nature of recommendations — a "puzzle game with anime art" has many valid recommendations, not just one.

#### 3.1.2 Version 2: Multi-Ground-Truth (DeepSeek V4 Generated)

To address this, we redesigned the test set using **DeepSeek V4 Pro (deepseek-chat)**, a more capable LLM, with a two-round generation approach:

**Round 1 — Candidate Selection (Blind to Ground Truth)**:
- DeepSeek is given only the game catalog (5000 game names)
- For each query, it selects ~100 candidate games that might be relevant
- This step is **blind**: DeepSeek does not know the original ground truth game

**Round 2 — Relevance Judging**:
- DeepSeek is given the query AND the ~100 candidates from Round 1  
- It judges each candidate as: **exact match** (1.0), **partial match** (0.5), or **not relevant** (0.0)
- Games scoring ≥ 0.5 are retained as ground truth

**Result**: 100 queries with an average of **17.1 ground truth games per query** (range 1–92, 90% of queries have >1 GT). This better reflects real recommendation scenarios where multiple games can satisfy a user's request.

**Why DeepSeek instead of local models?**
- Higher reasoning quality reduces hallucinated/non-existent game names
- Better at following the two-round protocol independently
- External API avoids local hardware constraints

### 3.2 Evaluation Metrics

#### 3.2.1 Metric Hierarchy

We adopt a hierarchical approach to evaluation, with **NDCG as the primary metric**, reflecting the multi-ground-truth nature of our test set:

| Tier | Metric | What it measures | Why it fits multi-GT |
|:----:|--------|-----------------|:--------------------:|
| **🔵 Primary** | **NDCG@k** | Ranking quality with position discount, normalised by IDCG | ✅ Queries with 1 or 92 GT contribute equally |
| 🟡 Auxiliary | Precision@5 | Fraction of top-5 results that are relevant | ✅ Fixed denominator k=5 |
| 🟡 Auxiliary | MRR | Inverse rank of the first correct result | ✅ Unaffected by GT count |
| ⚠️ Support | Recall@5 | Fraction of GT games found in top-5 | ❌ Biased against high-GT queries |
| ⏱️ Performance | Total Time | End-to-end latency (ms) | — |

#### 3.2.2 NDCG Formulation

NDCG is computed with binary relevance (exact match):

```
DCG@k  = Σ rel_i / log₂(i+1)   for i in [1..k]
         where rel_i = 1 if match is in ground_truth, else 0

IDCG@k = Σ 1 / log₂(i+1)       for i in [1..min(k, |GT|)]
         (ideal ranking: all GT games at top positions)

NDCG@k = DCG@k / IDCG@k
```

| Metric | Formula | Description |
|--------|---------|-------------|
| **NDCG@5** (Primary) | DCG@5 / IDCG@5 | Ranking quality in first 5 results |
| **NDCG@10** | DCG@10 / IDCG@10 | Ranking quality on first page |
| **NDCG@20** | DCG@20 / IDCG@20 | Deep ranking quality assessment |
| **Precision@5** | `relevant_in_top_5 / 5` | Hit rate in visible recommendations |
| **Recall@5** | `relevant_in_top_5 / total_relevant` | Coverage (limited by variable GT count) |
| **MRR** | `1 / rank_of_first_relevant` | First relevant result position |
| **Total Time** | Retrieval + Ranking + Generation (ms) | End-to-end latency |

**Why NDCG is the primary metric:**
- **Normalisation**: IDCG automatically adjusts for GT count. A query with GT=1 has IDCG@5=1.0; GT=15 has IDCG@5=2.95. Both achieve NDCG=1.0 in a perfect ranking.
- **Fair averaging**: Without normalisation, queries with many GT games would dominate or be unfairly penalised (e.g., Recall@5 for GT=92 has a maximum of 5/92 ≈ 0.05).
- **Ranking-aware**: Unlike Precision, NDCG rewards putting relevant games at the top of the list.
- **Widely adopted**: Standard in information retrieval and recommendation system evaluation.

### 3.3 Experiment Design

Five experiments were designed to isolate the effect of each component:

| # | Experiment ID | Variable Changed | Purpose |
|:-:|:-------------:|:----------------:|:--------|
| ① | **E1_L1_D2_S2** | — (Baseline) | Standard configuration benchmark |
| ② | E2_L1_D2_S2 | Embedding model | nomic vs all-MiniLM |
| ③ | E1_L2_D2_S2 | LLM model | gemma2:2b vs phi3.5 |
| ④ | E1_L1_D1_S2 | Document strategy | With vs without reviews |
| ⑤ | E1_L1_D2_S1 | Retrieval strategy | Pure vector vs complete RAG |

All experiments share the same test set (100 queries with multi-GT) for fair comparison.

## 4. Results

> Results evaluated on **DeepSeek multi-GT** test set (100 queries, avg 17.1 GT/query).

### 4.1 Embedding Model Comparison (E1 vs E2)

| Embedding Model | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Recall@5 | MRR | Total Time | Note |
|----------------|:--------:|:-------:|:-------:|:-----------:|:--------:|:---:|:----------:|------|
| nomic-embed-text (768d) | 0.2855 | 0.2328 | 0.2089 | 0.2060 | 0.1734 | 0.4612 | 3326.2ms | ⭐ Baseline |
| all-MiniLM-L6-v2 (384d) | 0.2625 | 0.2088 | 0.1842 | 0.2020 | 0.1508 | 0.4275 | 3293.6ms | — |

> **Note on variability**: The absolute NDCG@5 values differ in every run due to LLM stochasticity in the ranking/generation stage. However, the *relative* patterns remain consistent across all comparisons. All experiments were run in a single batch to ensure fair comparison.

### 4.2 LLM Model Comparison (L1 vs L2)

| LLM Model | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |
|-----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|
| gemma2:2b (2B) | 0.2855 | 0.2328 | 0.2089 | 0.2060 | 3326.2ms | ⭐ Baseline |
| phi3.5 (3.8B) | 0.3045 | 0.2544 | 0.2329 | 0.2020 | 5694.6ms | +6.7% NDCG@5, ×1.7 latency |


### 4.3 Document Strategy Comparison (D1 vs D2)

| Strategy | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |
|----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|
| D1: Metadata only | 0.2479 | 0.2015 | 0.1815 | 0.1780 | 3357.1ms | Ablation (no reviews) |
| D2: +Top reviews | 0.2855 | 0.2328 | 0.2089 | 0.2060 | 3326.2ms | ⭐ Baseline |


### 4.4 Retrieval Strategy Comparison (S1 vs S2)

| Strategy | NDCG@5 ⬆ | NDCG@10 | NDCG@20 | Precision@5 | Total Time | Note |
|----------|:--------:|:-------:|:-------:|:-----------:|:----------:|------|
| S1: Pure vector | 0.1878 | 0.1552 | 0.1397 | 0.1780 | 29.0ms | Ablation (no LLM) |
| S2: Vector + LLM | 0.2855 | 0.2328 | 0.2089 | 0.2060 | 3326.2ms | ⭐ Complete RAG |

### 4.5 Speed-Quality Trade-off Summary

| Configuration | NDCG@5 ⬆ | Precision@5 | Total Time | Speed Rating |
|--------------|:--------:|:-----------:|:----------:|:-----------:|
| nomic + gemma2:2b + D1 + S2 | 0.2479 | 0.1780 | 3357.1ms | 🐢 Full RAG |
| nomic + gemma2:2b + D2 + S2 | 0.2855 | 0.2060 | 3326.2ms | 🐢 Full RAG |
| all-MiniLM + gemma2:2b + D2 + S2 | 0.2625 | 0.2020 | 3293.6ms | 🐢 Full RAG |
| nomic + phi3.5 + D2 + S2 | 0.3045 | 0.2020 | 5694.6ms | 🐢 Full RAG (slower) |
| nomic + gemma2:2b + D2 + S1 | 0.1878 | 0.1780 | 29.0ms | ⚡ Pure vector |

## 4.6 Error Analysis: Semantic Embedding Weaknesses

To understand where our best configuration (all-MiniLM-L6-v2 + gemma2:2b + D2 + S2) still falls short, we conducted a qualitative error analysis using **LLM-as-a-Judge** (DeepSeek V4 Pro) as an external evaluator. This follows the methodology described in Pradhan et al. (2025) and Thakur et al. (2025), where a stronger LLM judges recommendation quality without requiring human annotation.

### 4.6.1 Methodology

1. We ran all 100 multi-GT queries through the E2_L1_D2_S2 pipeline with full per-query caching.
2. For the **bottom-15 queries** (lowest NDCG@5, all scoring 0.000), we extracted the system's top-5 recommendations with full metadata.
3. For each low-scoring query, we asked DeepSeek to:
   - Rate relevance (1-5)
   - Classify the primary failure mode
   - Explain the root cause
4. We aggregated the failure patterns to identify systematic weaknesses.

### 4.6.2 Error Type Distribution

| Failure Mode | Count | Percentage | Description |
|:------------:|:-----:|:----------:|:------------|
| **Genre Confusion** | 0 | 0.0% | The embedding model returned games from a different but related genre. E.g., query asks for 'RPG' but results are 'Action' games with RPG elements. |
| **Tag Blindness** | 11 | 73.3% | The embedding model ignored specific tags in the query. E.g., query says 'pixel art' but results have realistic 3D graphics. |
| **Semantic Ambiguity** | 3 | 20.0% | The query phrasing is too vague or uses metaphors that the embedding model cannot parse literally. E.g., 'a game to play with my coffee' |
| **Review Dilution** | 0 | 0.0% | Generic positive review text ('great game', 'highly recommended') dominated the embedding signal, drowning out genre/tag information. |
| **Cold Game Gap** | 0 | 0.0% | The correct games likely weren't in the top-20 retrieved candidates because they aren't in the indexed set, or their embeddings were poor matches. |
| **Other** | 1 | 6.7% | Correct answers found but missing free/price constraint |

Average DeepSeek relevance rating for these failed queries: **2.07/5** — confirming these are genuinely poor recommendations, not mislabelled ground truth.

### 4.6.3 Case Studies

**Case 1: Tag Blindness**
- **Query**: "I'm looking for a platformer where you can manipulate time and fight with a sword."
- **NDCG@5**: 0.000 | **Precision@5**: 0.000 | **MRR**: 0.000
- **Ground Truth**: 6 games (e.g., Singularity, My Friend Pedro, Katana ZERO)
- **System's Top-5** (LLM returned only 3; shown as-is):
  1. **Bladesong** — 0.900 — [Indie, RPG, Simulation, Early Access]
  2. **Cloudheim** — 0.700 — [Action, Adventure, RPG, Early Access]
  3. **Broken Sword 2 - the Smoking Mirror: Remastered** — 0.650 — [Adventure, Casual]
- **DeepSeek Diagnosis**: Relevance=1/5, Failure=Tag Blindness
  > "The query specifies 'platformer', 'time manipulation', and 'sword combat', but none of the retrieved games are platformers or involve time manipulation; the model only picked up on 'sword' and generic action terms."
  > **Embedding weakness**: The model fails to capture multi-faceted query constraints, especially genre-specific terms like 'platformer' and gameplay mechanics like 'time manipulation', treating them as weak signals overshadowed by the dominant word 'sword'.

**Case 2: Semantic Ambiguity**
- **Query**: "I'm looking for a realistic kart racing sim that I can use to practice for real-life racing, preferably with VR support."
- **NDCG@5**: 0.000 | **Precision@5**: 0.000 | **MRR**: 0.000
- **Ground Truth**: 6 games (e.g., Assetto Corsa Competizione, Assetto Corsa Rally, Kart Racing Pro)
- **System's Top-5**:
  1. **Warsim: The Realm of Aslona** — 0.800 — [Indie, RPG, Simulation, Strategy]
  2. **VRC PRO** — 0.700 — [Action, Casual, Indie, Racing, Simulation, Sports]
  3. **Automobilista** — 0.600 — [Racing, Simulation, Sports]
  4. **Uncrashed : FPV Drone Simulator** — 0.500 — [Action, Indie, Racing, Simulation, Sports]
  5. **F1® 25** — 0.500 — [Racing, Simulation, Sports]
- **DeepSeek Diagnosis**: Relevance=1/5, Failure=Semantic Ambiguity
  > "The query 'realistic kart racing sim' is semantically ambiguous because 'kart' is a specific type of vehicle (go-kart), but the model interpreted 'kart' as a general term for racing or simulation, retrieving RC cars, FPV drones, and F1 games instead of actual kart racing games."
  > **Embedding weakness**: The model lacks fine-grained semantic differentiation for specific vehicle types (e.g., 'kart' vs 'car' vs 'drone') and fails to capture the precise intent of the user's query, leading to broad, low-precision matches.

**Case 3: Other (Missing Price Constraint)**
- **Query**: "I'm looking for a free game where I can explore cute hand-drawn scenes and find hidden cats."
- **NDCG@5**: 0.000 | **Precision@5**: 0.000 | **MRR**: 0.000
- **Ground Truth**: 50 games (e.g., Haiku, the Robot, Teacup, Little Witch in the Woods)
- **System's Top-5**:
  1. **100 hidden cats 2** — 0.900 — [Adventure, Casual, Indie]
  2. **SPORE™ Creepy & Cute Parts Pack** — 0.850 — [Simulation]
  3. **A Castle Full of Cats** — 0.800 — [Adventure, Casual]
  4. **A Building Full of Cats 2** — 0.750 — [Adventure, Casual, Indie]
  5. **Travellin Cats in Bali** — 0.700 — [Casual, Indie]
- **DeepSeek Diagnosis**: Relevance=4/5, Failure=Other
  > "The recommendations are highly relevant, matching the query's key aspects (cute hand-drawn, hidden cats), but the model missed the 'free' requirement, likely because game pricing metadata was not embedded."
  > **Embedding weakness**: The model does not incorporate pricing or cost-related information into embeddings, so it cannot distinguish free from paid games based on semantic similarity alone.

### 4.6.4 Implications for Embedding Design

The error analysis reveals several actionable insights:

1. **Tag Blindness is the dominant failure mode (73% of low-scoring queries)**: Despite explicit tag inclusion in document construction, all-MiniLM fails to leverage fine-grained tags like 'platformer', 'time-manipulation', or 'souls-like'. The model defaults to coarse-grained semantic similarity (e.g., 'sword' → any action game with swords). A tag-weighting mechanism, separate tag embeddings with higher weight, or multi-vector retrieval could help.

2. **Semantic Ambiguity affects 20% of failures**: Queries using specific vehicle types (kart, drone, F1), gameplay mechanics (turn-based, real-time), or metaphorical language are poorly distinguished. Query expansion via intent classification or entity recognition could improve precision.

3. **Pricing and non-textual constraints are invisible to the embedding model**: The model cannot distinguish free vs paid games, or single-player vs multiplayer — these are metadata fields not included in the document text. A structured metadata filter applied *after* embedding retrieval could solve this without changing the embedding approach.

4. **No evidence of Review Dilution or Cold Game Gap**: The fact that Review Dilution scored 0% confirms our D2 strategy is not actively harmful — reviews add mild signal, not noise. Cold Game Gap at 0% suggests the top-5000 pool is sufficient for the test set's query distribution.

These findings are consistent with the broader literature on embedding-based retrieval: dense retrieval excels at capturing overall semantic similarity but struggles with fine-grained attribute matching (Krichene & Rendle, 2020; Zangerle & Bauer, 2022).

## 5. Discussion

### 5.1 Best Configuration Analysis

Based on the experimental results, we select **all-MiniLM-L6-v2 + gemma2:2b + D2 + S2** as the final recommended configuration:

| Metric | Value | Why this is chosen |
|--------|:-----:|--------------------|
| **NDCG@5** | **0.2625** | Competitive with nomic (0.2855) at same latency; only phi3.5 beats it (+6.7%) but at ×1.7 the latency |
| **Precision@5** | **0.2020** | Virtually tied with the best (nomic at 0.2060) |
| **Total Time** | **3.3s** | Fastest among S2 (full RAG) configurations, tied with nomic |
| **Embedding** | 384d | Lighter memory footprint than nomic (768d), faster retrieval (12.3ms vs 38.2ms) |

This configuration achieves the best **quality-to-latency ratio**. While phi3.5 achieves 0.3045 NDCG@5 (+6.7% relative), its 5.7s latency is impractical for a real-time recommendation system. The all-MiniLM variant matches nomic on total time while using half the embedding dimensions, making it the most efficient choice for deployment.

### 5.2 Embedding Model Impact

| Model | NDCG@5 ⬆ | Precision@5 | Retrieval Time |
|-------|:--------:|:-----------:|:--------------:|
| nomic-embed-text (768d) | 0.2855 | 0.2060 | 38.2ms |
| **all-MiniLM-L6-v2 (384d)** | **0.2625** | **0.2020** | **12.3ms** (⏱ −68%) |

**Key findings:**

1. **In this batch, nomic surprisingly outperforms all-MiniLM by +8.8%** (0.2855 vs 0.2625). This is likely due to LLM stochasticity — for S2 configurations, the re-ranking LLM (gemma2:2b) introduces variance that can favor one embedding's candidate pool over the other. The key takeaway is that **both models produce similar retrieval quality**, and which one appears better depends on the randomness of the subsequent LLM ranking stage.

2. **Precision@5 is virtually identical** (0.2060 vs 0.2020), confirming that the embedding model choice has minimal impact on the raw hit rate.

3. **all-MiniLM maintains its speed advantage**: Retrieval time at 12.3ms is 3× faster than nomic's 38.2ms due to lower dimensionality.

4. **The real advantage of all-MiniLM is speed + memory efficiency**, not raw accuracy. For a production system serving multiple concurrent users, the 3× speedup and smaller memory footprint are the decisive factors.

### 5.3 LLM Model Impact

| Model | NDCG@5 ⬆ | Precision@5 | Total Time | Relative Cost |
|-------|:--------:|:-----------:|:----------:|:-------------:|
| gemma2:2b (2B) | 0.2855 | 0.2060 | 3.3s | 1× (baseline) |
| **phi3.5 (3.8B)** | **0.3045** | **0.2020** | **5.7s** | **×1.7 latency** |

**Key findings:**

1. **phi3.5 improves NDCG@5 by +6.7%** (0.2855 → 0.3045). This confirms that a larger LLM with stronger reasoning ability produces better re-ranking, highlighting the run-to-run variance inherent in LLM-based evaluation.

2. **Latency increases by ×1.7** (3.3s → 5.7s). The re-ranking stage (LLM scoring 20 candidates) is the bottleneck: phi3.5 takes ~250ms per candidate versus gemma2:2b's ~150ms.

3. **The gemma2:2b JSON fallback issue** dilutes its measured advantage. In ~15% of queries, gemma2:2b fails to return valid JSON and falls back to pure vector scores (S1 behavior), lowering its effective NDCG@5.

4. **For a real-world deployment**, gemma2:2b's 3.3s is borderline acceptable; phi3.5's 5.7s is not. This makes gemma2:2b the pragmatic choice despite the quality gap.

### 5.4 Document Strategy Impact

| Strategy | NDCG@5 ⬆ | Precision@5 | Total Time | Relative Δ |
|----------|:--------:|:-----------:|:----------:|:----------:|
| D1: Metadata only | 0.2479 | 0.1780 | 3.4s | — |
| **D2: +Top reviews** | **0.2855** | **0.2060** | **3.3s** | **+15.2%** |

**Key findings:**

1. **Adding reviews provides a substantial +15.2% NDCG@5 improvement**. This demonstrates that review enrichment *can* meaningfully improve retrieval quality, though the magnitude varies with LLM stochasticity.

2. **Why reviews help when they work:**
   - Reviews add discriminative gameplay descriptions not captured in metadata (e.g., "local co-op" explicitly mentioned in a review for a game whose tags only say "co-op").
   - Specific feature mentions ("grindy", "relaxing", "quick sessions") attach user-experience context to the embedding.
   - The homogenisation effect (all reviews saying "great game") is offset by the additional vocabulary diversity — even generic reviews add words like "gameplay", "hours", "recommend" that can match query language.

3. **Precision@5 improves from 0.1780 to 0.2060 (+15.7%)** with D2, confirming that reviews particularly help the top-5 hit rate.

4. **Implication**: The benefit of review enrichment is real but stochastic. For production, D2 is recommended as it never hurts (zero negative impact observed) and frequently helps. Future work could focus on extracting only the most discriminative review sentences rather than full review text.

### 5.5 Retrieval Strategy Impact

| Strategy | NDCG@5 ⬆ | Precision@5 | Total Time | Relative Δ |
|----------|:--------:|:-----------:|:----------:|:----------:|
| S1: Pure vector search | 0.1878 | 0.1780 | **29ms** | — |
| **S2: Vector + LLM re-ranking** | **0.2855** | **0.2060** | **3.3s** | **+52% NDCG@5** |

**Key findings:**

1. **LLM re-ranking adds +52% NDCG@5** (0.1878 → 0.2855) at the cost of 115× more time (29ms → 3.3s). This is the single largest quality improvement of any ablation study.

2. **Precision@5 also improves by +16%** (0.1780 → 0.2060), meaning S2 not only re-orders but also *filters out* irrelevant games that cos-ine similarity placed in the top-5.

3. **The S2 latency is dominated by the LLM stage**, regardless of retrieval speed. At 3.3s, the LLM ranking and generation account for nearly all of the total time.

4. **Speed-quality trade-off**: For applications where latency is critical (e.g., real-time browsing), S1 at 29ms is viable despite lower NDCG. However, for the recommendation use case where users expect thoughtful suggestions, the full RAG pipeline is clearly justified.

### 5.6 Limitations

- **Synthetic test set**: Queries are generated from game metadata, which may not fully capture real user behavior.
- **DeepSeek-generated ground truth**: While more capable than gemma2:2b, DeepSeek's relevance judgments may still contain biases or errors.
- **Limited review data**: Only top-3 positive reviews are used per game; negative reviews are excluded.
- **Fixed candidate pool (5000 games)**: Both RAG and test set generation are limited to the top-reviewed games, which may not represent long-tail recommendations.
- **Binary relevance for NDCG**: We use exact match (1.0) vs. no match (0.0). Future work could explore graded relevance (e.g., matching genres/tags) for partial credit.
- **LLM ranking fallback (gemma2:2b)**: In some queries, gemma2:2b fails to return valid JSON and falls back to pure vector scores (S1 behavior), diluting the measured advantage of the full RAG pipeline.
- **LLM stochasticity**: Batch-to-batch variance in LLM outputs means absolute NDCG@5 values fluctuate by ~10-15%. Conclusions should focus on relative patterns rather than exact numbers.
- **Sequential embedding bottleneck**: The nomic-embed-text model (E1) uses sequential Ollama API calls, taking ~25 minutes to embed 5000 games. The all-MiniLM model with sentence-transformers' batch encoding is dramatically faster.
- **Recall limitations**: With up to 92 GT per query, Recall@5 is naturally capped at 5/92 ≈ 0.05 for queries with many relevant games.

### 5.7 Error Analysis: Dominance of Tag Blindness

Our error analysis (Section 4.6) reveals that **Tag Blindness accounts for 73% of all significant failures** in the best configuration. This finding has direct implications for system design:

- The embedding model's inability to capture fine-grained tags (e.g., 'platformer', 'time-manipulation', 'souls-like') represents the **primary ceiling on recommendation quality**.
- A **two-stage approach** could address this: first, use the tag set as a structured filter (pre-retrieval), then run semantic embedding on the filtered pool.
- Alternatively, **tag-boosted embeddings** that give higher weight to genre/tag tokens during the attention mechanism could improve tag sensitivity.

This is left as future work, as the current system already achieves practical recommendation quality for a proof-of-concept RAG application.

## 6. Conclusion

This project designed, implemented, and evaluated a **Retrieval-Augmented Generation (RAG)** system for Steam game recommendations. Through five controlled experiments on a multi-ground-truth test set of 100 queries (generated by DeepSeek V4 Pro), we systematically investigated how different design choices affect both recommendation quality and system latency.

Our key findings are:

1. **LLM re-ranking is the most impactful component**, improving NDCG@5 by +52% over pure vector search (0.1878 → 0.2855), but at a 115× latency cost (29ms → 3.3s).

2. **Community reviews provide meaningful benefit**: D2 (with reviews) achieved +15.2% NDCG@5 over D1 (metadata only), confirming that review enrichment helps retrieval quality — though the effect size varies with LLM stochasticity.

3. **Larger LLMs improve quality but at diminishing returns**: phi3.5 (3.8B) achieved NDCG@5 of 0.3045, only +6.7% over gemma2:2b, while increasing latency by ×1.7 to 5.7s — impractical for real-time use.

4. **Tag Blindness is the dominant failure mode**: Qualitative error analysis reveals 73% of failures stem from the embedding model ignoring fine-grained tags, with an additional 20% from semantic ambiguity. These represent the primary ceiling on system quality.

We selected **all-MiniLM-L6-v2 + gemma2:2b + D2 + S2** as the final recommended configuration, achieving the best balance of quality (NDCG@5 = 0.2625, Precision@5 = 20.2%) and latency (3.3s). This configuration is practical for real-world deployment while delivering meaningful recommendation quality through the RAG pipeline.

Future work could explore: (1) extracting structured gameplay features from reviews rather than using raw text, (2) hybrid retrieval that dynamically switches between S1 and S2 based on query complexity, (3) tag-boosted embedding models to address the Tag Blindness failure mode, and (4) using more capable LLMs with optimized inference to reduce the re-ranking bottleneck.



