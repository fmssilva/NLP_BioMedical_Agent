# Phase 1 — Comprehensive Project Status & Quality Audit

> **Audit date:** 2026-07-14  
> **Auditor:** Full file-by-file review of every source file, notebook, report, guide, reference, and plan  
> **Goal:** Verify all required tasks for grade 20, identify extra work, suggest improvements

---

## Table of Contents

- [1. Requirements Checklist (Project Guide)](#1-requirements-checklist-project-guide)
- [2. Professor's Notes Compliance](#2-professors-notes-compliance)
- [3. Code Quality vs \_guidelines.md](#3-code-quality-vs-guidelinesmd)
- [4. Extra Tasks Completed (Beyond Requirements)](#4-extra-tasks-completed-beyond-requirements)
- [5. Suggested Additional Tasks](#5-suggested-additional-tasks)
- [6. Potential Issues / Risks](#6-potential-issues--risks)
- [7. Full Task Matrix](#7-full-task-matrix)
- [8. Verdict](#8-verdict)

---

## 1. Requirements Checklist (Project Guide)

The project guide (`proj_guide_NLPS_-_TREC_2025_BioGen.md`) specifies these **explicit Phase 1 requirements**:

### 1.1 Document Retrieval — 4 Required Strategies

| #   | Required Strategy               | Status | Implementation                       | Notes                                                     |
| --- | ------------------------------- | ------ | ------------------------------------ | --------------------------------------------------------- |
| 1   | **BM25 similarity**             | ✅ Done | `src/retrieval/bm25.py`              | k1=1.2, b=0.75 default; tuned to k1=1.5, b=1.0            |
| 2   | **LM Jelinek-Mercer smoothing** | ✅ Done | `src/retrieval/lm_jelinek_mercer.py` | Both λ=0.1 and λ=0.7 implemented as separate index fields |
| 3   | **LM Dirichlet smoothing**      | ✅ Done | `src/retrieval/lm_dirichlet.py`      | μ=2000 default; tuned to μ=75 via sweep                   |
| 4   | **KNN with LLM embeddings**     | ✅ Done | `src/retrieval/knn.py`               | msmarco-distilbert-base-v2, 768-dim HNSW                  |

**Extra strategy not required:** RRF fusion (BM25+KNN) — `src/retrieval/rrf.py`. This is **bonus work**.

### 1.2 Index Configuration

| Requirement                                         | Status | Notes                                                                                            |
| --------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| Define similarity metrics at index field definition | ✅      | 3 custom similarities registered: `lmjm_01_similarity`, `lmjm_07_similarity`, `lmdir_similarity` |
| number_of_replicas: 0                               | ✅      | Confirmed in `index_builder.py`                                                                  |
| number_of_shards: 4                                 | ✅      | Confirmed                                                                                        |
| refresh_interval: -1                                | ✅      | Confirmed                                                                                        |
| knn: true                                           | ✅      | Confirmed                                                                                        |
| Index articles by title, abstract and PMID          | ✅      | `doc_id` (keyword), `contents` (text), `embedding` (knn_vector)                                  |

### 1.3 Evaluation — Required Metrics

| Metric                                     | Status | Implementation                                                                           | Notes                                                     |
| ------------------------------------------ | ------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Precision@10**                           | ✅      | `src/evaluation/metrics.py` → `precision_at_k()`                                         | Computed for all strategies, train+test                   |
| **Recall@100**                             | ✅      | `src/evaluation/metrics.py` → `recall_at_k()`                                            | Added 2026-03-31                                          |
| **NDCG** (graded relevance)                | ✅      | `src/evaluation/metrics.py` → `ndcg_at_k()`, `mean_ndcg_at_k()`                          | Uses graded qrels (supporting=2, neutral=1)               |
| **Precision-Recall curves**                | ✅      | `src/evaluation/metrics.py` → `pr_curve()`, `interpolated_pr_curve()`, `mean_pr_curve()` | 11-point interpolated, Lab03 pattern                      |
| **Highest AP query** (individual PR curve) | ✅      | Topic 170 (AP=0.962) — highlighted in green                                              | `src/evaluation/plots.py` → `plot_individual_pr_curves()` |
| **Lowest AP query** (individual PR curve)  | ✅      | Topic 140 (AP=0.141) — highlighted in red                                                | Same function                                             |
| **One additional query** for comparison    | ✅      | Topic 162 (AP=0.608) — highlighted in orange (median)                                    | Same function                                             |
| **mAP** across all queries                 | ✅      | `mean_average_precision()`                                                               | TREC standard: 0-relevant topics excluded                 |

### 1.4 Ground Truth Transformation

| Requirement                                | Status | Notes                                                                          |
| ------------------------------------------ | ------ | ------------------------------------------------------------------------------ |
| Analyze `citation_assessment` fields       | ✅      | `src/data/qrels_builder.py` — parses `evidence_relation` from submissions JSON |
| Assign numerical values to labels for nDCG | ✅      | supporting=2, neutral=1, others=0 (graded qrels)                               |
| Binary qrels for P@k, R@k, MAP             | ✅      | supporting=1, others=0                                                         |
| Split: odd IDs = train, even IDs = test    | ✅      | `src/data/splitter.py` — 32 train, 33 test                                     |

### 1.5 Deliverables

| Deliverable | Status | Location                                                     |
| ----------- | ------ | ------------------------------------------------------------ |
| Code        | ✅      | `src/` — 20 Python files across 6 packages                   |
| Report      | ✅      | `tasks/report.md` — comprehensive, all sections filled       |
| Notebook    | ✅      | `tasks/phase1/phase1_search.ipynb` — 39 cells, full pipeline |

---

## 2. Professor's Notes Compliance

From `prof_guide_professor_notes.md`:

| Professor Instruction                             | Status | How It's Addressed                                                                                    |
| ------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------- |
| Start with one query field, then combine all 3    | ✅      | Query field ablation in §9 of notebook: topic-only, question-only, concatenated. Winner: concatenated |
| Split 50/50 based on queries only (not documents) | ✅      | `src/data/splitter.py` — odd/even on topic IDs. 32 train / 33 test                                    |
| Odd query-topic pairs for train, even for test    | ✅      | Exactly as specified                                                                                  |
| Ground truth = file with "submission" in name     | ✅      | Uses `biogen_2024_submissions.json`                                                                   |
| Create OpenSearch index to query documents        | ✅      | Index `usernlp03` with 6 fields, 4194 docs                                                            |

---

## 3. Code Quality vs `_guidelines.md`

### 3.1 Structure & Organization

| Guideline                                             | Compliance | Notes                                                                                                    |
| ----------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| Domain-centered structure (not tests/, plots/)        | ✅          | `src/data/`, `src/indexing/`, `src/embeddings/`, `src/retrieval/`, `src/evaluation/`, `src/tuning/`      |
| Small files, single responsibility                    | ✅          | Largest file is `metrics.py` (503 lines) — acceptable for 12 metric functions + Lab03 verification tests |
| No utils.py dumping ground                            | ✅          | No utils.py anywhere. Closest is `cv_utils.py` which is focused (k-fold CV only)                         |
| Simple, clean, no over-engineering                    | ✅          | Functions are straightforward. No unnecessary abstractions                                               |
| Use what libraries provide (OpenSearch, transformers) | ✅          | Uses `opensearchpy`, `transformers`, `torch`, `numpy`, `matplotlib`                                      |
| Only implement what's currently needed                | ✅          | No speculative code. Tuning scripts are all justified by experiments                                     |

### 3.2 Comments & Logs

| Guideline                         | Compliance    | Notes                                                                                                                                           |
| --------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Short comment above each function | ✅             | All functions have a docstring or inline comment                                                                                                |
| Inline comments explaining code   | ✅             | Good density of inline comments throughout                                                                                                      |
| Concise, casual, dev-to-dev style | ✅             | e.g., "# idempotent: skip if already done", "# Lab03 pattern exactly"                                                                           |
| No emojis in code                 | ⚠️ **Partial** | Notebook markdown cells use emojis (🔍, 🟢, 🔴, 🟠, 🔬). Source `.py` files are emoji-free. **Recommendation: remove emojis from notebook markdown** |
| Logs: single-line, concise        | ✅             | Uses `print()` and `logging.warning()` appropriately                                                                                            |

### 3.3 Testing

| Guideline                                           | Compliance | Notes                                                                                        |
| --------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------- |
| Each file has `if __name__ == "__main__"` block     | ✅          | Every `.py` file in `src/` has a `__main__` block with tests                                 |
| Connection tests                                    | ✅          | `opensearch_client.py` tests connectivity with 3-fallback probes                             |
| Data tests (shapes, splits, qrels)                  | ✅          | `loader.py`, `splitter.py`, `qrels_builder.py` all have thorough tests                       |
| Retrieval tests (100 results, descending, no dupes) | ✅          | All 5 retrievers test exactly this in `__main__`                                             |
| Evaluation tests (metrics reproduce Lab03)          | ✅          | `metrics.py` `__main__` reproduces Lab03 toy example: AP(A)=1.0, AP(B)=0.7095, NDCG@4=0.9688 |
| Test small first, then full                         | ✅          | Tuning scripts have `--coarse` flags for quick testing                                       |

### 3.4 Workflow

| Guideline                                       | Compliance | Notes                                                                        |
| ----------------------------------------------- | ---------- | ---------------------------------------------------------------------------- |
| implement → local test → fix → notebook         | ✅          | All `src/` files tested via `__main__` before notebook was built             |
| Never update notebook until Python files tested | ✅          | Notebook loads pre-saved results; source code is the single source of truth  |
| Follow lab reference code closely               | ✅          | `references/README.md` maps Lab01/Lab03 patterns → project code line-by-line |

### 3.5 Report & Documentation

| Guideline                                  | Compliance | Notes                                                                          |
| ------------------------------------------ | ---------- | ------------------------------------------------------------------------------ |
| Report updated after each task             | ✅          | `tasks/report.md` has timestamped entries (2026-03-30, 2026-03-31, 2026-07-14) |
| Concise bullet points (not full prose yet) | ✅          | Report is in bullet-point format with tables                                   |
| Numbers and analysis included              | ✅          | All metric values, delta comparisons, interpretations present                  |
| Code locations referenced                  | ✅          | References to `src/` files and functions throughout                            |

---

## 4. Extra Tasks Completed (Beyond Requirements)

These go **beyond** what the project guide explicitly asks for Phase 1:

### 4.1 Extra Strategy: RRF Fusion
- **What:** Reciprocal Rank Fusion of BM25 + KNN (k=60)
- **Why it adds value:** Demonstrates understanding of hybrid retrieval. RRF wins P@10 on both train and test.
- **Effort:** `src/retrieval/rrf.py` (127 lines) + unit test for `rrf_merge()`
- **Grade impact:** ✅ Shows initiative beyond the 4 required strategies

### 4.2 Extra Metric: MRR
- **What:** Mean Reciprocal Rank — not explicitly required by the guide
- **Why it adds value:** Measures first-hit quality — directly relevant for Phase 2 answer grounding
- **Grade impact:** ✅ Additional evaluation depth

### 4.3 Hyperparameter Tuning with 5-Fold CV (3 experiments)

The project guide does **not** explicitly require hyperparameter tuning for Phase 1 — it just asks to implement the 4 strategies and evaluate them. The tuning work is substantial extra effort:

| Experiment                        | Script                                       | Effort                                                     | Finding                                   |
| --------------------------------- | -------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------- |
| LM-Dir μ sweep (7 values)         | `src/tuning/lmdir_mu_sweep.py` (373 lines)   | Added new similarity fields to live index for each μ value | μ=75 beats default μ=2000 by +0.036 MAP   |
| BM25 k1/b grid (20 valid configs) | `src/tuning/bm25_param_sweep.py` (412 lines) | Added new BM25 similarity fields for each k1/b combo       | k1=1.5, b=1.0 beats default by +0.007 MAP |
| Encoder comparison (3 models)     | `src/tuning/alt_encoder_eval.py` (421 lines) | Pure-Python exact cosine on CPU, 3 encoder models          | MedCPT beats msmarco by +0.183 MAP        |

Supporting infrastructure: `cv_utils.py` (258 lines), `tuning_plots.py` (382 lines)

- **Total tuning code:** ~1,846 lines across 5 files
- **Grade impact:** ✅ This is the strongest differentiator — shows scientific rigor, proper train/test discipline, and deep understanding of the retrieval models

### 4.4 Graded Qrels + NDCG

The guide says "assign numerical values to labels for nDCG" — implemented as:
- `build_qrels_graded()` in `qrels_builder.py` (supporting=2, neutral=1, others=0)
- `ndcg_at_k()`, `mean_ndcg_at_k()`, `results_to_ranking_graded()` in `metrics.py`
- All test results include NDCG@10 column

### 4.5 Individual Per-Query PR Curves
- Guide asks for 3 specific query PR curves (best AP, worst AP, one more)
- Implementation goes further: plots ALL 33 test-topic curves overlaid (gray) with 3 highlighted
- `plot_individual_pr_curves()` in `plots.py` — 95 lines of visualization code

### 4.6 PR Curve Interpretation Reference
- Notebook §15 includes a `plot_pr_interpretation()` cell with reference curve shapes
- Not required — shows educational depth

### 4.7 Per-Topic AP Box Plots
- `plot_per_topic_variance()` — shows AP distribution per strategy
- Reveals topic-difficulty variance — useful for Phase 2 planning

### 4.8 Train vs Test Comparison Table
- Report includes explicit train/test delta analysis for all strategies
- Shows no overfitting (|Δ MAP| ≤ 0.021)

### 4.9 Comprehensive Notebook Markdown
- 19 markdown cells with theory explanations (BM25 formula, LM-JM derivation, LM-Dir theory, RRF formula)
- LaTeX equations for all scoring functions
- Design rationale for every choice (why standard analyzer, why not tune RRF weights, why MedCPT doesn't need fine-tuning)

### 4.10 Reference Code Index
- `references/README.md` (387 lines) maps every Lab01/Lab02/Lab03 pattern to this project's code
- Quick Reference table at the end for fast lookup

---

## 5. Suggested Additional Tasks

These are things that could further strengthen the project, ordered by **impact/effort ratio**:

### 5.1 HIGH IMPACT — Should Do Before Submission

| #   | Task                                           | Why                                                                                                                                                                                                                                                                                     | Effort     |
| --- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1   | **Run tuned LM-Dir (μ=75) on test set**        | The test results still use μ=2000. The μ=75 sweep was train-only. You should re-run LM-Dir on test with μ=75 and update the test table. This is the correct scientific workflow: tune on train, evaluate tuned config on test. Without this, the +0.036 MAP gain is only a train claim. | ~30 min    |
| 2   | **Run tuned BM25 (k1=1.5, b=1.0) on test set** | Same reasoning. The test BM25 results use k1=1.2, b=0.75.                                                                                                                                                                                                                               | ~30 min    |
| 3   | **Add MedCPT KNN to index + test evaluation**  | The encoder comparison was pure-Python on train only. Adding MedCPT vectors to the index and running KNN on test would validate the +0.183 MAP claim. This is the single biggest result and should be confirmed on test.                                                                | ~2–3 hours |
| 4   | **Update RRF with tuned components**           | After items 1-3, re-run RRF with tuned BM25 + MedCPT KNN. This would be the "Phase 1 best" system.                                                                                                                                                                                      | ~1 hour    |
| 5   | **Remove emojis from notebook markdown**       | `_guidelines.md` says "NO EMOJIS ANYWHERE". The notebook has 🔍, 🟢, 🔴, etc.                                                                                                                                                                                                              | 10 min     |

### 5.2 MEDIUM IMPACT — Nice to Have

| #   | Task                               | Why                                                                                                                                                                                                                        | Effort     |
| --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 6   | **Statistical significance test**  | Paired t-test or bootstrap confidence intervals on per-topic AP differences (BM25 vs RRF, BM25 vs LM-JM). With 33 test topics, you can compute p-values. Currently the report says "essentially tied" without quantifying. | ~1–2 hours |
| 7   | **Error analysis on hard topics**  | Topics 140 (AP=0.14) and 156 (AP=0.17) — examine the actual queries and relevant docs. Why does BM25 fail? Vocabulary mismatch? Missing key terms? This shows analytical depth.                                            | ~1 hour    |
| 8   | **Query length vs AP correlation** | Plot query length (in words) vs AP per topic. Tests the hypothesis that longer queries help. Simple scatter plot + Pearson correlation.                                                                                    | ~30 min    |
| 9   | **Document length analysis**       | Plot distribution of document lengths for relevant vs non-relevant docs. Tests the b=1.0 finding (full length normalization helps).                                                                                        | ~30 min    |
| 10  | **Strategy agreement analysis**    | Venn diagram or Jaccard overlap of top-100 results between strategies. Shows what RRF fusion actually adds — how many unique docs come from each signal.                                                                   | ~1 hour    |

### 5.3 LOW IMPACT — Only If Time Permits

| #   | Task                                       | Why                                                                                                       | Effort  |
| --- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------- |
| 11  | **IDF analysis of hard vs easy topics**    | Check if hard topics have low-IDF query terms (common words). Would explain BM25 failure.                 | ~30 min |
| 12  | **Reciprocal rank distribution**           | Histogram of rank-of-first-relevant-doc across topics. Shows MRR in more detail.                          | ~20 min |
| 13  | **Train set evaluation with tuned params** | Report the tuned-params train results alongside default-params train results to show improvement clearly. | ~1 hour |
| 14  | **Confusion matrix at P@10**               | Per-strategy: how many of the top-10 are relevant, not-relevant, neutral? Uses graded qrels.              | ~30 min |

---

## 6. Potential Issues / Risks

### 6.1 Test Results Use Untuned Parameters ⚠️ IMPORTANT
The test set evaluation table in `report.md` and notebook §14 uses:
- BM25: k1=1.2, b=0.75 (default, NOT the tuned k1=1.5, b=1.0)
- LM-Dir: μ=2000 (default, NOT the tuned μ=75)
- KNN: msmarco-distilbert (NOT MedCPT)

The tuning experiments show improvements but they were only validated on the train set. **A professor reviewing this would expect the final test table to use the tuned parameters.** This is the most impactful fix available.

### 6.2 Notebook Emojis Violate Guidelines
`_guidelines.md` explicitly says "NO EMOJIS ANYWHERE". The notebook has:
- 🔍 Part A, 🔬 Part B headings
- 🟢 🔴 🟠 in §17 table
- Not critical for grading, but shows attention to detail.

### 6.3 Binary Qrels Definition
The report says binary qrels use `supporting=1, others=0`. But `recall_at_k` and `precision_at_k` in the code use these binary qrels — which count ONLY "supporting" as relevant. This is correct for the stated definition but could be questioned: should "neutral" citations (score=1 in graded) also count as relevant for binary metrics? The current choice is defensible but should be explicitly justified in the report (it is, in the qrels section — ✅).

### 6.4 No `ranx` Usage
The guidelines mention ranx as a resource. The project implements all metrics from scratch following Lab03 patterns. This is explicitly justified in the report and is the correct approach for a learning project. ✅ No issue.

### 6.5 PHASE_1_PLAN.md Says "Step 15 tested 2026-03-30" with Only MAP/MRR/P@10
The plan shows test results without R@100 and NDCG@10 columns. These metrics were added 2026-03-31. The report and notebook include all 5 metrics. The plan is slightly outdated but the deliverables are correct.

---

## 7. Full Task Matrix

### 7.1 Required Tasks (from project guide)

| #   | Task                                | Status | Location                                             |
| --- | ----------------------------------- | ------ | ---------------------------------------------------- |
| R1  | Load PubMed corpus                  | ✅      | `src/data/loader.py` → `load_corpus()`               |
| R2  | Load query topics                   | ✅      | `src/data/loader.py` → `load_topics()`               |
| R3  | Train/test split (odd/even)         | ✅      | `src/data/splitter.py`                               |
| R4  | Build ground truth from submissions | ✅      | `src/data/qrels_builder.py`                          |
| R5  | Assign numerical values for nDCG    | ✅      | `build_qrels_graded()` — supporting=2, neutral=1     |
| R6  | Create OpenSearch index             | ✅      | `src/indexing/index_builder.py`                      |
| R7  | Index documents                     | ✅      | `src/indexing/document_indexer.py`                   |
| R8  | BM25 retrieval                      | ✅      | `src/retrieval/bm25.py`                              |
| R9  | LM-JM retrieval                     | ✅      | `src/retrieval/lm_jelinek_mercer.py`                 |
| R10 | LM-Dirichlet retrieval              | ✅      | `src/retrieval/lm_dirichlet.py`                      |
| R11 | KNN with LLM embeddings             | ✅      | `src/retrieval/knn.py` + `src/embeddings/encoder.py` |
| R12 | Precision@10                        | ✅      | `src/evaluation/metrics.py`                          |
| R13 | Recall@100                          | ✅      | `src/evaluation/metrics.py`                          |
| R14 | NDCG (graded)                       | ✅      | `src/evaluation/metrics.py`                          |
| R15 | PR curves (system-wide)             | ✅      | `src/evaluation/metrics.py` + `plots.py`             |
| R16 | Individual PR: best AP query        | ✅      | Topic 170 highlighted                                |
| R17 | Individual PR: worst AP query       | ✅      | Topic 140 highlighted                                |
| R18 | Individual PR: one additional query | ✅      | Topic 162 (median) highlighted                       |
| R19 | mAP across all queries              | ✅      | `mean_average_precision()`                           |
| R20 | Code deliverable                    | ✅      | `src/` — 20 files                                    |
| R21 | Report deliverable                  | ✅      | `tasks/report.md`                                    |
| R22 | Notebook deliverable                | ✅      | `tasks/phase1/phase1_search.ipynb`                   |

**All 22 required tasks: ✅ COMPLETE**

### 7.2 Extra Tasks Completed

| #   | Task                                                    | Location                                                  |
| --- | ------------------------------------------------------- | --------------------------------------------------------- |
| E1  | RRF fusion (5th strategy)                               | `src/retrieval/rrf.py`                                    |
| E2  | MRR metric                                              | `src/evaluation/metrics.py`                               |
| E3  | LM-Dir μ sweep (7 values, 5-fold CV)                    | `src/tuning/lmdir_mu_sweep.py`                            |
| E4  | BM25 k1/b grid sweep (20 configs, 5-fold CV)            | `src/tuning/bm25_param_sweep.py`                          |
| E5  | Encoder comparison (3 models)                           | `src/tuning/alt_encoder_eval.py`                          |
| E6  | 5-fold CV infrastructure                                | `src/tuning/cv_utils.py`                                  |
| E7  | Tuning visualization                                    | `src/tuning/tuning_plots.py`                              |
| E8  | Per-topic AP box plots                                  | `src/evaluation/plots.py` → `plot_per_topic_variance()`   |
| E9  | All 33 individual PR curves overlaid + 3 highlights     | `src/evaluation/plots.py` → `plot_individual_pr_curves()` |
| E10 | Train vs test generalization analysis                   | `tasks/report.md`                                         |
| E11 | Graded qrels (supporting=2, neutral=1)                  | `src/data/qrels_builder.py`                               |
| E12 | Query field ablation experiment                         | `src/evaluation/evaluator.py` → `field_ablation()`        |
| E13 | LM-JM lambda selection experiment                       | `src/evaluation/evaluator.py` → `lmjm_lambda_selection()` |
| E14 | Comprehensive notebook theory sections (LaTeX formulas) | `phase1_search.ipynb` — 19 markdown cells                 |
| E15 | Reference code index (Lab→project mapping)              | `references/README.md`                                    |
| E16 | PR curve interpretation reference plot                  | `src/tuning/tuning_plots.py` → `plot_pr_interpretation()` |

### 7.3 Suggested Tasks (Not Yet Done)

| #   | Task                                       | Priority | See §5   |
| --- | ------------------------------------------ | -------- | -------- |
| S1  | Run tuned LM-Dir (μ=75) on test set        | 🔴 HIGH   | §5.1 #1  |
| S2  | Run tuned BM25 (k1=1.5, b=1.0) on test set | 🔴 HIGH   | §5.1 #2  |
| S3  | Add MedCPT to index + test evaluation      | 🔴 HIGH   | §5.1 #3  |
| S4  | Update RRF with tuned components           | 🔴 HIGH   | §5.1 #4  |
| S5  | Remove emojis from notebook                | 🟡 MEDIUM | §5.1 #5  |
| S6  | Statistical significance tests             | 🟡 MEDIUM | §5.2 #6  |
| S7  | Error analysis on hard topics              | 🟡 MEDIUM | §5.2 #7  |
| S8  | Query length vs AP correlation             | 🟢 LOW    | §5.2 #8  |
| S9  | Document length analysis                   | 🟢 LOW    | §5.2 #9  |
| S10 | Strategy agreement / Jaccard overlap       | 🟢 LOW    | §5.2 #10 |

---

## 8. Verdict

### 8.1 Grade Assessment

**All 22 required tasks are complete.** The project exceeds requirements with 16 extra tasks, most notably:

- **5 retrieval strategies** (4 required + RRF fusion)
- **5 evaluation metrics** (P@10, R@100, NDCG required + MAP + MRR extra)
- **3 hyperparameter tuning experiments** with proper 5-fold CV (not required for Phase 1)
- **~1,846 lines of tuning code** with visualization
- **Comprehensive notebook** (39 cells) with theory, demos, tuning, final evaluation
- **Detailed report** (~500 lines) covering all findings

### 8.2 What Makes This a Grade 20 Project

1. **Scientific rigor:** Train/test discipline strictly followed. Hyperparameters locked on train before test evaluation. No data leakage.
2. **Deep understanding:** Every retrieval model explained with formulas, rationale for parameters, and interpretation of results. Not just "run and report numbers."
3. **Code quality:** Clean domain-centered structure, Lab reference patterns followed, comprehensive `__main__` tests in every file.
4. **Extra depth:** Tuning experiments show the student can design experiments, run cross-validation, interpret results, and draw actionable conclusions (MedCPT finding, μ correction).
5. **Report quality:** Timestamped, structured, includes limitations and caveats (noisy qrels, domain mismatch, small test set).

### 8.3 What Could Prevent Grade 20

1. **⚠️ Test results use untuned parameters** — The most visible gap. The tuning section shows clear improvements (+0.036 MAP for LM-Dir, +0.183 MAP for MedCPT) but the final test table still uses defaults. A professor would ask: "You found better parameters — why didn't you use them?" **Fix: items S1-S4 in §7.3.**
2. **⚠️ Notebook emojis** — Minor, but `_guidelines.md` is explicit. Easy fix.
3. **The test table shows BM25 and RRF essentially tied** (Δ=0.002 MAP). This is fine — it's a real finding — but could benefit from a significance test to make the claim rigorous.

### 8.4 Bottom Line

**Current state: strong 18–19.** The foundation is excellent. All requirements met, substantial extra work, proper methodology.

**To reach 20:** Apply the tuned parameters to the test set (S1-S4). This closes the gap between "we found better parameters" and "we validated them on test." This is the expected scientific workflow and its absence is the most notable gap.

---

*End of audit. All file paths verified against workspace. All metric values cross-checked against `tasks/report.md`, `tasks/phase1/PHASE_1_PLAN.md`, and `tasks/phase1/phase1_search.ipynb`.*