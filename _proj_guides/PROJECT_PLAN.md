# BioMedical NL Agents — Project Plan

**Course:** Natural Language Processing and Search 2025/2026  
**Based on:** TREC BioGen 2024/2025  
**Professors:** João Magalhães, David Semedo  
**Deadlines:** Phase 1 → **Apr 13** *(updated 2026-03-31)* | Phase 2 → May 4 | Phase 3 → Jun 1

---

## 1. Project Overview

This project builds a full biomedical information retrieval and generation pipeline in three phases:

1. **Phase 1 — Search & Evaluation:** Index PubMed abstracts in OpenSearch, implement 5 retrieval strategies (BM25, LM-JM, LM-Dir, Dense KNN, RRF fusion), evaluate with standard IR metrics.
2. **Phase 2 — RAG & LLM-Judges:** Cross-encoder reranking, LLM answer generation with citations, LLM-as-judge evaluation.
3. **Phase 3 — Deep Research Agent:** ReAct-style agentic loop — plan sub-topics, retrieve, aggregate evidence, write structured report.

---

## 2. Data

All data is small enough to keep in the repo under `data/`.

| File                                 | Description                                                                                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/BioGen2024topics.json`         | 65 query topics (id, topic, question, narrative) — IDs 116–180                                                                                     |
| `data/biogen_2024_submissions.json`  | Ground truth — reference answers from multiple systems per topic, with sentence-level citation assessments (supporting/contradicting/not relevant) |
| `data/filtered_pubmed_abstracts.txt` | 4194 PubMed abstracts in JSONL format (`id`, `contents`) — the document corpus                                                                     |

**Corpus document structure:** Each JSONL line has `id` (the PMID as a string) and `contents` (the full text — title + abstract concatenated). Index the `contents` field as-is, store `id` as `doc_id`. Do not split or modify the text — the concatenated form is already the correct indexing unit.

**Query split (fixed, never change):**
- Train: odd topic IDs (117, 119, 121, …)
- Test: even topic IDs (116, 118, 120, …)
- Split is on queries only — the corpus is shared and never split.
- Do this split once in `src/data/splitter.py` and save results. Never re-derive inline.

**Query fields:** Each topic has `topic`, `question`, and `narrative`. Start by querying with one field, then experiment with combining all three (e.g. concatenate into one string). The professor recommends this progression explicitly.

**Ground truth (`biogen_2024_submissions.json`) — important clarification:**

The professor's note says: *"The ground truth is the file which contains 'submission' in the name."* That is `biogen_2024_submissions.json`.

This file contains full system answers with sentence-level citation assessments. Relevance is derived from the `evidence_relation` field on each citation.

**Confirmed `evidence_relation` values from actual data:** `"supporting"` (10240), `"neutral"` (1412), `"not relevant"` (2201), `"contradicting"` (275), `"invalid citation"` (635).

**Graded relevance (required by new project guide, 2026-03-31):**
The new guide explicitly states: *"numerical values must be assigned to the labels to enable the calculation of nDCG."*
- Mapping (confirmed by professor's notes): `"supporting"` → 2, `"neutral"` → 1, all others → 0
- Binary threshold for P@k (use score ≥ 1 as relevant)
- **Two qrels files:**
  - `results/qrels.json` — binary (supporting=1, others=0) — kept for backward compat
  - `results/qrels_graded.json` — graded (supporting=2, neutral=1, others=0) — for nDCG

**Known limitation to mention in the report:** These are not human relevance judgments — they are citation assessments from automated systems. A PMID cited as "supporting" may only weakly address the query. This makes qrels noisier than standard TREC judgments. Acceptable for a course project, but noted explicitly in the experimental setup section.

This derivation is done once in `src/data/qrels_builder.py` and saved. Never re-derive inline.

---

## 3. Architecture

**Option A — Monorepo with shared `src/`** (chosen).

Phase 3 is architecturally defined as Phase 1 + Phase 2 running in a loop per sub-topic. Shared `src/` is the only sensible choice — the agent calls `retrieval.search(subtopic)` and `generation.generate(docs, subtopic)` directly. Any "independent phases" structure forces cross-directory imports anyway and creates duplication of OpenSearch client, embeddings, and evaluation code.

Notebooks are the **report artifact** — they run the pipeline end-to-end and show all outputs inline. Python files in `src/` are where all logic lives. Pattern from the DL project: each notebook detects whether it is running in Colab or locally, clones/pulls the repo if in Colab, and adapts paths accordingly.

---

## 4. Repository Structure

```
nlp-biomedical-agent/
│
├── _proj_guides/                    ← course guides, notes, this plan (not shipped)
│
├── src/
│   ├── data/
│   │   ├── loader.py                # load topics JSON, corpus JSONL
│   │   ├── splitter.py              # odd/even query split, saves train/test JSON
│   │   └── qrels_builder.py         # build binary qrels from biogen_2024_submissions.json
│   │
│   ├── indexing/
│   │   ├── opensearch_client.py     # connect to api.novasearch.org, health check, index-exists
│   │   ├── index_builder.py         # full index mapping: BM25 + LM-JM(λ=0.1) + LM-JM(λ=0.7) + LM-Dir + KNN (one index)
│   │   └── document_indexer.py      # bulk index with tqdm, idempotent (checks doc count first)
│   │
│   ├── retrieval/
│   │   ├── base.py                  # BaseRetriever interface
│   │   ├── bm25.py                  # match query on contents field (BM25 similarity)
│   │   ├── lm_jelinek_mercer.py     # match query on contents_lmjm_01 or _07 field (tune on train)
│   │   ├── lm_dirichlet.py          # match query on contents_lmdir field (mu=2000)
│   │   └── knn.py                   # encode query → knn search on embedding field
│   │
│   ├── embeddings/
│   │   ├── encoder.py               # sentence-transformer wrapper (msmarco-distilbert-base-v2)
│   │   └── corpus_encoder.py        # encode all docs once, save/load .npy
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # P@k, R@k, AP, MAP, MRR, PR curve — implemented in Python
│   │   └── plots.py                 # PR curves, comparison table (matplotlib)
│   │
│   ├── reranking/
│   │   └── cross_encoder.py         # cross-encoder reranker, returns top-k sentences per doc
│   │
│   ├── generation/
│   │   ├── answer_generator.py      # prompt builder + LLM call + citation injection
│   │   └── answer_validator.py      # enforce: ≤2500 words, ≤3 PMIDs/sentence, valid PMIDs only
│   │
│   ├── judge/
│   │   ├── prompts.py               # sentence-alignment + entailment judge prompt templates
│   │   └── llm_judge.py             # GPT-4o via IAedu API (not vLLM), parse structured output
│   │
│   └── agent/                       # Phase 3 only
│       ├── planner.py               # topic → list of sub-topics via LLM
│       ├── explorer.py              # ReAct loop: reason → SEARCH → observe (per sub-topic)
│       ├── aggregator.py            # merge evidence across sub-topics, dedup + cite-check
│       └── report_writer.py         # final structured report with citations
│
├── phase1_search.ipynb              ← Phase 1 pipeline + report
├── phase2_rag.ipynb                 ← Phase 2 pipeline + embedding viz exercise
├── phase3_agent.ipynb               ← Phase 3 agent demo + report
│
├── data/                            ← committed (small, ~few MB total)
│   ├── BioGen2024topics.json
│   ├── biogen_2024_submissions.json
│   └── filtered_pubmed_abstracts.txt
│
├── results/                         ← committed to git (JSON files are tiny, a few KB each)
│   ├── splits/
│   │   ├── train_queries.json
│   │   └── test_queries.json
│   ├── qrels.json                   ← binary relevance derived from submissions
│   ├── phase1/
│   │   ├── bm25_run.json
│   │   ├── lmjm_run.json            ← best LM-JM lambda variant
│   │   ├── lmdir_run.json
│   │   ├── knn_run.json
│   │   └── rrf_run.json             ← RRF fusion of BM25 + KNN
│   ├── phase2/
│   │   ├── reranked_run.json
│   │   ├── generated_answers.json
│   │   └── judge_labels.json
│   └── phase3/
│       └── agent_reports.json
│
├── embeddings/                      ← gitignored, heavy files
│   └── pubmed_knn_vectors.npy
│
├── configs/
│   └── opensearch.yaml              # index name, host (no credentials here)
│
├── .env                             ← gitignored — credentials only
├── .env.example                     ← committed — template with key names, no values
├── requirements.txt
└── README.md
```

---

## 5. Environment & Infrastructure

**Python env:** `cnn` (Python 3.10.19 via conda) — the same environment used in the DL project, no new environment needed.

**OpenSearch server:** Shared course instance at `api.novasearch.org:443` with `url_prefix='opensearch_v3'`. Credentials in `.env`. The corpus is small (~4194 docs) — indexing is fast. Each student has their own index (index name = username). One index, multiple similarity fields configured at creation time — not one index per strategy.

**LLM server (Phase 2+):** vLLM at `amalia.novasearch.org` via OpenAI-compatible API (`openai` Python client, just with a different `base_url`). API key provided by the course.

**Colab:** Notebooks detect `IN_COLAB` via `"google.colab" in sys.modules`. When in Colab: clone/pull the GitHub repo, install requirements, load credentials. When local: auto-reload `src/`, load `.env` directly. This is the same pattern used in the DL project.

**Google Drive (optional fallback):** Only needed if re-encoding embeddings in Colab. Can save/load `pubmed_knn_vectors.npy` from Drive to avoid re-running the encoder every session. Not required if running locally.

**`.env` keys:**
```
OPENSEARCH_HOST=api.novasearch.org
OPENSEARCH_PORT=443
OPENSEARCH_USER=
OPENSEARCH_PASS=
OPENSEARCH_INDEX=          # = your username, as per Lab 01
VLLM_BASE_URL=https://amalia.novasearch.org/vlm/v1
VLLM_API_KEY=amalia012026
VLLM_MODEL=               # query the server at startup to get available models
```

---

## 6. Notebook Pattern

Every notebook detects the environment and adapts. Based on the DL project pattern:

```python
# ── 0. SETUP ──────────────────────────────────────────────────────────
import sys, os
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    if not os.path.exists("/content/nlp-biomedical-agent"):
        !git clone https://github.com/fmssilva/nlp-biomedical-agent.git /content/nlp-biomedical-agent
    else:
        !git -C /content/nlp-biomedical-agent pull --ff-only
    os.chdir("/content/nlp-biomedical-agent")
    %pip install -r requirements.txt -q
else:
    %load_ext autoreload
    %autoreload 2

ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Load credentials ──────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # loads .env locally; in Colab, set env vars manually or from Drive

# ── Connect to OpenSearch ─────────────────────────────────────────────
from src.indexing.opensearch_client import get_client, check_index
client = get_client()
check_index(client)   # fails loudly if index does not exist or is empty
```

Markdown cells explain theory and decisions. Code cells run the pipeline. Outputs (tables, plots) inline. Anyone opening the notebook sees what happened and why.

---

## 7. Phase 1 — Search & Evaluation ✅ COMPLETE (2026-03-30)

**Goal:** Build the retrieval testbed and evaluate 5 strategies.

### 7.1 OpenSearch Index ✅

One index, multiple fields with different similarity methods. This is explicitly confirmed by the professor: **one index, not one per strategy.**

**Index structure (confirmed from Lab 01 pattern):**
```python
index_body = {
    "settings": {
        "index": {
            "number_of_shards": 4,
            "number_of_replicas": 0,
            "refresh_interval": "-1",
            "knn": "true"
        },
        "similarity": {
            "lmjm_01_similarity": {"type": "LMJelinekMercer", "lambda": 0.1},
            "lmjm_07_similarity": {"type": "LMJelinekMercer", "lambda": 0.7},
            "lmdir_similarity":   {"type": "LMDirichlet",     "mu": 2000}
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "doc_id":           {"type": "keyword"},
            "contents":         {"type": "text", "analyzer": "standard", "similarity": "BM25"},
            "contents_lmjm_01": {"type": "text", "analyzer": "standard", "similarity": "lmjm_01_similarity"},
            "contents_lmjm_07": {"type": "text", "analyzer": "standard", "similarity": "lmjm_07_similarity"},
            "contents_lmdir":   {"type": "text", "analyzer": "standard", "similarity": "lmdir_similarity"},
            "embedding":        {
                "type": "knn_vector", "dimension": 768,
                "method": {"name": "hnsw", "space_type": "innerproduct",
                           "engine": "faiss", "parameters": {"ef_construction": 256, "m": 48}}
            }
        }
    }
}
```

**Critical:** KNN field dimension cannot be changed after indexing. Fix the encoder model first, then create the index.

The `standard` analyzer (Lab01 pattern) does tokenization and lowercasing — no stemming. This is correct for biomedical text where stemming can mangle medical terms like "hepatitis" → "hepat". The `english` analyzer applies Porter stemming — avoid it for this corpus.

Indexing is idempotent: `document_indexer.py` checks doc count before inserting.

> ✅ **Done (2026-03-30):** Index `usernlp03` created with correct mapping. All 6 fields + 3 similarities confirmed live. 4194 docs indexed at ~198 docs/sec (~21s). Idempotency verified. `standard` analyzer used throughout.

### 7.2 Retrieval Strategies ✅

We bake two LM-JM variants into the index at creation time (λ=0.1 for short queries, λ=0.7 for long queries) as separate fields. This avoids any re-indexing for tuning — the corpus is only 4194 docs so the extra storage is trivial. The winning λ is selected on the train set.

| #   | Strategy          | Query approach                                                                          |
| --- | ----------------- | --------------------------------------------------------------------------------------- |
| 1   | BM25              | `match` on `contents` field (default BM25 similarity)                                   |
| 2   | LM Jelinek-Mercer | `match` on `contents_lmjm_01` (λ=0.1) vs `contents_lmjm_07` (λ=0.7) — tune on train set |
| 3   | LM Dirichlet      | `match` on `contents_lmdir` field, μ=2000 (standard default, tune if time allows)       |
| 4   | Dense KNN         | Encode query → `knn` query on `embedding` field                                         |
| 5   | RRF Fusion        | Reciprocal Rank Fusion of BM25 + KNN run files (k=60, Cormack 2009)                     |

**Important — `size` parameter:** OpenSearch returns 10 results by default. Every retriever must explicitly set `size=100` (or more) in the query body. Forgetting this means Recall@k for k>10 is always 0.

Each retriever returns a ranked list of `(pmid, score)` pairs. Save each run as a JSON dict: `{topic_id: [(pmid, score), ...]}`.

**BaseRetriever interface** (`src/retrieval/base.py`): all five retrievers must implement:
```python
def search(self, query: str, size: int = 100) -> list[tuple[str, float]]:
    """Returns ranked list of (pmid, score) pairs, highest score first."""
```

Hyperparameter tuning is done on **train queries only**. Fix the best params before evaluating on test.

**`size=100` is sufficient for this corpus** (4194 docs, biomedical topics) — no pagination needed. The full ranked list for PR curves fits within a single OpenSearch response at size=100.

> ✅ **Done (2026-03-30):** All 5 retrievers implemented in `src/retrieval/`. Each tested with query "obstructive sleep apnea treatment" — 100 results, descending scores, no duplicate PMIDs confirmed. **Locked hyperparams (train set):** query field = `concatenated`, LM-JM λ = 0.7.
>
> **Test set results (33 queries):** BM25 MAP=0.5685 (best MAP), LM-JM MRR=0.8409 (best MRR), RRF P@10=0.6364 (best P@10).

**Optional — LLM-based query expansion (add if Phase 1 is complete early):** Before sending the query to OpenSearch, call the vLLM server to expand with synonyms and related medical terms (e.g. "sleep apnea" → "sleep apnea obstructive sleep apnea OSA CPAP upper airway"). This is ~10 lines in a wrapper retriever and makes a strong analysis point in the report — original vs expanded queries. It directly connects Phase 1 retrieval with the LLM tools from Phase 2.

### 7.3 Encoder Model for Dense Retrieval ✅

**Use `sentence-transformers/msmarco-distilbert-base-v2`** — this is the model used in Lab 01. It produces 768-dimensional embeddings with mean pooling + L2 normalization, trained on MS MARCO passage retrieval with contrastive loss. Because embeddings are L2-normalized, inner product equals cosine similarity — so `space_type: innerproduct` is correct.

Lab01 implements the encoder directly via `transformers` (`AutoTokenizer` + `AutoModel` + `mean_pooling` function + `F.normalize`), not via the `sentence-transformers` high-level API. Follow that pattern — it's explicit, matches the lab exactly, and avoids hidden config differences.

Encode all corpus documents once offline, save to `embeddings/pubmed_knn_vectors.npy`. Load and index into OpenSearch KNN field. For query time, encode the query on the fly with the same model and same pooling/normalization logic.

**Note on domain-specific models:** MedCPT (`ncbi/MedCPT-Query-Encoder`) is a biomedical bi-encoder that may give better results. It is a legitimate experiment/comparison if time allows, but is not required. Primary model to get working: `msmarco-distilbert-base-v2`.

> ✅ **Done (2026-03-30):** `src/embeddings/encoder.py` + `corpus_encoder.py`. Lab01 pattern followed exactly (`AutoTokenizer` + `AutoModel`, mean pooling, L2 norm). 4194 docs encoded in ~3 min on CPU (batch=32). Output shape (4194, 768), all L2 norms = 1.0. Saved to `embeddings/pubmed_knn_vectors.npy` (12.3 MB, gitignored). Semantic check passed (biomedical similarity >> random text similarity).

### 7.4 Evaluation Metrics ✅

**Implement metrics in Python (numpy/matplotlib)** — exactly what Lab 03 demonstrates. No `ranx` dependency.

Required metrics:
- **P@k** — fraction of top-k results that are relevant
- **R@k** — fraction of all relevant docs found in top-k
- **AP** (Average Precision) — mean of P@k at each relevant doc's rank
- **MAP** — AP averaged across all queries
- **MRR** — average of 1/rank_of_first_relevant_doc
- **PR curves** — 11-point interpolated, averaged across queries (Lab03 pattern)

Report a comparison table across all strategies on the test set. Plot mean PR curves for all strategies on one chart (with shaded areas for variance, Lab03 style).

Evaluation format: for each query, a ranked list of PMIDs + a set of relevant PMIDs from qrels. Use the same `results_to_ranking()` pattern from Lab 03 to convert OpenSearch results.

> ✅ **Done (2026-03-30):** `src/evaluation/metrics.py` — all 10 functions implemented in plain Python/numpy. Lab03 toy example reproduced exactly (AP(A)=1.0, AP(B)=0.7095, 11-point PR curve). TREC-standard MAP/MRR excludes 0-relevant topics (with WARNING log). `src/evaluation/plots.py` — 4 plot functions, all PNG-verified. `results/phase1/pr_curves.png` and `ap_boxplot.png` saved.

### 7.5 Phase 1 Tests ✅

```
- OpenSearch: reachable, index exists, doc count == 4194                           ✅
- Each strategy: returns 100 results (not 10!), scores decrease monotonically,    ✅
  no duplicate PMIDs
- Evaluation: P@10, MAP, MRR all compute without error for all queries;            ✅
  PR curve has correct shape
- Split: train has ~33 queries (odd IDs), test has ~32 queries (even IDs),         ✅
  no overlap
- qrels: every topic in the split has at least one relevant PMID in qrels          ✅
```

---

## 8. Phase 2 — RAG & LLM-Judges

**Goal:** Improve retrieval with reranking, generate grounded answers, evaluate generation quality.

### 8.1 Cross-Encoder Reranking

Take top-N results from the best Phase 1 retriever. Re-score each (query, doc) pair with a cross-encoder — it jointly encodes query + document and produces a fine-grained relevance score.

**Confirmed model options from new guide (2026-03-31):** `MedCPT` (preferred — biomedical domain) or `BioBERT`. As a fallback, general-purpose `cross-encoder/ms-marco-MiniLM-L-6-v2` can be used.

**Recommended primary model:** `ncbi/MedCPT-Cross-Encoder` — trained on PubMed click data, same domain as our task. The encoder comparison in Phase 1 tuning showed MedCPT gives +0.183 MAP over ms-marco on biomedical retrieval — expect similar advantage for reranking.

Also extract the **top-3 most relevant sentences per document** (by cross-encoder score) — these become the context for the LLM generator, not the full abstract.

### 8.2 Answer Generation

**LLM:** vLLM server at `amalia.novasearch.org` via `openai` Python client (same pattern as Lab 02). Query the server at startup to get the available model.

Hard constraints from the assignment:
- Answer **≤ 2500 words** *(updated 2026-03-31 — was 250, confirmed in new project guide)*
- ≤ 3 PMIDs cited per sentence
- All cited PMIDs must come from the retrieved valid set

`answer_validator.py` checks all three. Build this before the generation loop.

### 8.3 LLM-as-Judge

Two judge tasks (per TREC BioGen methodology):
1. **Sentence relevance:** does this answer sentence address the query?
2. **Citation entailment:** does the cited PMID's text support the answer sentence?

**LLM for judge:** **GPT-4o via IAedu API** (`https://iaedu.pt/pt`) — *confirmed in new project guide (2026-03-31)*. NOT the vLLM server at amalia.novasearch.org (that server is for generation only). Use the `openai` Python client pointing to the IAedu endpoint.

Use structured output (JSON) — same pattern as Lab 02 JSON mode. The lab shows how to use `response_format={"type": "json_object"}`. Calibrate prompts on 5-10 manual examples first.

**IAedu API key:** obtain from `https://iaedu.pt/pt` — set in `.env` as `IAEDU_API_KEY` and `IAEDU_BASE_URL`.

### 8.4 BERT Contextual Embedding & Attention Visualization (Required Exercise)

The course requires a **specific positional embedding exercise** (confirmed in new guide, 2026-03-31):

**Required exercise (from new project guide, verbatim):**
1. Insert a sequence of text with the **same word repeated 200 times**
2. Visualize the embeddings: compute the **distance of all tokens to the first token**
3. Plot in **2D** — shows how positional encodings drift as position increases
4. Also produce a **distance matrix** (pairwise token distances) — shows the positional structure

**Why this matters:** BERT uses absolute positional embeddings (learned, not fixed). When the same word repeats 200 times, the meaning is identical — any variation in token distances is purely due to positional encoding. This demonstrates contextuality and positional awareness simultaneously.

**Implementation sketch:**
```python
text = "word " * 200  # same word repeated 200 times
tokens = tokenizer(text, return_tensors="pt", truncation=False)
with torch.no_grad():
    output = model(**tokens, output_hidden_states=True)
# last layer embeddings: shape (1, seq_len, 768)
embeddings = output.last_hidden_state[0]  # (seq_len, 768)
# distance from first token to all others
dists = torch.cdist(embeddings[0:1], embeddings).squeeze()
# also: full pairwise distance matrix
dist_matrix = torch.cdist(embeddings, embeddings)
# plot dists as line plot + dist_matrix as heatmap
```

**Secondary exercise (also required in guide):** Take a sentence with a word used in different senses (e.g. "bank"). Show that the same word has different vector representations depending on context — this is contextuality. Extract token embeddings from all 12 layers, compare.

This goes in `phase2_rag.ipynb` as a dedicated section with markdown explanation + code + figures.

### 8.5 Phase 2 Tests

```
- Reranker: returns scores for all candidate docs, no crashes on max-length abstracts
- Generator: output ≤2500 words, ≤3 PMIDs/sentence, all cited PMIDs in corpus
- Judge: output is parseable JSON, labels in expected set (GPT-4o via IAedu)
- Validator: catches a known violation in a hand-crafted bad example
```

---

## 9. Phase 3 — Deep Research Agent

**Goal:** Agentic pipeline that plans, retrieves, aggregates, and writes a structured report.

### 9.1 Architecture

```
User Topic
    ↓
planner.py      → LLM decomposes topic into N sub-topics
    ↓ (for each sub-topic)
explorer.py     → ReAct loop:
                    reason: "I need to find evidence about X"
                    SEARCH: "query string"
                    observe: [retrieved docs]
                    ... (repeat until enough evidence or max steps)
    ↓
aggregator.py   → merge evidence across sub-topics, deduplicate PMIDs, cite-check
    ↓
report_writer.py → structured final report with section headers + inline citations
```

### 9.2 ReAct Loop Design

Use a strict structured format for LLM actions (not free text):

```xml
<reason>I need to find information about weight loss interventions for sleep apnea.</reason>
<action>SEARCH</action>
<query>weight loss treatment obstructive sleep apnea</query>
```

Parse the XML tags. Any output that doesn't match = logged error + retry or skip.

Hard cap: max N iterations per sub-topic (e.g. 5–8). Log warning if cap is hit.

### 9.3 LLM for Agent

Same vLLM server as Phase 2 (`amalia.novasearch.org`). The planner, explorer reasoning steps, and report writer all call this.

### 9.4 Phase 3 Tests

```
- Planner: returns ≥1 sub-topic, output is valid list
- Explorer: terminates within max iterations, returns non-empty evidence
- Aggregator: no duplicate PMIDs, all PMIDs exist in corpus
- Report: has ≥1 section, all citations valid
- End-to-end: full run on one topic completes without crash
```

---

## 10. Cross-Cutting Rules

### Code Quality

- Small files, single clear responsibility. Name everything self-documenting.
- No `utils.py` dumping ground. If a file grows beyond one concern, split it.
- Only implement what is currently needed. No speculative code.
- One-line comment above each function. Inline comments in natural language — casual, one dev to another. No emojis anywhere (terminal encoding issues).
- Logs: only what's needed to pinpoint errors. Single-line, concise.

### Before Implementing Any Task

1. Read all relevant existing files — avoid duplication.
2. Think 3 options at architecture level (where/how it fits).
3. Think 3 options at implementation level (how to write it).
4. Choose the cleanest, simplest, most maintainable option.

### Testing During Development

Every `src/` file has an `if __name__ == "__main__"` block with a quick sanity test. Run locally first on a small subset. Only move to Colab once local tests pass.

### OpenSearch Discipline

- Always check index exists before querying — fail loudly with a clear error if not.
- Indexing is one-time and idempotent — check doc count before bulk insert.
- Never delete the shared index without team agreement.
- Index mapping must be defined before any documents are inserted. Cannot change field types or KNN dimension after the fact.

### Workflow

```
implement → local test (CPU, small data) → fix → notebook (Colab if GPU needed)
```

Never update the notebook with new code until the `.py` files are tested and working.

### Notebook Startup (every notebook)

1. Detect `IN_COLAB` — if true: clone/pull repo, `pip install -r requirements.txt`
2. If local: `%load_ext autoreload` + `%autoreload 2`
3. Load `.env` credentials
4. Connect to OpenSearch — confirm health + index doc count
5. Only then does the actual work begin

---

## 11. What to Watch Out For (Known Pitfalls)

| Issue                             | Correct approach                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| One index per strategy            | One index, multiple similarity fields configured at creation                                                       |
| Splitting documents               | Split queries only (odd/even), corpus is shared                                                                    |
| Tuning on test set                | Hyperparams fixed on train; test is final evaluation only                                                          |
| Re-indexing accidentally          | Idempotency check in `document_indexer.py` — abort if docs already there                                           |
| Changing KNN field after indexing | KNN vector dimension is immutable after index creation                                                             |
| Wrong index name                  | Lab 01 confirms: index name = your username                                                                        |
| `size=10` default in queries      | Always set `size=100` (or more) explicitly — default returns only 10 results, making Recall@k for k>10 always 0    |
| LM-JM lambda tuning               | Bake λ=0.1 and λ=0.7 as two separate fields at index creation time — pick winner on train set, no re-indexing      |
| Free-text LLM action parsing      | Use strict XML/JSON format for ReAct actions                                                                       |
| Agent looping forever             | Hard iteration cap per sub-topic, logged as warning                                                                |
| Implementing metrics with ranx    | Metrics are implemented in plain Python/numpy (Lab 03 pattern) — no ranx dependency                                |
| `english` analyzer for biomedical | Use `standard` analyzer (Lab01 pattern) — `english` applies stemming that mangles medical terms                    |
| Citation hallucinations           | `answer_validator.py` checks every cited PMID against the corpus                                                   |
| Answer length limit               | ≤2500 words *(2026-03-31: guide says 2500, earlier draft said 250 — the guide is authoritative)*                   |
| LLM judge vs generator            | Judge = **GPT-4o via IAedu** (`iaedu.pt`); Generator = vLLM at amalia.novasearch.org — different endpoints         |
| Binary vs graded qrels            | `qrels.json` = binary (MAP/P@k); `qrels_graded.json` = graded (nDCG) — both from `qrels_builder.py`                |
| Team run file sharing             | Commit small run JSON files to `results/` in git — they're a few KB each. Git history serves as the comparison log |

---

## 12. Guidelines Update (from DL Project)

The existing `_guidelines.md` is almost entirely applicable. Keep all of: code quality rules, comment style, file structure philosophy, testing discipline, "think 3 options" rule, and the `if __name__ == "__main__"` testing pattern.

**Changes:**
- Replace `PyTorch/sklearn` references → `opensearch-py, sentence-transformers, transformers`
- Replace `Colab+GPU workflow` → keep it but adapt: indexing is fast on CPU; GPU helps for encoding (~4194 docs) and cross-encoder reranking
- Keep `cnn (3.10.19)` conda env — same environment as DL project, no new env needed
- Replace DL testing checklist → IR/NLP testing checklist (see Section 10)
- `PowerShell: use ; not &&` → still valid for local Windows use

**Add:**
- OpenSearch discipline rules (Section 10)
- Notebook `IN_COLAB` detection pattern (Section 6)
- Rule: `results/` JSON run files **are committed to git** (tiny, a few KB each — git history is the comparison log). Only `embeddings/` is gitignored (large .npy files, regeneratable by running corpus_encoder.py).

---

## 13. Implementation Order

### Phase 1 (due Apr 13) — ✅ ALL COMPLETE (2026-03-30), graded qrels + NDCG + R@100 + individual PR curves added (2026-03-31)

1. ✅ `src/data/loader.py` — load corpus JSONL, topics JSON
2. ✅ `src/data/splitter.py` — odd/even split, save to `results/splits/`
3. ✅ `src/data/qrels_builder.py` — binary + graded qrels from submissions JSON
4. ✅ `src/indexing/opensearch_client.py` — connect to `api.novasearch.org`, health check
5. ✅ `src/indexing/index_builder.py` — full mapping: BM25 + LM-JM + LM-Dir + KNN fields
6. ✅ `src/indexing/document_indexer.py` — bulk index with tqdm, idempotent
7. ✅ `src/embeddings/encoder.py` + `corpus_encoder.py` — encode docs with `msmarco-distilbert-base-v2`, save .npy
8. ✅ `src/retrieval/base.py` + five strategy files (`bm25.py`, `lm_jelinek_mercer.py`, `lm_dirichlet.py`, `knn.py`, `rrf.py`)
9. ✅ `src/evaluation/metrics.py` — P@k, R@k, AP, MAP, MRR, NDCG (plain Python/numpy, Lab 03 pattern)
10. ✅ `src/evaluation/plots.py` — PR curves, comparison table, individual per-query PR curves
11. ✅ `phase1_search.ipynb` — end-to-end pipeline + results + PR curves

### Phase 2 (due May 4)

1. `src/reranking/cross_encoder.py`
2. `src/generation/answer_generator.py` + `answer_validator.py`
3. `src/judge/prompts.py` + `llm_judge.py`
4. `phase2_rag.ipynb` — reranking + generation + judge + embedding viz

### Phase 3 (due Jun 1)

1. `src/agent/planner.py`
2. `src/agent/explorer.py` — ReAct loop
3. `src/agent/aggregator.py`
4. `src/agent/report_writer.py`
5. `phase3_agent.ipynb` — full agent demo

---

## 14. Key Technology Stack

| Component       | Library/Tool                                                                  | Notes                                                        |
| --------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Search engine   | `opensearch-py` → `api.novasearch.org:443`                                    | `url_prefix='opensearch_v3'`, SSL, shared instance           |
| Dense retrieval | `sentence-transformers` → `msmarco-distilbert-base-v2`                        | 768-dim, mean pooling (Lab01 pattern), L2 norm, innerproduct |
| LM similarities | two LM-JM fields (λ=0.1, λ=0.7) + LM-Dirichlet                                | baked into index at creation, no re-indexing for tuning      |
| IR evaluation   | Plain Python + numpy + matplotlib                                             | P@k, R@k, AP, MAP, MRR, PR curves — Lab 03 pattern           |
| LLM (Phases 2+) | `openai` client → `amalia.novasearch.org/vlm/v1`                              | vLLM, OpenAI-compatible API, Lab 02 pattern (generation)     |
| LLM Judge       | `openai` client → **IAedu API** (`https://iaedu.pt/pt`) → GPT-4o              | GPT-4o for judge — confirmed in new guide (2026-03-31)       |
| Cross-encoder   | `sentence-transformers` cross-encoder                                         | e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`                  |
| Embedding viz   | BERT layers + attention (required) + `sklearn` TSNE / `umap-learn` (optional) | Phase 2 exercise                                             |
| Env management  | `python-dotenv`                                                               | Load `.env` credentials                                      |
| Progress bars   | `tqdm`                                                                        | Indexing, encoding, generation loops                         |
| Python env      | `cnn (3.10.19)` conda                                                         | Same as DL project, no new env needed                        |

