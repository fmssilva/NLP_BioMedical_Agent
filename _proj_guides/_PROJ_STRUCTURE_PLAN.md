# Project Structure Plan — BioMedical NLP Agent
### Full structure for all 3 phases

---

## Design Philosophy

**Guiding rule for notebook vs src split:**
- `src/` = anything that benefits from being named and reused (loaded by name → reader
  understands what it does). Functions with clear single-responsibility names that read
  almost like prose in the notebook: `split_sentences(abstract)`, `score_sentences(query, sentences, model)`,
  `generate_answer(question, context, client)`, `judge_entailment(answer, refs, client)`.
- **Notebook** = the wiring layer + everything where seeing the full code is *more* illuminating
  than a function name. Examples: prompt templates (always show the full text), simple display
  helpers, one-off visualisation loops, manual calibration cells.

**Rule for encapsulation depth:**
- Max 2 layers: notebook → `src/module.function()`. No nested helper chains.
- Prefer a few more concrete functions over a complex multi-flag multipurpose one.
  `format_context_for_generation` and `format_context_for_judge` are clearer
  than `format_context(mode="generation"|"judge")`.
- Only extract to `src/` when the same logic is called from ≥2 places (DRY), or when it
  would make the notebook cell too long (> ~25 lines of non-trivial logic).

---

## Status Legend
```
[keep]     existing, well done — leave as is
[keep+]    existing, minor cleanup welcome (rename, docstring) — do NOT change signatures
[new-P2]   create for Phase 2
[new-P3]   create for Phase 3 (plan now, implement later)
```

---

## Full Tree

```
NLP_Biomedical_Agent/
│
├── tasks/                          # one notebook per phase (the "cover")
│   ├── phase1_search.ipynb         [keep]  Phase 1 complete
│   ├── phase2_rag.ipynb            [new-P2] Phase 2 notebook
│   └── phase3_agent.ipynb          [new-P3] Phase 3 notebook
│
├── src/                            # all reusable logic (no notebooks here)
│   ├── __init__.py                 [keep]
│   │
│   ├── data/                       # loading and transforming raw data
│   │   ├── __init__.py             [keep]
│   │   ├── loader.py               [keep]   load_corpus(), load_topics()
│   │   │                           # NOTE: after loading, notebook builds:
│   │   │                           #   corpus_lookup = {doc["id"]: doc["contents"] for doc in corpus}
│   │   │                           #   valid_pmids   = {doc["id"] for doc in corpus}
│   │   │                           # These are built in the notebook setup cell, not in loader.py.
│   │   ├── query_builder.py        [keep]   build_query(topic, field)
│   │   ├── splitter.py             [keep]   run_splitter() -> train/test JSON files
│   │   ├── qrels_builder.py        [keep]   run_qrels_builder() -> qrels JSON files
│   │   ├── __loader_test.py        [keep]
│   │   └── __qrels_builder_test.py [keep]
│   │
│   ├── indexing/                   # OpenSearch index creation and document indexing
│   │   ├── __init__.py             [keep]
│   │   ├── opensearch_client.py    [keep]   get_client(), check_health(), check_index()
│   │   ├── index_builder.py        [keep]   create_or_update_index(), IndexSettings, float_tag()
│   │   ├── document_indexer.py     [keep]   index_documents()
│   │   └── __index_builder_test.py [keep]
│   │
│   ├── embeddings/                 # sentence encoding (bi-encoder for Phase 1 KNN)
│   │   ├── __init__.py             [keep]
│   │   ├── encoder.py              [keep]   Encoder class (singleton cache, pooling modes)
│   │   ├── corpus_encoder.py       [keep]   create_embeddings(), save/load .npy cache
│   │   └── __encoder_test.py       [keep]
│   │
│   ├── retrieval/                  # Phase 1 retrieval strategies
│   │   ├── __init__.py             [keep]
│   │   ├── base.py                 [keep]   BaseRetriever, SparseRetriever
│   │   ├── bm25.py                 [keep]   BM25Retriever
│   │   ├── lm_jelinek_mercer.py    [keep]   LMJMRetriever
│   │   ├── lm_dirichlet.py         [keep]   LMDirichletRetriever
│   │   ├── knn.py                  [keep]   KNNRetriever
│   │   ├── rrf.py                  [keep]   RRFRetriever
│   │   └── __retrieval_test.py     [keep]
│   │
│   ├── evaluation/                 # metrics + plots (shared across phases)
│   │   ├── __init__.py             [keep]
│   │   ├── metrics.py              [keep]   P@k, R@k, AP, MAP, MRR, NDCG, PR curves
│   │   ├── evaluator.py            [keep]   evaluate_retriever()
│   │   ├── plots.py                [keep]   plot_pr_comparison(), plot_metric_table(), ...
│   │   └── __final_eval.py         [keep]   final eval script (run once, save results)
│   │
│   ├── tuning/                     # hyperparameter sweep utilities (Phase 1)
│   │   ├── __init__.py             [keep]
│   │   ├── sweeper.py              [keep]   run_bm25_sweep(), run_lmdir_sweep(), ...
│   │   ├── cv_utils.py             [keep]   run_cv()
│   │   ├── tuning_plots.py         [keep]   plot_sweep(), plot_encoder_comparison(), ...
│   │   └── __sweeper_test.py       [keep]
│   │
│   ├── reranking/                  # Phase 2: cross-encoder sentence re-ranking [DONE]
│   │   ├── __init__.py             [DONE]
│   │   ├── cross_encoder.py        [DONE]  CrossEncoder class — score_pairs(), score_query_vs_sentences()
│   │   ├── sentence_splitter.py    [DONE]  split_sentences(), select_top_sentences()
│   │   └── __reranking_test.py     [DONE]  36 tests — all passing
│   │
│   ├── generation/                 # Phase 2: RAG answer generation
│   │   ├── __init__.py             [new-P2]
│   │   ├── llm_client.py           [new-P2] see details below
│   │   ├── context_builder.py      [new-P2] see details below
│   │   ├── answer_generator.py     [DONE] see details below
│   │   ├── answer_parser.py        [DONE] see details below
│   │   └── __generation_test.py    [new-P2] see details below
│   │
│   ├── judging/                    # Phase 2: LLM-as-a-judge evaluation
│   │   ├── __init__.py             [new-P2]
│   │   ├── llm_judge.py            [new-P2] see details below
│   │   └── __judging_test.py       [new-P2] see details below
│   │
│   ├── analysis/                   # Phase 2: transformer internals visualisation [DONE]
│   │   ├── __init__.py             [DONE]
│   │   ├── transformer_inspector.py [DONE]  get_hidden_states_and_attentions(), get_positional_embeddings(), cosine_distance_matrix()
│   │   ├── attention_plots.py      [DONE]  plot_positional_embedding_scatter(), plot_pairwise_distance_heatmap(), plot_contextual_embeddings_grid(), plot_attention_matrix()
│   │   └── __analysis_test.py      [DONE]  30 tests — all passing
│   │
│   └── agent/                      # Phase 3: ReAct agent
│       ├── __init__.py             [new-P3]
│       ├── planner.py              [new-P3] identify sub-topics from main query
│       ├── research_loop.py        [new-P3] execute Phase 1+2 cycle per sub-topic
│       ├── synthesiser.py          [new-P3] aggregate evidence into final report
│       └── __agent_test.py         [new-P3]
│
├── results/                        # outputs — small JSONs committed, large files gitignored
│   │                               # POLICY: JSON run files (KB-sized) → committed to git
│   │                               #         .npy embeddings (MB-sized) → gitignored
│   │                               #         figures (PNG/SVG) → committed to git
│   ├── qrels/                      [new-P2] qrels.json, qrels_graded.json  (moved from results/)
│   ├── splits/                     [new-P2] train_queries.json, test_queries.json
│   ├── embeddings/                 [keep]   *.npy per encoder alias
│   ├── phase1/                     [keep+]  all Phase 1 run JSONs + figures + tuning CSVs
│   │   ├── tuning/
│   │   └── figures/
│   ├── phase2/                     [new-P2] all Phase 2 outputs
│   │   ├── train_answers.json          [DONE] RAG answers for training queries
│   │   ├── test_answers.json           RAG answers for test queries
│   │   ├── sentence_alignment_train.json
│   │   ├── sentence_alignment_test.json
│   │   ├── entailment_train.json
│   │   ├── entailment_test.json
│   │   └── figures/
│   └── phase3/                     [new-P3]
│
├── data/                           [keep]   raw data files (gitignored corpus)
├── configs/                        [keep]   opensearch.yaml
├── report/                         [keep]   LaTeX report
├── references/                     [keep]   lab notebooks + README index
├── _proj_guides/                   [keep]   planning docs
├── requirements.txt                [keep+]  add Phase 2 deps: nltk, bertviz
├── .env                            [keep]   credentials (gitignored)
└── .env.example                    [keep]
```

---

## Detailed File Specifications

### `src/reranking/cross_encoder.py`  `[new-P2]`
```python
# Functions / class:

class CrossEncoder:
    """
    Load a HuggingFace cross-encoder model (AutoModelForSequenceClassification).
    Singleton cache keyed by model_name + device (same pattern as Encoder).
    """
    def __init__(self, model_name: str, device: str | None = None)
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]
        # pairs = [(query, sentence), ...] -> list of logit scores (float)
    def score_query_vs_sentences(self, query: str, sentences: list[str], batch_size: int = 16) -> list[float]
        # convenience: repeats query for each sentence, calls score_pairs
```
**Design note:** Two methods — `score_pairs` is the low-level workhorse (also useful for
Phase 3). `score_query_vs_sentences` is the high-level convenience used in the notebook.
No flags, no multi-purpose merging — clean and simple.

**IMPORTANT — output logit verification:** Some HuggingFace cross-encoders output a 2-class
logit tensor `[irrelevant, relevant]`, not a single scalar. For these, you need `logits[:, 1]`
(positive/relevant class), not `logits[:, 0]`. MedCPT-Cross-Encoder outputs a single logit
per pair, but add an assertion in the smoke test to verify: `assert output.shape[-1] == 1 or
(output.shape[-1] == 2 and "use logits[:, 1]")`. This is a common bug with HF cross-encoders.

---

### `src/reranking/sentence_splitter.py`  `[new-P2]`
```python
# Functions:

def split_sentences(text: str) -> list[str]
    # split abstract into sentences using nltk.sent_tokenize (punkt)
    # strip empty strings, strip leading/trailing whitespace
    # returns [] for empty/None input (never raises)
    # NOTE: NLTK >=3.9 uses 'punkt_tab'; download both 'punkt' and 'punkt_tab' to be safe

def select_top_sentences(
    query: str,
    abstract: str,
    cross_encoder: CrossEncoder,
    top_n: int = 3,
) -> list[dict]
    # splits abstract -> scores each sentence -> returns top_n
    # returns: [{"sentence": str, "score": float, "rank": int}, ...]
    # sorted by score descending
    # if abstract has < top_n sentences, returns whatever is available
    # if abstract has 0 sentences (empty/title-only), returns []
    # callers (bulk pipeline) should handle empty returns gracefully — skip silently
```
**Design note:** `select_top_sentences` is the function called from the notebook. It
combines splitting + scoring in one call — this is the right level for the notebook to call
(one meaningful action). Internally it calls `split_sentences` + `cross_encoder.score_query_vs_sentences`.

---

### `src/generation/llm_client.py`  `[new-P2]` [DONE]
```python
# Functions:

def get_gpt4o_client() -> openai.OpenAI
    # connect to IAedu GPT-4o — reads IAEDU_URL + IAEDU_KEY from env
    # primary client for both generation and judging

def get_vlm_client() -> openai.OpenAI
    # connect to amalia.novasearch.org vLLM — reads VLLM_URL + VLLM_KEY from env
    # secondary/optional generator for comparison experiments
    # prints available model list on first call

def get_vlm_model_name(client: openai.OpenAI) -> str
    # returns client.models.list().data[0].id (first available model)
```
**Design note:** Two separate clean getter functions, not a factory with a `backend=` flag.
The notebook cell reads: `gpt_client = get_gpt4o_client(); vlm_client = get_vlm_client()`.
GPT-4o is the primary LLM for both generation and judging (user preference, D4 resolved).

---

### `src/generation/context_builder.py`  `[new-P2]` [DONE]
```python
# Functions:

def build_context(
    selected: list[dict],       # [{"pmid": str, "sentence": str, "score": float}, ...]
    valid_pmids: set[str],       # corpus PMID set — warn if a PMID is outside
) -> str
    # formats: "[PMID 12345] Sentence text.\n[PMID 23456] Next sentence.\n..."
    # warns (does not raise) if a PMID is not in valid_pmids
    # returns empty string if selected is empty
```
**Design note:** One function, one purpose. The notebook call is:
`context = build_context(selected_sentences, valid_pmids)`.

---

### `src/generation/answer_generator.py`  `[DONE]`
```python
# Functions:

SYSTEM_PROMPT = "..."   # module-level constant — visible and editable at src level

def generate_answer(
    question:   str,
    context:    str,
    client:     openai.OpenAI,
    model_name: str,
    temperature: float = 0.1,
    max_tokens:  int   = 400,
) -> str
    # single chat.completions.create call
    # returns raw answer string (not parsed yet)
```
**Design note:** Prompt templates live here as module-level string constants (`SYSTEM_PROMPT`,
`USER_PROMPT_TEMPLATE`) — not buried inside the function. The notebook imports and
**displays** the prompt template in a markdown cell before calling `generate_answer`. This
way the reader sees the actual prompt text in the notebook, and the template lives in
one place in src.

---

### `src/generation/answer_parser.py`  `[DONE]`
```python
# Functions:

def parse_answer(raw: str, valid_pmids: set[str]) -> dict
    # extracts sentences + inline PMIDs from the raw answer string
    # returns:
    # {
    #   "text":         str,                        # full answer text (citations stripped)
    #   "sentences":    [{"text": str, "pmids": [str, ...]}],
    #   "word_count":   int,
    #   "all_pmids":    list[str],                  # flat, deduplicated
    #   "violations":   {
    #       "over_word_limit":        bool,          # word_count > 250
    #       "sentences_over_3_pmids": list[int],     # sentence indices that exceed 3 PMIDs
    #       "invalid_pmids":          list[str],     # cited PMIDs not in valid_pmids
    #   }
    # }

def check_constraints(parsed: dict) -> bool
    # returns True if no violations found (all constraint checks pass)
```
**Design note:** `parse_answer` returns a rich dict with the violations sub-dict already
computed — the notebook can display a clean pass/fail table without any extra logic.
`check_constraints` is a one-liner helper so the notebook can write `assert check_constraints(parsed)`.

---

### `src/judging/llm_judge.py`  `[new-P2]`
```python
# Module-level prompt constants (visible in notebook via import + display):
# Starting prompts are the EXACT TREC BioGen guide prompts (uncalibrated).
# After Section 4.4 calibration, any adaptations are documented in the notebook.
SENTENCE_ALIGNMENT_SYSTEM_PROMPT = "..."   # guide labels: Required/Unnecessary/Borderline/Inappropriate
SENTENCE_ALIGNMENT_USER_TEMPLATE = "..."
ENTAILMENT_SYSTEM_PROMPT         = "..."   # guide labels: Supported/Partially Supported/Unsupported
ENTAILMENT_USER_TEMPLATE         = "..."

# Functions:

def judge_sentence_alignment(
    query:    str,
    sentence: str,
    pmid:     str,
    client:   openai.OpenAI,
    model:    str = "gpt-4o",
) -> dict
    # returns {"label": str, "pmid": str}
    # label ∈ {Required, Unnecessary, Borderline, Inappropriate}
    # uses response_format={"type": "json_object"}
    # retries once on parse failure

def judge_answer_entailment(
    question:   str,
    answer:     str,
    references: list[str],
    client:     openai.OpenAI,
    model:      str = "gpt-4o",
) -> dict
    # returns {"label": str, "unsupported_claims": list[str]}
    # label ∈ {Supported, Partially Supported, Unsupported}

def batch_judge_alignment(
    query:      str,
    sentences:  list[dict],   # [{"pmid": str, "sentence": str}, ...]
    client:     openai.OpenAI,
    model:      str = "gpt-4o",
) -> list[dict]
    # convenience: calls judge_sentence_alignment for each sentence, returns list of results
```
**Design note:** The prompt templates are module-level constants (not hidden inside functions)
so the notebook can import and print them for the calibration cell. This is the same design
as `answer_generator.py` — prompts are always visible to the reader.
The starting prompts use the TREC BioGen guide labels verbatim. During calibration (§4.4),
if we decide to modify them (e.g., add JSON output format, add reasoning), we document the
changes in the notebook analysis cell.

---

### `src/analysis/transformer_inspector.py`  `[new-P2]`
```python
# Functions:

def get_hidden_states_and_attentions(
    text:      str,
    tokenizer,
    model,
) -> tuple[list[str], np.ndarray, np.ndarray]
    # returns (tokens, hidden_states, attentions)
    # tokens:        list[str]  len=seq_len (decoded from tokenizer)
    # hidden_states: np.ndarray (n_layers+1, seq_len, hidden_dim) — layer 0 = embedding layer
    # attentions:    np.ndarray (n_layers, n_heads, seq_len, seq_len)

def get_positional_embeddings(
    tokenizer,
    model,
    word: str  = "the",
    n:    int  = 200,
) -> tuple[list[str], np.ndarray]
    # builds word repeated n times, extracts layer-0 embeddings only
    # returns (tokens, embeddings) where embeddings.shape = (n_tokens, hidden_dim)

def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray
    # pairwise cosine distance, shape (N, N)
    # used for both: distance-from-token-0 vector and the full pairwise matrix

# if __name__ == "__main__" block for quick local testing:
#   - load bert-base-uncased, run get_positional_embeddings, print shape
#   - run get_hidden_states_and_attentions on a test sentence, print shapes
#   - run cosine_distance_matrix, print shape
#   (allows: python -m src.analysis.transformer_inspector)
```

---

### `src/analysis/attention_plots.py`  `[new-P2]`
```python
# Functions:

def plot_positional_embedding_scatter(
    embeddings: np.ndarray,
    save_path:  str | None = None,
) -> plt.Figure
    # PCA to 2D, scatter colored by distance from token 0 (Plot C in plan)

def plot_pairwise_distance_heatmap(
    distance_matrix: np.ndarray,
    save_path:       str | None = None,
) -> plt.Figure
    # seaborn heatmap of (N, N) cosine distance matrix (Plot D)

def plot_contextual_embeddings_grid(
    hidden_states: np.ndarray,
    tokens:        list[str],
    save_path:     str | None = None,
) -> plt.Figure
    # 12 PCA scatter subplots, one per layer (Plot E)

def plot_attention_matrix(
    attentions: np.ndarray,
    tokens:     list[str],
    layer:      int = 11,
    save_path:  str | None = None,
) -> plt.Figure
    # mean attention across heads for a single layer, heatmap (Plot F/G fallback)
```
**Design note:** `bertviz` calls (`head_view`, `model_view`) stay directly in the notebook —
they return HTML widgets and are better shown inline than hidden in a src function. The
matplotlib fallbacks above go in `src/analysis/` because they are reusable plot functions.

---

### `src/reranking/__reranking_test.py`  `[new-P2]`
Tests to run before any notebook cell in Section 1:
- model loads, outputs scalar per pair
- `score_pairs` returns list of floats, same length as input
- `score_query_vs_sentences` returns list sorted descending after `select_top_sentences`
- `split_sentences` on empty string returns `[]`, on abbreviation text doesn't over-split
- `select_top_sentences` returns exactly `top_n` items, sorted by score descending

### `src/generation/__generation_test.py`  `[new-P2]`
Tests to run before any notebook cell in Section 3:
- `build_context` formats correctly, warns on invalid PMID (does not raise)
- `parse_answer` word count correct, PMIDs extracted correctly, violations detected
- `check_constraints` returns True/False correctly
- `generate_answer` with mock/real client returns non-empty string
- Full integration: `build_context` → `generate_answer` → `parse_answer` on 1 real query

### `src/judging/__judging_test.py`  `[new-P2]`
Tests to run before any notebook cell in Section 4:
- `judge_sentence_alignment` returns correct schema `{label, pmid}`
- `label` is in {Required, Unnecessary, Borderline, Inappropriate}
- `judge_answer_entailment` returns correct schema `{label, unsupported_claims}`
- `label` is in {Supported, Partially Supported, Unsupported}
- Parse failure handled gracefully (retry or fallback)

---

## What Stays in the Notebook (not in src/)

These are things where seeing the code directly is better for the reader:

| What                                        | Why keep in notebook                                           |
| ------------------------------------------- | -------------------------------------------------------------- |
| Colab/local setup cell                      | One-time boilerplate, specific to this repo URL + secret names |
| Constants cell                              | Configuration by design — meant to be edited by the reader     |
| Prompt template display (markdown cells)    | Reader needs to see the actual text                            |
| `bertviz` head_view / model_view calls      | Returns interactive HTML widget, must be inline                |
| Demo/display loops (print results as cards) | Simple formatting, no reuse needed                             |
| Manual calibration table (4.4)              | By design manual — not automatable                             |
| `print_results` helper in Phase 1 notebook  | Used only once, 10-line display helper                         |

---

## `requirements.txt` additions needed for Phase 2  `[new-P2]`
```
nltk>=3.8
bertviz>=1.4.0
scikit-learn>=1.3          # PCA for embedding visualisation (already likely present)
seaborn>=0.12              # heatmaps (already likely present)
```

After installing, download NLTK data (both punkt variants for compatibility):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

---

## Phase 3 Modules (plan now, implement June 2026)

### `src/agent/planner.py`  `[new-P3]`
```python
def identify_subtopics(query: str, client: openai.OpenAI, model: str) -> list[str]
    # prompt LLM to decompose the main query into 3-5 sub-topics
    # returns list of sub-topic strings
```

### `src/agent/research_loop.py`  `[new-P3]`
```python
def research_subtopic(
    subtopic: str,
    retriever,          # Phase 1 best retriever (locked) — MUST already be initialized with live OpenSearch client
    cross_encoder,      # Phase 2 best cross-encoder (locked)
    vlm_client,
    vlm_model: str,
    corpus: list[dict],
) -> dict
    # full Phase 1+2 pipeline for one sub-topic
    # returns: {"subtopic": str, "selected_sentences": list, "answer": str, "parsed": dict}
    # NOTE: retriever must be initialized with a live OpenSearch client before passing in.
    # The dependency chain is: get_client() → Retriever(client) → research_subtopic(retriever, ...)
```

### `src/agent/synthesiser.py`  `[new-P3]`
```python
def synthesise_report(
    subtopic_results: list[dict],
    main_query:       str,
    client:           openai.OpenAI,
    model:            str,
) -> str
    # prompt LLM to aggregate sub-topic answers into a coherent final report with citations
```
