# src/ Structure Plan — BioMedical NL Agents

Design philosophy: each file owns one clear responsibility. Notebooks call src/ functions
directly — no multi-layer wrappers that hide what's happening. Reader of the notebook sees
the real call, not an abstraction on top of an abstraction.

Tests: small self-tests go in `if __name__ == "__main__"` at the bottom of the file.
If a file needs more than ~10 assertions (e.g. retrievers, generation, judge) it gets a
dedicated `__xxx_test.py` next to it.

---

## Current State (Phase 1 — all done)

```
src/
  __init__.py

  data/
    __init__.py
    loader.py               [DONE] load_corpus(), load_topics()
    splitter.py             [DONE] run_splitter() -> train/test JSON
    qrels_builder.py        [DONE] run_qrels_builder() -> binary + graded qrels JSON
    query_builder.py        [DONE] build_query(topic, field) -> str
    __loader_test.py        [DONE]
    __qrels_builder_test.py [DONE]

  indexing/
    __init__.py
    opensearch_client.py    [DONE] get_client(), check_index()
    index_builder.py        [DONE] build_index_mapping(), create_or_update_index(), get_live_fields(), float_tag()
    document_indexer.py     [DONE] index_documents()
    __index_builder_test.py [DONE]

  embeddings/
    __init__.py
    encoder.py              [DONE] Encoder class — mean pooling + L2 norm, singleton cache
    corpus_encoder.py       [DONE] create_embeddings(), load_embeddings()
    __encoder_test.py       [DONE]

  retrieval/
    __init__.py
    base.py                 [DONE] BaseRetriever, SparseRetriever, _extract_hits()
    bm25.py                 [DONE] BM25Retriever
    lm_jelinek_mercer.py    [DONE] LMJMRetriever
    lm_dirichlet.py         [DONE] LMDirichletRetriever
    knn.py                  [DONE] KNNRetriever
    rrf.py                  [DONE] RRFRetriever, rrf_merge()
    __retrieval_test.py     [DONE]

  evaluation/
    __init__.py
    metrics.py              [DONE] precision_at_k, recall_at_k, average_precision,
                                   mean_average_precision, reciprocal_rank, mean_reciprocal_rank,
                                   ndcg_at_k, mean_ndcg_at_k, pr_curve, mean_pr_curve,
                                   results_to_ranking, results_to_ranking_graded
    evaluator.py            [DONE] evaluate_retriever(), metrics_from_run(), save_run(), load_run()
    final_eval.py           [DONE] build and save all final eval tables / per-query tables
    plots.py                [DONE] plot_pr_curves(), plot_ap_boxplot(), plot_comparison_table(),
                                   plot_per_query_pr_curves()

  tuning/
    __init__.py
    cv_utils.py             [DONE] run_cv() -> 5-fold cross-validation utility
    sweeper.py              [DONE] run_bm25_sweep(), run_lmjm_sweep(), run_lmdir_sweep(),
                                   run_encoder_sweep(), run_query_field_sweep()
                                   SweepResult dataclass + to_dataframe()
    tuning_plots.py         [DONE] plot_sweep_heatmap(), plot_sweep_line(), plot_sweep_bar()
    __sweeper_test.py       [DONE]
```

---

## Phase 2 — To Build (due May 4)

### Design choices for the notebook balance

The notebook should show:
- the raw model call and its output (not hidden behind a "run_everything()" wrapper)
- the prompt string that goes to the LLM (transparency)
- the raw LLM response + the parsed result

So `reranking/`, `generation/`, and `judge/` each provide:
- low-level building blocks (score one pair, build one prompt, call the API once)
- a thin loop helper for batch processing (so the notebook doesn't have 40-line for-loops inline)

The BERT visualisation exercise is notebook-only code — no src/ file needed,
since it is a pedagogical exercise with no reuse in Phase 3.

---

```
src/

  reranking/                   -- NEW for Phase 2
    __init__.py
    cross_encoder.py
    __cross_encoder_test.py
```

**`reranking/cross_encoder.py`**

What it owns: load the cross-encoder model, score (query, doc) pairs, select top sentences.

```
Classes / Functions:

  class CrossEncoder
    - __init__(model_name: str, device: str | None = None)
        load model once, same singleton pattern as Encoder in embeddings/encoder.py
        default model: "ncbi/MedCPT-Cross-Encoder"
    - score(query: str, docs: list[str]) -> list[float]
        score a batch of (query, doc) pairs -> one float per doc (higher = more relevant)
        used in notebook for the "demo on one pair" cell, and in rerank() below
    - rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 20)
              -> list[tuple[str, float]]
        takes [(pmid, text), ...], returns [(pmid, score), ...] sorted desc
        this is the main call from the notebook reranking loop
    - top_sentences(query: str, text: str, n: int = 3) -> list[str]
        split text into sentences, score each (query, sentence), return top-n
        used in the sentence-selection section of the notebook

  Helper (module-level, not in class):
    _split_sentences(text: str) -> list[str]
        basic sentence splitting using regex (no spaCy dependency needed for abstracts)
        split on ". " and "? " and "! " — works fine for PubMed abstract format
```

Tests in `__cross_encoder_test.py`:
- model loads without error
- score() returns list of floats, same length as input docs
- scores are finite (no NaN/Inf)
- rerank() returns at most top_k items, sorted descending
- top_sentences() returns <= n sentences, all are non-empty strings
- longer test: rerank top-20 from a real BM25 run on one query, confirm P@5 >= P@5 of original order

---

```
  generation/                  -- NEW for Phase 2
    __init__.py
    answer_generator.py
    answer_validator.py
    __generation_test.py
```

**`generation/answer_generator.py`**

What it owns: build the prompt, call vLLM, return raw answer text.

```
Classes / Functions:

  class AnswerGenerator
    - __init__(model: str, base_url: str, api_key: str)
        openai client pointing to amalia.novasearch.org/vlm/v1
        model name queried from server at init time if not supplied
    - build_prompt(query: str,
                   evidence: list[dict],   # [{"pmid": "...", "sentences": [...]}]
                   system_role: str | None = None) -> list[dict]
        returns the messages list (system + user) ready for chat.completions.create()
        keep this public so the notebook can PRINT the prompt before calling the LLM
        this is the transparency: reader sees exact input to the model
    - generate(query: str, evidence: list[dict]) -> str
        calls build_prompt() then chat.completions.create()
        returns the raw answer string
        does NOT validate — that is answer_validator.py's job

  Module-level constants (set defaults, overridable from notebook):
    SYSTEM_ROLE_TEMPLATE: str    -- the system prompt text (concise, cite inline, <=3 PMIDs/sentence)
    MAX_TOKENS: int = 3500       -- generous ceiling for <= 2500 word answer
```

**`generation/answer_validator.py`**

What it owns: check all three hard constraints on a generated answer.
Keep each check as a separate function so the notebook can call them individually
and show which constraint fired.

```
Functions:

  count_words(text: str) -> int
      split on whitespace, count tokens

  extract_cited_pmids(text: str) -> list[str]
      regex: find all [PMID_XXXXXX] or (PMID: XXXXXX) patterns in the answer
      returns flat list of all PMIDs mentioned anywhere in the text

  cited_pmids_per_sentence(text: str) -> list[tuple[str, list[str]]]
      split into sentences, extract PMIDs from each
      returns [(sentence_text, [pmid, ...]), ...]

  validate_answer(answer: str, valid_pmids: set[str]) -> dict
      runs all 3 checks, returns:
      {
        "word_count":         int,
        "word_count_ok":      bool,   # <= 2500
        "max_pmids_per_sent": int,
        "pmids_per_sent_ok":  bool,   # all sentences <= 3 PMIDs
        "invalid_pmids":      list[str],
        "all_pmids_valid":    bool,
        "valid":              bool    # True only if all 3 pass
      }
      notebook displays this dict inline after each generation call
```

Tests in `__generation_test.py`:
- count_words: known string gives expected count
- extract_cited_pmids: regex catches [PMID_12345] and (PMID: 12345) formats
- validate_answer: hand-crafted good answer -> valid=True
- validate_answer: >2500 words -> word_count_ok=False
- validate_answer: 4 PMIDs in one sentence -> pmids_per_sent_ok=False
- validate_answer: invented PMID not in corpus -> all_pmids_valid=False
- generate() (integration, needs .env): returns a non-empty string for a real query

---

```
  judge/                       -- NEW for Phase 2
    __init__.py
    prompts.py
    llm_judge.py
    __judge_test.py
```

**`judge/prompts.py`**

What it owns: the prompt templates only — no API calls here. Kept separate
so templates can be read/printed/compared without importing the openai client.

```
Functions:

  build_relevance_prompt(query: str, sentence: str) -> list[dict]
      returns messages list for: "is this sentence relevant to the query?"
      output instruction: JSON {"relevant": true/false, "reason": "..."}

  build_entailment_prompt(query: str,
                          answer_sentence: str,
                          pmid: str,
                          abstract_text: str) -> list[dict]
      returns messages list for: "does this abstract support this answer sentence?"
      output instruction: JSON {"entailed": true/false, "reason": "..."}
```

**`judge/llm_judge.py`**

What it owns: call the IAedu/GPT-4o API, parse the JSON response.

```
Classes / Functions:

  class LLMJudge
    - __init__(model: str = "gpt-4o",
               base_url: str = ...,     # IAedu endpoint
               api_key: str = ...)
        openai client pointing to IAedu — NOT amalia.novasearch.org
    - judge_relevance(query: str, sentence: str) -> dict
        calls build_relevance_prompt(), calls API with response_format=json_object,
        parses -> {"relevant": bool, "reason": str}
        returns raw dict — notebook can display it inline
    - judge_entailment(query: str,
                       answer_sentence: str,
                       pmid: str,
                       abstract_text: str) -> dict
        calls build_entailment_prompt(), parses -> {"entailed": bool, "reason": str}
    - judge_answer(query: str,
                   answer: str,
                   corpus: dict[str, str]) -> dict
        convenience: runs relevance + entailment for every sentence in the answer
        returns per-sentence breakdown + aggregate stats
        this is the batch call used in the full loop in the notebook

  _parse_json_response(raw: str) -> dict
      module-level helper — extract JSON from the LLM response string
      handles: clean JSON, JSON wrapped in ```json ... ```, stray text before/after
      raises ValueError with the raw string if parsing fails (so caller can log it)
```

Tests in `__judge_test.py`:
- build_relevance_prompt: returns list of 2 dicts (system + user), non-empty strings
- build_entailment_prompt: same
- _parse_json_response: parses clean JSON, parses JSON in code block, raises on garbage
- judge_relevance (integration, needs .env + IAedu key): returns dict with "relevant" bool
- judge_entailment (integration): returns dict with "entailed" bool
- judge_answer: on a known good answer -> all_relevant rate and all_entailed rate both > 0

---

## Phase 3 — To Build (due Jun 1)

The agent is an orchestration layer on top of Phase 1 retrievers + Phase 2 generator.
It does NOT reimplement retrieval or generation — it imports them directly.
Notebook shows each stage explicitly: plan -> explore -> aggregate -> write.

```
src/

  agent/                       -- NEW for Phase 3
    __init__.py
    planner.py
    explorer.py
    aggregator.py
    report_writer.py
    __agent_test.py
```

**`agent/planner.py`**

What it owns: decompose a biomedical topic into sub-topics via LLM.

```
Functions:

  def plan(query: str,
           topic: str,
           question: str,
           narrative: str,
           llm_client,            # openai client (same vLLM as generation)
           model: str,
           max_subtopics: int = 5) -> list[str]
      builds a prompt asking the LLM to decompose the topic into N focused sub-topics
      parses the response as a numbered list
      returns list[str] — each is a focused sub-topic query string
      logs a warning if < 1 sub-topic returned

  _build_plan_prompt(query: str, topic: str, question: str, narrative: str,
                     max_subtopics: int) -> list[dict]
      prompt builder — kept separate so notebook can print it before calling the LLM
```

No class needed here — stateless function. The llm_client is injected by the notebook.

---

**`agent/explorer.py`**

What it owns: the ReAct loop for ONE sub-topic. Takes a sub-topic string, runs
Reason -> SEARCH -> Observe iterations, accumulates evidence, stops when done.

```
Classes:

  class Explorer
    - __init__(retriever,           # any BaseRetriever (the best Phase 1 retriever)
               cross_encoder,       # CrossEncoder instance
               llm_client,
               model: str,
               max_iterations: int = 6,
               min_evidence_docs: int = 3)
    - explore(subtopic: str) -> ExplorationResult
        runs the ReAct loop:
          1. reason: call LLM with current evidence state, ask what query to run next
          2. parse <action>SEARCH</action><query>...</query> from LLM output
          3. retrieve top-K docs with retriever.search()
          4. extract top sentences with cross_encoder.top_sentences()
          5. add to evidence buffer
          6. check stopping conditions
        returns ExplorationResult (see below)
        logs each iteration: reason snippet, query, #docs retrieved, stopping condition

  @dataclass ExplorationResult
      subtopic:        str
      iterations:      int
      stop_reason:     str           # "enough_evidence" | "max_iterations" | "loop_detected"
      evidence:        list[dict]    # [{"pmid": str, "sentences": list[str], "score": float}]
      queries_issued:  list[str]     # all search queries the LLM generated

  _build_reason_prompt(subtopic: str, evidence_so_far: list[dict]) -> list[dict]
      builds the prompt for the "reason" step
      includes: sub-topic, how many docs found so far, brief evidence summary
      kept module-level so it is testable and printable in notebook

  _parse_react_action(text: str) -> str | None
      parse <action>SEARCH</action><query>...</query> from LLM output
      returns the query string, or None if parsing fails
      logs a warning on parse failure
```

---

**`agent/aggregator.py`**

What it owns: merge evidence dicts from all sub-topics, deduplicate, cite-check.

```
Functions:

  def aggregate(exploration_results: list[ExplorationResult],
                corpus: dict[str, str]) -> AggregatedEvidence
      merges all evidence lists across sub-topics
      dedup by PMID (same PMID may appear under multiple sub-topics — keep all associations)
      removes any PMID not found in corpus (invalid citation — log a warning per removal)
      returns AggregatedEvidence

  @dataclass AggregatedEvidence
      by_subtopic:    dict[str, list[dict]]   # {subtopic: [{pmid, sentences, score}]}
      all_pmids:      list[str]               # unique PMIDs across all sub-topics, deduped
      n_removed:      int                     # PMIDs removed for not being in corpus
```

No class needed — stateless transformation. The notebook calls this once after all
exploration_results are collected.

---

**`agent/report_writer.py`**

What it owns: write the final structured report from aggregated evidence.

```
Classes / Functions:

  class ReportWriter
    - __init__(llm_client, model: str)
    - write_section(subtopic: str,
                    evidence: list[dict]) -> str
        prompt: "write a paragraph about {subtopic} citing these abstracts inline"
        returns one section of text (a few sentences + citations)
        kept small so the notebook can call it per-section and show each section as it's built
    - write_report(aggregated: AggregatedEvidence) -> str
        loops over all sub-topics, calls write_section() for each,
        joins with section headers (## {subtopic})
        validates full report with answer_validator.validate_answer()
        returns final report string

  _build_section_prompt(subtopic: str, evidence: list[dict]) -> list[dict]
      kept module-level — printable in notebook before calling LLM
```

---

**`agent/__agent_test.py`**

Tests covering all 4 agent components:

```
Planner tests:
  - _build_plan_prompt: returns valid messages list, subtopic count in prompt matches max_subtopics
  - plan() (integration, needs vLLM): returns >= 1 string for a real biomedical topic
  - plan(): output list is non-empty strings, no empty/whitespace entries

Explorer tests:
  - _parse_react_action: parses valid XML, returns None on garbage input
  - _build_reason_prompt: returns valid messages list, mentions subtopic in user message
  - ExplorationResult: fields populated correctly after a mocked explore() call
  - explore() (integration): terminates within max_iterations for a real subtopic
  - explore(): stop_reason is one of the 3 valid values

Aggregator tests:
  - aggregate(): dedup works — same PMID under 2 subtopics appears once in all_pmids
  - aggregate(): invalid PMID (not in corpus) is removed, n_removed > 0
  - aggregate(): by_subtopic keys match input subtopic strings

ReportWriter tests:
  - _build_section_prompt: returns valid messages list, mentions the subtopic
  - write_section() (integration): returns non-empty string with at least one PMID citation
  - write_report(): final report passes validate_answer() — valid=True
  - write_report(): has >= 1 section header (starts with ##)

End-to-end test:
  - full pipeline on 1 real topic: planner -> explorer (1 iteration cap for speed) ->
    aggregate -> write_report -> validate
    confirm: no crash, valid=True, >= 1 cited PMID in corpus
```

---

## Complete src/ Tree (final state after all phases)

```
src/
  __init__.py

  data/
    __init__.py
    loader.py               load_corpus(), load_topics()
    splitter.py             run_splitter()
    qrels_builder.py        run_qrels_builder()
    query_builder.py        build_query(topic, field)
    __loader_test.py
    __qrels_builder_test.py

  indexing/
    __init__.py
    opensearch_client.py    get_client(), check_index()
    index_builder.py        build_index_mapping(), create_or_update_index(),
                            get_live_fields(), float_tag()
    document_indexer.py     index_documents()
    __index_builder_test.py

  embeddings/
    __init__.py
    encoder.py              Encoder — mean pooling + L2 norm, singleton cache
    corpus_encoder.py       create_embeddings(), load_embeddings()
    __encoder_test.py

  retrieval/
    __init__.py
    base.py                 BaseRetriever (ABC), SparseRetriever, _extract_hits()
    bm25.py                 BM25Retriever
    lm_jelinek_mercer.py    LMJMRetriever
    lm_dirichlet.py         LMDirichletRetriever
    knn.py                  KNNRetriever
    rrf.py                  RRFRetriever, rrf_merge()
    __retrieval_test.py

  evaluation/
    __init__.py
    metrics.py              precision_at_k, recall_at_k, average_precision,
                            mean_average_precision, reciprocal_rank, mean_reciprocal_rank,
                            ndcg_at_k, mean_ndcg_at_k, pr_curve, mean_pr_curve,
                            results_to_ranking, results_to_ranking_graded
    evaluator.py            evaluate_retriever(), metrics_from_run(),
                            save_run(), load_run()
    final_eval.py           build/save final eval tables + per-query tables
    plots.py                plot_pr_curves(), plot_ap_boxplot(),
                            plot_comparison_table(), plot_per_query_pr_curves()

  tuning/
    __init__.py
    cv_utils.py             run_cv()
    sweeper.py              run_bm25_sweep(), run_lmjm_sweep(), run_lmdir_sweep(),
                            run_encoder_sweep(), run_query_field_sweep()
                            SweepResult dataclass
    tuning_plots.py         plot_sweep_heatmap(), plot_sweep_line(), plot_sweep_bar()
    __sweeper_test.py

  reranking/                -- Phase 2
    __init__.py
    cross_encoder.py        CrossEncoder — score(), rerank(), top_sentences()
    __cross_encoder_test.py

  generation/               -- Phase 2
    __init__.py
    answer_generator.py     AnswerGenerator — build_prompt(), generate()
                            SYSTEM_ROLE_TEMPLATE, MAX_TOKENS
    answer_validator.py     count_words(), extract_cited_pmids(),
                            cited_pmids_per_sentence(), validate_answer()
    __generation_test.py

  judge/                    -- Phase 2
    __init__.py
    prompts.py              build_relevance_prompt(), build_entailment_prompt()
    llm_judge.py            LLMJudge — judge_relevance(), judge_entailment(),
                            judge_answer()
                            _parse_json_response()
    __judge_test.py

  agent/                    -- Phase 3
    __init__.py
    planner.py              plan(), _build_plan_prompt()
    explorer.py             Explorer, ExplorationResult, _build_reason_prompt(),
                            _parse_react_action()
    aggregator.py           aggregate(), AggregatedEvidence
    report_writer.py        ReportWriter — write_section(), write_report(),
                            _build_section_prompt()
    __agent_test.py
```

---

## Testing Strategy Summary

| File                             | Where tests live            | What is tested                                                                        |
| -------------------------------- | --------------------------- | ------------------------------------------------------------------------------------- |
| `data/loader.py`                 | `__loader_test.py`          | corpus count, topic count, field presence, encoding                                   |
| `data/splitter.py`               | `__loader_test.py`          | train/test sizes, no ID overlap, odd/even split                                       |
| `data/qrels_builder.py`          | `__qrels_builder_test.py`   | evidence_relation mapping, binary vs graded values, no topic missing                  |
| `indexing/*.py`                  | `__index_builder_test.py`   | OpenSearch reachable, index exists, doc count = 4194, all fields present              |
| `embeddings/*.py`                | `__encoder_test.py`         | output shape, L2 norms = 1.0, semantic sanity check                                   |
| `retrieval/*.py`                 | `__retrieval_test.py`       | each strategy: 100 results, descending scores, no duplicate PMIDs                     |
| `evaluation/metrics.py`          | `if __name__ == "__main__"` | Lab03 toy example: AP(A)=1.0, AP(B)=0.7095                                            |
| `evaluation/evaluator.py`        | `if __name__ == "__main__"` | runs without error on 5 test topics, expected metric keys returned                    |
| `evaluation/plots.py`            | `if __name__ == "__main__"` | all 4 plot functions render without error, PNGs saved                                 |
| `tuning/sweeper.py`              | `__sweeper_test.py`         | BM25/LM-JM/LM-Dir sweeps return SweepResult with rows; best param selectable          |
| `reranking/cross_encoder.py`     | `__cross_encoder_test.py`   | model loads, score() shapes, rerank() sorted, top_sentences() non-empty               |
| `generation/answer_generator.py` | `__generation_test.py`      | prompt structure, generate() returns non-empty string (integration)                   |
| `generation/answer_validator.py` | `__generation_test.py`      | all 3 constraint checks pass/fail correctly on crafted examples                       |
| `judge/prompts.py`               | `__judge_test.py`           | prompt format — 2-message list, non-empty, JSON instruction present                   |
| `judge/llm_judge.py`             | `__judge_test.py`           | JSON parse helper, judge calls return correct keys (integration)                      |
| `agent/*.py`                     | `__agent_test.py`           | planner >= 1 subtopic, explorer terminates, aggregator deduplicates, report validates |

### Testing discipline

- All `if __name__ == "__main__"` tests run on tiny data (5-10 docs, 2-3 topics) for speed.
- `__xxx_test.py` files run with real data but still small batches.
- Integration tests (anything hitting OpenSearch or a LLM API) are gated by `.env` available — skip gracefully with a printed message if credentials not found.
- Workflow: implement in .py -> run local tests -> fix -> only then update the notebook.

---

## What stays notebook-only (no src/ file needed)

| Notebook cell                                    | Why it stays in the notebook                              |
| ------------------------------------------------ | --------------------------------------------------------- |
| `IN_COLAB` detection + git clone                 | one-off setup, no reuse                                   |
| Constants block (FORCE_*, BEST_*, PATHS)         | single source of truth per notebook session               |
| BERT positional embedding experiment (200-token) | pure pedagogical exercise, no reuse                       |
| BERT layer-by-layer contextual embedding plot    | same — no reuse in Phase 3                                |
| Self-attention heatmap visualization             | same                                                      |
| `print_results()` helper                         | 5 lines, notebook-only display util                       |
| PR curve inline display                          | already in `evaluation/plots.py` — notebook just calls it |
| Agent full trace print-out                       | notebook-specific verbose logging, not a library function |
| Report display / formatted output cell           | display only                                              |

---

## Key design decisions (and why)

**1. No utils.py**
Every function lives in the file whose domain it belongs to.
If a function is used by both `evaluator.py` and `sweeper.py` it lives in `metrics.py`.

**2. Prompt builders are separate from API callers**
`prompts.py` has no imports of `openai`. `llm_judge.py` imports prompts and calls the API.
Reason: the notebook can print the exact prompt before running it — important for transparency.
Same pattern for `answer_generator.py` (build_prompt is public) and `agent/*.py`.

**3. Stateless functions where possible, classes only when state matters**
- `aggregate()`, `plan()`, `_parse_react_action()` are module-level functions — no state needed.
- `CrossEncoder`, `AnswerGenerator`, `LLMJudge`, `Explorer`, `ReportWriter` are classes
  because they hold a loaded model or API client that must persist across calls.

**4. llm_client injected, not constructed inside src/**
`Explorer`, `ReportWriter`, `AnswerGenerator`, `LLMJudge` all receive the already-constructed
openai client as a constructor arg. The notebook creates the client (visible, explicit) and
passes it in. This keeps the notebook transparent about which endpoint is being called.

**5. `ExplorationResult` and `AggregatedEvidence` are dataclasses**
Notebook cells can inspect `.evidence`, `.queries_issued`, `.stop_reason` directly.
No magic dict access — attribute names are self-documenting.

**6. Cross-encoder sentence splitting is inline regex, not spaCy**
PubMed abstracts are well-structured. A regex split on `. ` / `? ` / `! ` is accurate enough
and avoids adding spaCy as a hard dependency for one helper function.
