# Phase 2 — Implementation Work Plan
### Notebook: `tasks/phase2_rag.ipynb`  |  Horizontal, step-by-step

---

## Guiding Rules

- **Horizontal flow:** implement notebook cell(s) → src function(s) → test → run → confirm output.
  Never implement a whole module speculatively without a notebook cell that uses it.
- **Minimum code:** only implement what the current step actually needs. No "might be useful later."
- **Test everything:** tests must exercise the full notebook infrastructure (not just src internals).
  Run with tiny `SAMPLE_SIZE` (5-10 topics, 50 docs) so tests are fast.
- **Output quality:** after each step, confirm: prints are clean, plots render correctly with right labels,
  tables display well (`pd.DataFrame.style` or `.display()`), saved JSON files are valid.
- **Local + Colab:** all notebook cells must work identically in both environments.
  Follow the pattern from `phase1_search.ipynb` cells exactly (Colab secrets, autoreload, ROOT path).

---

## !!!!! DOUBTS — ALL RESOLVED

**D1 — Cross-encoder model ✅ RESOLVED**
> **Decision:** `ncbi/MedCPT-Cross-Encoder` as primary. MS-MARCO as comparison baseline in §1.7 ablation only.
> **Note:** Verify output logit shape on first load — add assertion for 1-class vs 2-class output.

**D2 — Sentence splitter ✅ RESOLVED**
> **Decision:** `nltk.sent_tokenize` (punkt). Simple, fast, no heavy dependencies.
> scispaCy noted as improvement in report. Download both `punkt` and `punkt_tab` for NLTK ≥3.9.

**D3 — Top-k docs into re-ranker ✅ RESOLVED**
> **Decision:** Configurable argument `top_k`, default 10. Can sweep in Phase 3.

**D4 — Generator LLM ✅ RESOLVED**
> **Decision:** GPT-4o via IAedu API for both generation AND judging.
> vLLM at `amalia.novasearch.org` kept as optional secondary for §5.3 comparison.

**D5 — Generation prompt style ✅ RESOLVED**
> **Decision:** Zero-shot first. Function designed to be flexible — easy to add few-shot examples later.

---

## Step-by-Step Plan

Each step has: what to build, what tests to run, what the output should look like.

---

### STEP 1 — Create notebook shell + Section 0: Setup & Data [DONE]

**What to build:**
- Create `tasks/phase2_rag.ipynb`
- Cells: title/index markdown, Colab/local setup cell, constants cell, imports cell,
  data loading cell, Phase 1 recap cell.

**Notebook cells (Section 0):**

```
[MD]  Title + Table of Contents  (copy structure from phase1_search.ipynb)
[MD]  0. Project Set Up — what this section does
[MD]  0.1 Constants & Config
[PY]  All Phase 2 constants (see below)
[MD]  0.2 Colab / Local Setup
[PY]  Colab/local setup — identical pattern to phase1_search.ipynb setup cell
[MD]  0.3 Imports
[PY]  All imports (torch, transformers, openai, nltk, matplotlib, seaborn, pandas, etc.)
      + GPU check: print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if cuda else "CPU")
      + NLTK data download: nltk.download('punkt'); nltk.download('punkt_tab')  # both for >=3.9 compat
[MD]  0.4 Data Loading — brief recap of corpus/topics/qrels
[PY]  Load corpus, train_topics, test_topics, qrels, qrels_graded (reuse src.data.* from Phase 1)
      Load Phase 1 best run file: results/phase1/knn_best_test_run.json (or whichever is best)
      Build corpus_lookup = {doc["id"]: doc["contents"] for doc in corpus}
      Build valid_pmids   = {doc["id"] for doc in corpus}
      Print: corpus size, train/test topic counts, qrels coverage
[MD]  0.5 Phase 1 Locked Configuration — table of best Phase 1 params + nDCG score
[PY]  Rebuild best Phase 1 retriever from constants. Run on 1 demo query. Print top-3 results.
```

**Phase 2 constants to define:**
```python
# ── Phase 2 flags ────────────────────────────────────────────────────────────
FORCE_RERUN_GENERATION  = False  # skip if results/phase2/train_answers.json exists
FORCE_RERUN_JUDGING     = False  # skip if judge result files exist

# ── Retrieval config (locked from Phase 1) ────────────────────────────────────
P2_RETRIEVER            = "KNN(MedCPT)"   # or whatever Phase 1 best was
P2_RETRIEVAL_TOP_K      = 10              # top-k docs to feed into re-ranker

# ── Cross-encoder config ────────────────────────────────────────────────────
CROSS_ENCODER_MODEL     = "ncbi/MedCPT-Cross-Encoder"
TOP_SENTENCES_PER_DOC   = 3               # keep top-3 sentences per document

# ── Generation config ────────────────────────────────────────────────────────
VLLM_URL                = "https://amalia.novasearch.org/vlm/v1"
VLLM_KEY                = "amalia012026"
GENERATION_TEMPERATURE  = 0.1
MAX_ANSWER_WORDS        = 250
MAX_PMIDS_PER_SENTENCE  = 3

# ── Judge / Primary LLM config (GPT-4o for both generation and judging) ────
IAEDU_URL               = "..."   # from env: IAEDU_URL
IAEDU_KEY               = "..."   # from env: IAEDU_KEY
GPT4O_MODEL             = "gpt-4o"

# ── Output paths ─────────────────────────────────────────────────────────────
PHASE2_DIR              = ROOT / "results/phase2"
TRAIN_ANSWERS_FILE      = PHASE2_DIR / "train_answers.json"
TEST_ANSWERS_FILE       = PHASE2_DIR / "test_answers.json"
SENTENCE_ALIGN_TRAIN    = PHASE2_DIR / "sentence_alignment_train.json"
SENTENCE_ALIGN_TEST     = PHASE2_DIR / "sentence_alignment_test.json"
ENTAILMENT_TRAIN        = PHASE2_DIR / "entailment_train.json"
ENTAILMENT_TEST         = PHASE2_DIR / "entailment_test.json"
PHASE2_FIGURES_DIR      = PHASE2_DIR / "figures"
```

**Src files needed:** none new (reuse all Phase 1 src).

**Tests:** none new at this step (Phase 1 src is already tested).

**Confirm output:**
- Prints show corpus=4194, train=32, test=33.
- Phase 1 best run loads cleanly (32 train queries, each with ≥10 PMIDs).
- Phase 1 demo retrieval returns 10 results, top-3 printed with PMID + score.

---

### STEP 2 — Section 2: Transformer Internals (analysis exercise) [DONE]

> Do Section 2 before Section 1 (cross-encoder) because it uses the model loaded fresh
> for analysis, with no dependency on sentence selection or generation.

**What to build:**
- `src/analysis/transformer_inspector.py` — `get_hidden_states_and_attentions()`, `get_positional_embeddings()`, `cosine_distance_matrix()`
- `src/analysis/attention_plots.py` — `plot_positional_embedding_scatter()`, `plot_pairwise_distance_heatmap()`, `plot_contextual_embeddings_grid()`, `plot_attention_matrix()`
- Notebook cells for Section 2 (see below).

**Notebook cells (Section 2):**
```
[MD]  # 2. Transformer Internals: Embeddings & Attention
      Explain the 3 things we will analyse + transformer block ASCII diagram
[MD]  ## 2.1 Model Setup for Analysis
      Short note: we use bert-base-uncased for clean positional embedding demo,
      then switch to MedCPT cross-encoder for the self-attention (relevance) analysis.
[PY]  Load bert-base-uncased with output_hidden_states=True, output_attentions=True
      Print: n_layers, hidden_dim, n_heads, vocab_size
[MD]  ## 2.2 Positional Embeddings — Distance from Token 0
      Theory: learned positional embeddings, same-word trick to isolate position.
      Math: cosine_distance(e_i, e_0) = 1 - cos(e_i, e_0)
[PY]  Call get_positional_embeddings(tokenizer, model, word="the", n=200)
      Compute cosine distance from token 0 -> vector (200,)
      Plot C: PCA 2D scatter, color by distance from token 0
[MD]  ## 2.3 Pairwise Distance Matrix
      Brief theory note.
[PY]  Call cosine_distance_matrix(embeddings) -> (200, 200)
      Plot D: seaborn heatmap
[MD]  Analysis cell: describe what the matrix structure tells us about BERT's learned positional encodings.
[MD]  ## 2.4 Contextual Embeddings — Layer-by-Layer Evolution
      Theory: how representations evolve from embeddings layer (layer 0) to deep layers.
[PY]  Pick biomedical sentence. Call get_hidden_states_and_attentions().
      For each of 12 layers: PCA project to 2D (fit on layer 11 for common space).
      Plot E: 3x4 grid of scatter plots, token labels, layer index in title.
[MD]  Analysis cell: describe early vs late layer differences.
[MD]  ## 2.5 Self-Attention Analysis
      Theory: cross-encoder attention — query tokens attend to document tokens.
[PY]  Reload MedCPT-Cross-Encoder with output_attentions=True.
      Pick (query, relevant_sentence) pair and (query, irrelevant_sentence) pair.
      Call get_hidden_states_and_attentions() for both.
      Plot F: attention matrix heatmap for relevant pair (layer 11, mean across heads).
      Plot G: attention matrix heatmap for irrelevant pair (layer 11, mean across heads).
      Try bertviz head_view inline — if it works, show it; if HTML rendering fails, skip.
[MD]  Critical analysis cell: compare the two heatmaps, describe what patterns differ.
```

**New src files:**
- `src/analysis/__init__.py` — empty
- `src/analysis/transformer_inspector.py` — 3 functions above
- `src/analysis/attention_plots.py` — 4 plot functions above

**Tests (no separate test file needed for analysis — test inline in notebook):**
- After loading: `assert hidden_states.shape[0] == 13` (12 layers + embedding layer)
- After positional embeddings: `assert embeddings.shape == (200, 768)` (or up to model max_length)
- After distance matrix: `assert dist_matrix.shape[0] == dist_matrix.shape[1]`
- Plots: confirm figure renders, axes have labels, grid shows 12 subplots.

**Confirm output:**
- All 4 plots render correctly in notebook (no errors, no blank figures).
- Scatter plot shows visible distance gradient (nearby positions have similar color).
- Heatmap is banded (diagonal + near-diagonal lower distances).
- Layer grid shows meaningful token movement across layers.
- Self-attention heatmaps clearly differ between relevant and irrelevant pairs.

---

### STEP 3 — Section 1: Cross-Encoder Re-Ranking [DONE]

> **D1 ✅ RESOLVED:** MedCPT-Cross-Encoder as primary; MS-MARCO as comparison baseline.
> **D2 ✅ RESOLVED:** NLTK `sent_tokenize` (punkt).

**What to build:**
- `src/reranking/__init__.py`
- `src/reranking/sentence_splitter.py` — `split_sentences()`, `select_top_sentences()`
- `src/reranking/cross_encoder.py` — `CrossEncoder` class
- `src/reranking/__reranking_test.py` — comprehensive tests

**Notebook cells (Section 1):**
```
[MD]  # 1. Cross-Encoder Re-Ranking
[MD]  ## 1.1 Bi-Encoder vs Cross-Encoder
      Explain the difference: bi-encoder (encode independently, dot product) vs
      cross-encoder (joint encoding, full attention). Formula: score(q,s) = f(BERT([q;s]))[0]
      Why cross-encoder is slower but more accurate. Why MedCPT is the right choice here.
[MD]  ## 1.2 Load Cross-Encoder
[PY]  from src.reranking.cross_encoder import CrossEncoder
      ce = CrossEncoder(CROSS_ENCODER_MODEL)
      # print: model name, n_layers, hidden_dim
      # smoke test: score 1 pair
      test_score = ce.score_query_vs_sentences("sleep apnea", ["CPAP is effective."])
      print(f"Smoke test score: {test_score[0]:.4f}")
      assert isinstance(test_score[0], float)
      # IMPORTANT: verify logit output shape — some HF cross-encoders output 2-class logits
      # [irrelevant, relevant]. If so, use logits[:, 1] not logits[:, 0]. MedCPT should be 1-class.
[MD]  ## 1.3 Sentence Segmentation
      Motivation: abstracts are multi-sentence, not all sentences equally relevant.
      Show the splitter (nltk.sent_tokenize) on 1 abstract, display all sentences with indices.
[PY]  from src.reranking.sentence_splitter import split_sentences
      example_abstract = corpus[0]["contents"]
      sentences = split_sentences(example_abstract)
      for i, s in enumerate(sentences): print(f"  [{i+1}] {s}")
[MD]  ## 1.4 Sentence Scoring Demo
      Walk through the full pipeline on 2 example queries:
      one with high Phase 1 AP, one with low Phase 1 AP.
[PY]  from src.reranking.sentence_splitter import select_top_sentences
      # Pick demo query
      demo_topic = train_topics[0]
      demo_results = best_retriever.search(build_query(demo_topic, BEST_QUERY_FIELD), size=P2_RETRIEVAL_TOP_K)
      # For first doc: show all sentences + scores, highlight top-3
      first_pmid = demo_results[0][0]
      first_abstract = corpus_lookup[first_pmid]
      top3 = select_top_sentences(build_query(demo_topic, BEST_QUERY_FIELD), first_abstract, ce, top_n=TOP_SENTENCES_PER_DOC)
      # print formatted table: sentence | score | selected?
[MD]  ## 1.5 Score Distribution Analysis
      What does the distribution of cross-encoder scores look like across relevant vs non-relevant docs?
[PY]  # Run sentence scoring on 15 train queries (sample)
      # Collect all (sentence, score, pmid, is_relevant) tuples
      # Plot A: histogram relevant vs non-relevant scores (overlapping, different colors)
      # Plot B: boxplot per relevance label
      # Table 1: mean ± std per label  (pd.DataFrame.style)
[MD]  Analysis: do cross-encoder scores separate relevant from non-relevant?
[MD]  ## 1.6 Ablation: Random vs Cross-Encoder Selection
[PY]  # Random baseline: randomly pick 3 sentences from abstract
      # For 15 train queries: compare % selected sentences from relevant PMIDs
      # Table 2: Cross-encoder top-3 vs Random top-3
```

**New src files:**
- `src/reranking/__init__.py`
- `src/reranking/cross_encoder.py`
- `src/reranking/sentence_splitter.py`
- `src/reranking/__reranking_test.py`

**Tests in `__reranking_test.py`:**
```python
# Group 1: CrossEncoder
- ce loads without error
- score_pairs([(q, s)]) returns list of len=1, value is float
- score_pairs with batch_size=2 on 5 pairs returns list of len=5
- scores are NOT sorted by input order (confirm batching doesn't sort)

# Group 2: sentence_splitter
- split_sentences("") returns []
- split_sentences(None) returns []
- split_sentences("One sentence.") returns ["One sentence."]
- split_sentences("First. Second. Third.") returns 3 items
- split_sentences with "e.g." and "i.v." doesn't over-split
- split_sentences on title-only text (no period) returns 1 item or []

# Group 3: select_top_sentences
- returns exactly top_n items
- returned items sorted by score descending
- assert top3[0]["score"] >= top3[-1]["score"]
- returns fewer if abstract has < top_n sentences

# Group 4: integration
- run full pipeline on 1 query + 1 abstract -> top3 has expected structure {sentence, score, rank}
```

**Confirm notebook output:**
- Cross-encoder loads cleanly (model name printed).
- Smoke test score is a float (likely in range [-10, 10] for MedCPT).
- Sentence segmentation demo shows ≥3 sentences for any abstract.
- Demo pipeline cell: top-3 sentences printed, sorted by score.
- Plot A: two overlapping histograms with different colors.
- Table 1 renders as a styled DataFrame.

---

### STEP 4 — Section 3: LLM Clients & Context Building [DONE]

**What to build:**
- `src/generation/__init__.py`
- `src/generation/llm_client.py` — `get_gpt4o_client()`, `get_vlm_client()`, `get_vlm_model_name()`
- `src/generation/context_builder.py` — `build_context()`

**Notebook cells (Section 3, part 1):**
```
[MD]  # 3. Answer Generation
[MD]  ## 3.1 RAG Pipeline Design
      Full pipeline diagram (text art): query -> retrieval -> sentences -> context -> LLM -> parse
      Constraints: <=250 words, <=3 PMIDs/sentence, only valid PMIDs.
[MD]  ## 3.2 LLM Client Setup
[PY]  from src.generation.llm_client import get_gpt4o_client, get_vlm_client, get_vlm_model_name
      gpt_client = get_gpt4o_client()    # primary: both generation and judging
      vlm_client = get_vlm_client()      # optional secondary generator
      vlm_model  = get_vlm_model_name(vlm_client)
      print(f"GPT-4o     : {GPT4O_MODEL} (primary)")
      print(f"vLLM model : {vlm_model} (secondary)")
      # smoke test: one call to GPT-4o
      resp = gpt_client.chat.completions.create(
          model=GPT4O_MODEL,
          messages=[{"role": "user", "content": "Say hello in one word."}],
          max_tokens=5,
      )
      print(f"GPT-4o ping: {resp.choices[0].message.content}")
[MD]  ## 3.3 Context Formatting
      Show the [PMID X] format we use and why (grounding, attribution).
[PY]  from src.generation.context_builder import build_context
      valid_pmids = {doc["id"] for doc in corpus}
      # demo: build context from top3 of first demo query
      example_selected = [{"pmid": s_pmid, "sentence": s_text, "score": s_score} for ...]
      context = build_context(example_selected, valid_pmids)
      print(context)
```

**New src files:**
- `src/generation/__init__.py`
- `src/generation/llm_client.py`
- `src/generation/context_builder.py`

**Tests (run in notebook after these cells):**
```python
# GPT-4o ping must return non-empty string
# vLLM ping (1 call): check response has .choices[0].message.content
# build_context with valid PMIDs: output contains "[PMID " prefix
# build_context with invalid PMID: should not raise, just warn
# build_context with empty list: returns ""
```

**Confirm notebook output:**
- GPT-4o ping returns 1-word response (primary LLM confirmed working).
- vLLM model name printed (non-empty string).
- Context string shows correctly formatted `[PMID XXXXXXXX] sentence text.` lines.

---

### STEP 5 [DONE] — Section 3 (continued): Answer Generator & Parser

> **D4 ✅ RESOLVED:** GPT-4o for both generation and judging.
> **D5 ✅ RESOLVED:** Zero-shot first, flexible to add few-shot later.

**What to build:**
- `src/generation/answer_generator.py` — `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `generate_answer()`
- `src/generation/answer_parser.py` — `parse_answer()`, `check_constraints()`
- `src/generation/__generation_test.py` — comprehensive tests

**Notebook cells (Section 3, part 2):**
```
[MD]  ## 3.4 Prompt Design
      Show SYSTEM_PROMPT and USER_PROMPT_TEMPLATE verbatim. Explain each design choice:
      - system role sets biomedical expert persona + grounding constraint
      - user section shows evidence labeled by PMID, then asks the question
      - format instruction: "write N sentences, each ending with [PMID X, PMID Y]"
      - temperature=0.1 for consistency
[PY]  from src.generation.answer_generator import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
      print("--- SYSTEM PROMPT ---")
      print(SYSTEM_PROMPT)
      print("\n--- USER TEMPLATE (with placeholders) ---")
      print(USER_PROMPT_TEMPLATE)
[MD]  ## 3.5 Generation + Parsing Functions
      Brief note on what generate_answer and parse_answer do.
[PY]  from src.generation.answer_generator import generate_answer
      from src.generation.answer_parser import parse_answer, check_constraints
      # end-to-end demo on 3 training queries
      for demo_topic in train_topics[:3]:
          query   = build_query(demo_topic, BEST_QUERY_FIELD)
          results = best_retriever.search(query, size=P2_RETRIEVAL_TOP_K)
          # get selected sentences for all top-k docs
          all_selected = []
          for pmid, _ in results:
              abstract = corpus_lookup.get(pmid, "")
              if not abstract: continue
              top3 = select_top_sentences(query, abstract, ce, top_n=TOP_SENTENCES_PER_DOC)
              all_selected += [{"pmid": pmid, **s} for s in top3]
          context = build_context(all_selected, valid_pmids)
          raw     = generate_answer(demo_topic["question"], context, gpt_client, GPT4O_MODEL)
          parsed  = parse_answer(raw, valid_pmids)
          # display as a nice card
          print(f"\n{'='*60}")
          print(f"Q: {demo_topic['question']}")
          print(f"\nA: {parsed['text']}")
          print(f"\nConstraints: words={parsed['word_count']} | pass={check_constraints(parsed)}")
[MD]  ## 3.6 Constraint Violation Analysis
[PY]  # Run on all 32 train queries
      all_parsed = run_bulk_generation(train_topics, ...) # see bulk generation below
      # Table 3: constraint compliance %
      # Plot H: histogram of word counts
      # Plot I: bar chart of PMIDs per sentence
```

**`generate_answer()` prompt drafts (show in notebook, live in src):**

*SYSTEM_PROMPT:*
```
You are a biomedical expert. Answer the given clinical question using ONLY the provided
reference sentences. Each sentence in your answer MUST end with citations in the format
[PMID X] or [PMID X, PMID Y]. Do not add any information not present in the references.
Total answer must be 250 words or fewer. Maximum 3 PMIDs per sentence.
```

*USER_PROMPT_TEMPLATE:*
```
Reference evidence:
{context}

Question: {question}

Write a concise biomedical answer (max 250 words). Cite each sentence with [PMID X].
```

**New src files:**
- `src/generation/answer_generator.py`
- `src/generation/answer_parser.py`
- `src/generation/__generation_test.py`

**Tests in `__generation_test.py`:**
```python
# Group 1: context_builder (can also run here)
- build_context([{"pmid": "1", "sentence": "test", "score": 1.0}], {"1"}) 
  -> "[PMID 1] test."

# Group 2: answer_parser
- parse_answer on a hand-crafted string with [PMID X] citations:
  -> word_count correct, sentences correct, pmids extracted
- parse_answer on string with >3 PMIDs in one sentence:
  -> violations["sentences_over_3_pmids"] is non-empty
- parse_answer on string with PMID not in corpus:
  -> violations["invalid_pmids"] is non-empty
- parse_answer on >250 word answer:
  -> violations["over_word_limit"] == True
- check_constraints on clean answer: True
- check_constraints on violating answer: False

# Group 3: integration (needs real vLLM connection)
- run_on_1_query: generate_answer -> parse_answer -> check_constraints
  -> answer is non-empty string, parsed has "text" and "sentences" keys
```

**Confirm notebook output:**
- 3 demo answers displayed as clean cards with Q + A + constraint check.
- All 3 demo answers pass constraints (if not, fix prompt and re-run).
- Histogram of word counts shows distribution ≤250 for most queries.

---

### STEP 6 [DONE] — Bulk Generation (Section 3.7)

**What to build:**
- Helper function `run_bulk_rag_pipeline()` — in notebook (not src) because it's
  wiring specific to this notebook's constants. About 20 lines.

**Notebook cells:**
```
[MD]  ## 3.7 Bulk Answer Generation — All Training Queries
      Note: ~32 queries × 10 docs × 3 sentences each = ~960 cross-encoder calls + 32 LLM calls.
      Estimated time: X min on GPU, Y min on CPU. FORCE_RERUN_GENERATION flag skips if cached.
[PY]  # inline helper — define and call in same cell
      def run_bulk_rag_pipeline(topics, retriever, cross_enc, llm_client, llm_model, corpus_lookup, valid_pmids):
          results = []
          for topic in tqdm(topics, desc="Generating answers"):
              query   = build_query(topic, BEST_QUERY_FIELD)
              hits    = retriever.search(query, size=P2_RETRIEVAL_TOP_K)
              selected = []
              for pmid, _ in hits:
                  abstract = corpus_lookup.get(pmid, "")
                  if abstract:
                      top_sents = select_top_sentences(query, abstract, cross_enc, TOP_SENTENCES_PER_DOC)
                      selected += [{"pmid": pmid, **s} for s in top_sents]
              context = build_context(selected, valid_pmids)
              raw     = generate_answer(topic["question"], context, llm_client, llm_model)
              parsed  = parse_answer(raw, valid_pmids)
              results.append({"query_id": str(topic["id"]), "question": topic["question"],
                               "selected_sentences": selected, "raw_answer": raw, "parsed": parsed})
          return results

      if not TRAIN_ANSWERS_FILE.exists() or FORCE_RERUN_GENERATION:
          train_answers = run_bulk_rag_pipeline(train_topics, best_retriever, ce, gpt_client, GPT4O_MODEL, corpus_lookup, valid_pmids)
          PHASE2_DIR.mkdir(parents=True, exist_ok=True)
          with open(TRAIN_ANSWERS_FILE, "w") as f: json.dump(train_answers, f, indent=2)
          print(f"Saved {len(train_answers)} answers -> {TRAIN_ANSWERS_FILE}")
      else:
          with open(TRAIN_ANSWERS_FILE) as f: train_answers = json.load(f)
          print(f"Loaded cached answers: {len(train_answers)} entries")

      n_pass = sum(check_constraints(a["parsed"]) for a in train_answers)
      print(f"Constraint compliance: {n_pass}/{len(train_answers)} ({100*n_pass/len(train_answers):.1f}%)")
```

**Src files needed:** none new — reuses all from Steps 3/4/5.

**Tests before running bulk:**
- Run `__generation_test.py` tests (from Step 5) — all must pass.
- Test with `SAMPLE_SIZE=5` topics first to confirm no errors, then full run.

**Confirm output:**
- `results/phase2/train_answers.json` exists and is valid JSON.
- All 32 entries have required keys: `query_id`, `question`, `selected_sentences`, `raw_answer`, `parsed`.
- Constraint compliance ≥80% (if lower, revisit prompt before proceeding).

---

### STEP 7 [DONE] — Section 4: LLM Judge Setup & Calibration

**What to build:**
- `src/judging/__init__.py`
- `src/judging/llm_judge.py` — prompt constants + 3 functions
- `src/judging/__judging_test.py` — tests

**Notebook cells (Section 4, part 1):**
```
[MD]  # 4. LLM-as-a-Judge Evaluation
[MD]  ## 4.1 Motivation: Why LLM Judges?
      Explain human annotation cost, LLM-as-judge literature (GPT-4 correlates ~0.85 with humans
      on biomedical tasks). Two evaluation axes: sentence alignment + answer entailment.
      Entailment != relevance — explain the distinction.
[MD]  ## 4.2 Judge Prompt Design — Sentence Alignment
      Show SENTENCE_ALIGNMENT_SYSTEM_PROMPT + SENTENCE_ALIGNMENT_USER_TEMPLATE verbatim.
      TREC BioGen labels: Required / Unnecessary / Borderline / Inappropriate.
      Explain that we start with the guide's exact uncalibrated prompt.
[PY]  from src.judging.llm_judge import (
          SENTENCE_ALIGNMENT_SYSTEM_PROMPT, SENTENCE_ALIGNMENT_USER_TEMPLATE,
          ENTAILMENT_SYSTEM_PROMPT, ENTAILMENT_USER_TEMPLATE,
          judge_sentence_alignment, judge_answer_entailment, batch_judge_alignment,
      )
      print(SENTENCE_ALIGNMENT_SYSTEM_PROMPT)
[MD]  ## 4.3 Judge Prompt Design — Answer Entailment
      Show ENTAILMENT_SYSTEM_PROMPT verbatim. TREC BioGen labels: Supported / Partially Supported / Unsupported.
[PY]  print(ENTAILMENT_SYSTEM_PROMPT)
[MD]  ## 4.4 Prompt Calibration on Manual Sample
      We test on 5 examples BEFORE bulk evaluation to ensure judge is calibrated.
[PY]  # Randomly select 5 training answers
      sample_5 = random.sample(train_answers, 5)
      calibration_results = []
      for ans in sample_5:
          # judge 1 sentence from this answer
          sent = ans["selected_sentences"][0]
          align = judge_sentence_alignment(ans["question"], sent["sentence"], sent["pmid"], gpt_client)
          entail = judge_answer_entailment(ans["question"], ans["parsed"]["text"],
                                           [s["sentence"] for s in ans["selected_sentences"]], gpt_client)
          calibration_results.append({**ans, "alignment": align, "entailment": entail})
      # Display as a pandas table
      # Table 4: query | sentence snippet | judge label | your manual label | match?
[MD]  Analysis cell: assess judge quality, note any calibration issues.
      Document any prompt changes made here. (Fill this in AFTER running the cell above.)
      If the guide's uncalibrated prompt needs adaptation (e.g., add reasoning output,
      refine label definitions), document the changes and rationale here.
```

**New src files:**
- `src/judging/__init__.py`
- `src/judging/llm_judge.py`
- `src/judging/__judging_test.py`

**Tests in `__judging_test.py`:**
```python
# Group 1: judge_sentence_alignment
- returns dict with keys: label (str), pmid (str)
- label is in {Required, Unnecessary, Borderline, Inappropriate}
- runs without error on a simple (query, sentence, pmid)

# Group 2: judge_answer_entailment
- returns dict with keys: label (str), unsupported_claims (list)
- label is in {Supported, Partially Supported, Unsupported}

# Group 3: batch_judge_alignment
- returns list of same length as input sentences
- each item has correct schema
```

**Confirm notebook output:**
- Calibration table shows 5 rows, labels visible (Required/Unnecessary/etc.).
- At least 1 cell in the calibration table looks plausible (relevant sentence labeled Required).
- If judge seems miscalibrated (always labeling Required, never Unnecessary), note it and revise prompt.

---

### STEP 8 [DONE] — Bulk Judging (Sections 4.5 & 4.6) + Analysis (4.7 & 4.8)

**What to build:**
- Bulk judging helper inline in notebook (same design as `run_bulk_rag_pipeline`).
- Analysis plots: inline in notebook (seaborn/matplotlib, ~15 lines each).

**Notebook cells:**
```
[MD]  ## 4.5 Bulk Sentence Alignment — Training Set
[PY]  # bulk alignment judge with caching (FORCE_RERUN_JUDGING flag)
      if not SENTENCE_ALIGN_TRAIN.exists() or FORCE_RERUN_JUDGING:
          align_results = []
          for ans in tqdm(train_answers, desc="Judging alignment"):
              row = {"query_id": ans["query_id"],
                     "judgements": batch_judge_alignment(ans["question"], ans["selected_sentences"], gpt_client)}
              align_results.append(row)
          with open(SENTENCE_ALIGN_TRAIN, "w") as f: json.dump(align_results, f, indent=2)
      else:
          with open(SENTENCE_ALIGN_TRAIN) as f: align_results = json.load(f)
      # Table 5: top-5 / bottom-5 queries by % Required labels  (pd.DataFrame.style)
      # Plot J: stacked bar chart - distribution of labels (Required/Unnecessary/Borderline/Inappropriate)
[MD]  ## 4.6 Bulk Answer Entailment — Training Set
[PY]  # bulk entailment judge with caching
      if not ENTAILMENT_TRAIN.exists() or FORCE_RERUN_JUDGING:
          entail_results = []
          for ans in tqdm(train_answers, desc="Judging entailment"):
              result = judge_answer_entailment(
                  ans["question"], ans["parsed"]["text"],
                  [s["sentence"] for s in ans["selected_sentences"]], gpt_client)
              entail_results.append({"query_id": ans["query_id"], **result})
          with open(ENTAILMENT_TRAIN, "w") as f: json.dump(entail_results, f, indent=2)
      else:
          with open(ENTAILMENT_TRAIN) as f: entail_results = json.load(f)
      # Table 6: % Supported / Partially Supported / Unsupported
      # Plot K: bar chart distribution of entailment labels
[MD]  ## 4.7 Correlation: Retrieval Quality -> Answer Quality
[PY]  # join Phase 1 per-query nDCG with Phase 2 entailment
      # Plot L: scatter nDCG vs entailment, regression line, Pearson r
      # Plot M: scatter mean alignment vs entailment
[MD]  Analysis: what does the correlation tell us about the pipeline bottleneck?
[MD]  ## 4.8 Error Analysis — Failure Cases
[PY]  # find 2 Unsupported and 2 Supported examples
      # display as formatted cards: Q | selected sentences | generated answer | judge label
```

**Confirm notebook output:**
- All 4 JSON files written to `results/phase2/`.
- Table 5 (top/bottom queries) renders correctly.
- Plot J: stacked bar chart with legend (Required=green, Unnecessary=orange, Borderline=yellow, Inappropriate=red).
- Plot K: clear bar chart of 3 entailment levels with % (Supported/Partially Supported/Unsupported).
- Plots L+M: scatter visible with regression line, correlation value printed.
- Error analysis cards readable with clear formatting.

---

### STEP 9 — Section 5: Test Set + Final Tables [DONE]

**What to build:**
- No new src files — reuse everything from Steps 1-8.
- Notebook cells for test set run + final tables.

**Notebook cells:**
```
[MD]  # 5. Test Set Evaluation & Summary
[MD]  ## 5.1 Test Set Generation + Judging
      Note: held-out set, no tuning allowed here.
[PY]  # Run full pipeline on test_topics (same code as bulk train, different topics + output file)
      # Then run both judges on test answers
      # Caching with FORCE_RERUN_GENERATION / FORCE_RERUN_JUDGING flags
[MD]  ## 5.2 Final Results Tables
[PY]  # Table 7: train vs test — alignment label dist + entailment label dist + % constraint compliance
      # Table 8: per-query breakdown test set (top-5 / bottom-5), with Phase 1 nDCG
[MD]  ## 5.3 (Optional) vLLM vs GPT-4o Generator Comparison
      Since GPT-4o is the primary generator, compare with vLLM on 5-10 test queries.
[MD]  ## 5.4 Phase 2 Summary & Locked Config for Phase 3
[PY]  P2_LOCKED_CONFIG = {
          "retriever":            P2_RETRIEVER,
          "top_k_docs":           P2_RETRIEVAL_TOP_K,
          "cross_encoder":        CROSS_ENCODER_MODEL,
          "top_n_sentences":      TOP_SENTENCES_PER_DOC,
          "generator":            GPT4O_MODEL,
          "generator_url":        IAEDU_URL,
          "judge":                GPT4O_MODEL,
          "pct_required_test":       ...,  # fill from Table 7
          "pct_supported_test":      ...,  # fill from Table 7
          "pct_constraints_met_test": ..., # fill from Table 7
      }
      print(json.dumps(P2_LOCKED_CONFIG, indent=2))
```

**Confirm notebook output:**
- Table 7: clean 2-row table (train/test), all values filled, includes % All Constraints Met.
- Table 8: 10-row table (top-5 + bottom-5), Phase 1 nDCG column present.
- `P2_LOCKED_CONFIG` printed as formatted JSON.
- All results files exist in `results/phase2/`.

---

## Quick Test Protocol (run at the start of each step)

For each step, before running notebook cells with real data:
1. Set `SAMPLE_SIZE = 5` (5 topics, use only first 50 corpus docs for sentence splitting).
2. Run the relevant `__xxx_test.py` file: `python -m src.reranking.__reranking_test`, etc.
3. Confirm all assertions pass.
4. Run the notebook cell — confirm no errors, output looks right.
5. Reset `SAMPLE_SIZE = None` for full run.

---

## File Creation Order (by step)

| Step | Notebook section           | New src files                                                                                  |
| ---- | -------------------------- | ---------------------------------------------------------------------------------------------- |
| 1    | 0. Setup                   | none                                                                                           |
| 2    | 2. Transformer internals   | `src/analysis/__init__.py`, `transformer_inspector.py`, `attention_plots.py`                   |
| 3    | 1. Cross-encoder           | `src/reranking/__init__.py`, `cross_encoder.py`, `sentence_splitter.py`, `__reranking_test.py` |
| 4    | 3. Context                 | `src/generation/__init__.py`, `llm_client.py`, `context_builder.py`                            |
| 5    | 3. Generator + parser      | `answer_generator.py`, `answer_parser.py`, `__generation_test.py`                              |
| 6    | 3. Bulk generation         | none (inline in notebook)                                                                      |
| 7    | 4. Judge setup             | `src/judging/__init__.py`, `llm_judge.py`, `__judging_test.py`                                 |
| 8    | 4. Bulk judging + analysis | none (inline in notebook)                                                                      |
| 9    | 5. Test set + summary      | none                                                                                           |

---

## `.env` additions needed

```bash
VLLM_URL=https://amalia.novasearch.org/vlm/v1
VLLM_KEY=amalia012026
IAEDU_URL=https://...          # from iaedu.pt API info page
IAEDU_KEY=...                  # from iaedu.pt API info page
```

Add to `.env.example` with placeholder values (never commit real keys).

---

## `requirements.txt` additions needed

```
nltk>=3.8
bertviz>=1.4.0
```

Run after updating:
```
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
