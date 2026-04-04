# TODO — BioMedical NL Agents Project Tasks
**Course:** NLP & Search 2025/2026 | **Based on:** TREC BioGen 2024/2025
**Deadlines:** Phase 1 → Apr 13 | Phase 2 → May 4 | Phase 3 → Jun 1

> Legend: ✅ Done · 🔲 To Do · 📓 Goes in notebook cell · 🐍 Python src file

---

## PHASE 1 — Search & Evaluation (Due Apr 13)

### 1.1 Environment & Setup
- ✅ 🔲 Set up conda env `cnn` (Python 3.10.19), install `requirements.txt`
- ✅ 🔲 Create `.env` with OpenSearch + vLLM credentials (never commit)
- ✅ 🔲 Confirm `api.novasearch.org:443` is reachable, health check passes
- ✅ 🔲 Confirm index name = your username (`usernlp03`)
- ✅ 🔲 Set up notebook Colab/local detection pattern (`IN_COLAB`, autoreload, dotenv)

### 1.2 Data Loading & Inspection 📓 🐍
- ✅ 🐍 `src/data/loader.py` — load `BioGen2024topics.json` (65 topics: id, topic, question, narrative)
- ✅ 🐍 `src/data/loader.py` — load `filtered_pubmed_abstracts.txt` (4194 JSONL docs: id, contents)
- ✅ 📓 Inspect data: show sample topic fields, sample document, count docs & topics
- ✅ 📓 Describe query fields: `topic`, `question`, `narrative` — what they contain
- ✅ 📓 Describe document structure: PMID as `doc_id`, concatenated title+abstract as `contents`

### 1.3 Query/Data Split 🐍 📓
- ✅ 🐍 `src/data/splitter.py` — split 65 topics: odd IDs → train (~33), even IDs → test (~32)
- ✅ 📓 Verify split: no ID overlap, correct counts, save to `results/splits/train_queries.json` + `test_queries.json`
- ✅ 📓 Show train vs test topic sample

### 1.4 Ground Truth / QRels 🐍 📓
- ✅ 🐍 `src/data/qrels_builder.py` — parse `biogen_2024_submissions.json` (evidence_relation field)
- ✅ 🐍 Build binary qrels: `supporting` → 1, all others → 0 → save `results/qrels/qrels.json`
- ✅ 🐍 Build graded qrels: `supporting` → 2, `neutral` → 1, all others → 0 → save `results/qrels/qrels_graded.json`
- ✅ 📓 Show qrels stats: distribution of evidence_relation values, topics with 0 relevant docs (edge cases)
- ✅ 📓 Document limitation: qrels derived from automated system citations, not human judgments → noisier than standard TREC

### 1.5 OpenSearch Index Creation 🐍 📓
- ✅ 🐍 `src/indexing/opensearch_client.py` — connect, health check, index-exists check
- ✅ 🐍 `src/indexing/index_builder.py` — define full index mapping (one index, multiple fields):
  - `contents` field → BM25 similarity (default)
  - `contents_lmjm_01` field → LM Jelinek-Mercer λ=0.1
  - `contents_lmjm_07` field → LM Jelinek-Mercer λ=0.7
  - `contents_lmdir` field → LM Dirichlet μ=2000
  - `embedding` field → KNN vector (768-dim, HNSW, innerproduct, faiss)
  - `standard` analyzer (no stemming — preserves medical terms)
- ✅ 🐍 `src/indexing/document_indexer.py` — bulk index 4194 docs (idempotent: check doc count first)
- ✅ 📓 Verify: doc count == 4194, all fields present, index healthy

### 1.6 Dense Embedding Encoder 🐍 📓
- ✅ 🐍 `src/embeddings/encoder.py` — `msmarco-distilbert-base-v2`, mean pooling + L2 norm (Lab01 pattern)
- ✅ 🐍 `src/embeddings/corpus_encoder.py` — encode all 4194 docs, save `embeddings/msmarco.npy`
- ✅ 📓 Verify: shape (4194, 768), all L2 norms ≈ 1.0, semantic sanity check (similar docs closer)
- 🔲 📓 Optional: also encode with `ncbi/MedCPT-Query-Encoder` → save `results/embeddings/medcpt.npy` for comparison

### 1.7 Retrieval — 5 Strategies 🐍 📓
- ✅ 🐍 `src/retrieval/base.py` — `BaseRetriever` interface: `search(query, size=100) → list[(pmid, score)]`
- ✅ 🐍 `src/retrieval/bm25.py` — `match` on `contents` field, `size=100` (critical: not default 10)
- ✅ 🐍 `src/retrieval/lm_jelinek_mercer.py` — `match` on `contents_lmjm_01` and `_07`, tune λ on train
- ✅ 🐍 `src/retrieval/lm_dirichlet.py` — `match` on `contents_lmdir` field
- ✅ 🐍 `src/retrieval/knn.py` — encode query on the fly → `knn` query on `embedding` field, `size=100`
- ✅ 🐍 `src/retrieval/rrf.py` — Reciprocal Rank Fusion of BM25 + KNN (k=60, Cormack 2009)
- ✅ 📓 Run each strategy on a sample query, inspect top-10 results — sanity check scores & PMIDs
- ✅ 📓 Save all run files: `results/phase1/bm25_run.json`, `lm_jm_run.json`, `lm_dir_run.json`, `knn_run.json`, `rrf_run.json`

### 1.8 Query Field Tuning 📓
- ✅ 📓 Experiment with query fields on train set:
  - Single field: `topic` only
  - Single field: `question` only
  - Concatenated: `topic + question + narrative`
- ✅ 📓 Compare MAP on train set for each query formulation
- ✅ 📓 Fix best query field → use for all strategies on test set
- 🔲 📓 Optional: LLM-based query expansion (biomedical synonyms) → compare vs original queries

### 1.9 Hyperparameter Tuning (Train Set Only) 📓
- ✅ 📓 LM-JM: compare λ=0.1 vs λ=0.7 on train set → pick winner (no re-indexing needed)
- 🔲 📓 LM-Dirichlet: optionally tune μ (try μ=1000, 2000, 3000) on train set
- 🔲 📓 RRF: optionally tune k parameter (try k=20, 60, 100) on train set
- ✅ 📓 Document all tuning decisions → fixed before test evaluation

### 1.10 Evaluation Implementation 🐍 📓
- ✅ 🐍 `src/evaluation/metrics.py` — implement in plain Python/numpy (Lab03 pattern, no ranx):
  - `P@k` (Precision at k)
  - `R@k` (Recall at k = 10, 100)
  - `AP` (Average Precision)
  - `MAP` (Mean Average Precision across queries)
  - `MRR` (Mean Reciprocal Rank)
  - `NDCG` (using graded qrels)
  - 11-point interpolated PR curve per query
- ✅ 🐍 `src/evaluation/plots.py` — PR curves, comparison tables, boxplots (matplotlib)
- 🔲 📓 Verify with Lab03 toy example (AP(A)=1.0, AP(B)=0.7095, etc.)

### 1.11 Final Evaluation — All Strategies on Test Set 📓
- ✅ 📓 Compute P@10, R@100, MAP, MRR, NDCG for each of 5 strategies on test queries
- ✅ 📓 Build comparison table: all 5 strategies × all metrics
- ✅ 📓 Plot mean PR curves for all 5 strategies on one chart (with variance shading)
- ✅ 📓 PR curve analysis — 3 specific queries must be discussed:
  - Query with highest AP
  - Query with lowest AP
  - One additional query for comparison
- ✅ 📓 Compute mAP across all test queries (final summary metric)
- ✅ 📓 Save `results/phase1/final_eval_summary.json`

### 1.12 Error & IDF Analysis 📓
- ✅ 📓 Identify worst-performing queries — analyse why (few relevant docs? vocabulary mismatch?)
- ✅ 📓 IDF analysis: which query terms are rare vs common in corpus?
- 🔲 📓 Significance tests between BM25 vs best model (optional but good for report)

### 1.13 Report Writing — Phase 1 Section 📓
- 🔲 📓 Introduction: briefly describe PubMed corpus, BioGen task, evaluation setup
- 🔲 📓 Experimental Setup: datasets, metrics, train/test split, query formulations, tuned hyperparams
- 🔲 📓 Results & Discussion: comparison table + PR curves + per-query analysis
- 🔲 📓 Limitations: qrel quality (automated, not human), corpus size, topics with 0 relevant docs

---

## PHASE 2 — RAG & LLM-Judges (Due May 4)

### 2.1 Cross-Encoder Reranking 🐍 📓
- 🔲 🐍 `src/reranking/cross_encoder.py` — load `ncbi/MedCPT-Cross-Encoder` (or `BioBERT` / `ms-marco-MiniLM-L-6-v2`)
- 🔲 🐍 Take top-N results from best Phase 1 retriever, re-score each (query, doc) pair jointly
- 🔲 🐍 Extract top-3 most relevant **sentences** per document (by cross-encoder score) → these become LLM context
- 🔲 📓 Evaluate reranked results: compare P@10, MAP, MRR before/after reranking on test set
- 🔲 📓 Save `results/phase2/reranked_run.json`

### 2.2 BERT Embeddings & Attention Visualization (Required Exercise) 📓
- 🔲 📓 **Positional Embeddings exercise (required, verbatim from guide):**
  - Insert same word repeated 200 times
  - Compute distance of all tokens to first token
  - Plot in 2D with color-code indexed to distance from first token
  - Produce full pairwise distance matrix heatmap
- 🔲 📓 **Contextual embeddings exercise:**
  - Visualize word embeddings from layer 0 to layer 11
  - Show how representations change across layers (early = syntax, late = semantics)
  - Use a word in different senses (e.g. "bank") — show context-dependent representations
- 🔲 📓 **Self-attention visualization:**
  - Examine self-attention weights in a cross-encoder on a (query, doc) pair
  - Show attention matrix heatmaps per head / per layer
  - Critical analysis: which tokens attend to which? Does model focus on query terms in doc?
- 🔲 📓 Discussion: what do you observe? Connect to why cross-encoders outperform bi-encoders

### 2.3 Reference Sentence Selection 🐍 📓
- 🔲 🐍 For each retrieved document abstract, split into sentences
- 🔲 🐍 Use cross-encoder to score each (query, sentence) pair
- 🔲 🐍 Select top-3 reference sentences per article as the grounding context for generation
- 🔲 📓 Show examples: query + top-3 selected sentences for a few topics

### 2.4 Answer Generation 🐍 📓
- 🔲 🐍 `src/generation/answer_generator.py` — prompt builder + vLLM call + citation injection
  - LLM: vLLM at `amalia.novasearch.org/vlm/v1` via `openai` client (Lab02 pattern)
  - Query server at startup to get available model name
  - Include selected reference sentences + PMIDs in prompt
  - Ask model to cite PMIDs inline per answer sentence
- 🔲 🐍 `src/generation/answer_validator.py` — enforce hard constraints:
  - ≤ 2500 words total answer length
  - ≤ 3 PMIDs cited per sentence
  - All cited PMIDs must be from the valid retrieved set (no hallucinated PMIDs)
- 🔲 📓 Show sample generated answer (with inline citations) for a few test topics
- 🔲 📓 Save `results/phase2/generated_answers.json`

### 2.5 LLM-as-Judge Evaluation 🐍 📓
- 🔲 🐍 `src/judge/prompts.py` — judge prompt templates (sentence relevance + citation entailment)
- 🔲 🐍 `src/judge/llm_judge.py` — GPT-4o via IAedu API (`https://iaedu.pt/pt`), NOT vLLM
  - Use `openai` client pointing to IAedu endpoint
  - Structured JSON output (Lab02 `response_format={"type": "json_object"}` pattern)
- 🔲 📓 **Sentence relevance judgment:** does each answer sentence address the biomedical query?
- 🔲 📓 **Citation entailment judgment:** does cited PMID's text support the answer sentence?
- 🔲 📓 Calibrate prompts on 5-10 manual examples first — verify GPT-4o agrees with human judgment
- 🔲 📓 Report aggregate judgment statistics across all test topics
- 🔲 📓 Save `results/phase2/judge_labels.json`

### 2.6 Phase 2 Report Section 📓
- 🔲 📓 Describe reranking approach and model choice (MedCPT vs alternatives)
- 🔲 📓 Describe generation prompt design and constraints enforcement
- 🔲 📓 Present before/after reranking evaluation table
- 🔲 📓 Present judge results: % sentences relevant, % citations entailed
- 🔲 📓 Include embedding & attention visualizations with discussion
- 🔲 📓 Limitations: judge prompt not calibrated to domain, LLM hallucinations, judge bias

---

## PHASE 3 — Deep Research Agent (Due Jun 1)

### 3.1 Agent Architecture 🐍
- 🔲 🐍 `src/agent/planner.py` — LLM decomposes a topic into N sub-topics
  - Input: biomedical topic string
  - Output: list of sub-topic strings (e.g. "weight loss", "side effects", "long-term outcomes")
  - LLM: vLLM at `amalia.novasearch.org`
- 🔲 🐍 `src/agent/explorer.py` — ReAct loop per sub-topic:
  - **Reason:** "I need evidence about X..."
  - **Act (SEARCH):** generate a query string
  - **Observe:** call Phase 1 retriever → get docs → format as context
  - **Repeat:** until enough evidence OR max iterations (e.g. 5–8 steps hard cap)
  - Use strict XML/JSON structured format for LLM actions (not free text)
  - Log warning if iteration cap is hit
- 🔲 🐍 `src/agent/aggregator.py` — merge evidence across all sub-topics:
  - Deduplicate PMIDs
  - Validate all citations exist in corpus
  - Resolve conflicting evidence (same PMID cited for multiple sub-topics)
- 🔲 🐍 `src/agent/report_writer.py` — structured final report:
  - Section headers per sub-topic
  - Inline PMID citations per sentence
  - Applies same ≤2500 word / ≤3 PMID constraints as Phase 2

### 3.2 ReAct Loop Design 📓
- 🔲 📓 Show the reasoning trace for 1-2 topics (reason → search → observe steps logged)
- 🔲 📓 Show final aggregated evidence before report writing
- 🔲 📓 Compare ReAct agent report vs simple single-query answer (Phase 2)

### 3.3 Agent Evaluation 📓
- 🔲 📓 Run agent on a subset of test topics
- 🔲 📓 Apply LLM-judge (Phase 2) to assess agent-generated report sentences
- 🔲 📓 Compare coverage: agent report covers more sub-aspects than single-shot answer?
- 🔲 📓 Measure: # sub-topics explored, # unique PMIDs cited, iterations per sub-topic
- 🔲 📓 Save `results/phase3/agent_reports.json`

### 3.4 Phase 3 Report Section 📓
- 🔲 📓 Describe ReAct agentic pattern (plan → browse → synthesize loop)
- 🔲 📓 Show example agent execution trace (reasoning steps visible)
- 🔲 📓 Evaluation results + comparison vs Phase 2 single-pass RAG
- 🔲 📓 Limitations: sub-topic quality depends on planner LLM, iteration cap tradeoff

---

## REPORT — Incremental Throughout All Phases

### Structure
- 🔲 1. Introduction — BioGen task, hallucination problem, project overview
- 🔲 2. BioMedical NL Agent
  - 🔲 2a. Data parsing, indexing and search (Phase 1)
  - 🔲 2b. LLM Augmented Generation (Phase 2)
  - 🔲 2c. LLM Agentic Patterns (Phase 3)
- 🔲 3. Evaluation
  - 🔲 3a. Experimental Setup: Datasets, Metrics, Protocols
  - 🔲 3b. Results and Discussion (tables, curves, qualitative analysis)
- 🔲 4. Conclusion
  - 🔲 4a. Achievements
  - 🔲 4b. Limitations

---

## OPTIONAL ENHANCEMENTS (Bonus / If Time Allows)

- 🔲 MedCPT bi-encoder comparison vs msmarco in Phase 1 (already encoded in `results/embeddings/medcpt.npy`)
- 🔲 LLM query expansion before retrieval (biomedical synonyms via vLLM) — +~10 lines, big discussion point
- 🔲 LM-Dirichlet μ tuning (μ=1000, 2000, 3000) on train set
- 🔲 RRF with 3 or 4 systems (BM25 + LM-JM + KNN + LM-Dir) instead of just 2
- 🔲 Significance testing between BM25 and best model
- 🔲 t-SNE / UMAP visualization of document embedding space (colored by topic)
- 🔲 Answer faithfulness metric: fraction of answer sentences with ≥1 supporting citation

---

## DOUBTS & QUESTIONS TO CLARIFY (For a 20/20 Grade)

### Phase 1
- ❓ **Query field combination:** The guide says "combine topic+question+narrative into one string" as the recommended approach. Should we present each field separately first (ablation), then combined? Or jump straight to combined?
- ❓ **`size` parameter in KNN query:** OpenSearch KNN queries use `k` not `size` — is `k=100` enough, or should we use a larger `k` and trim to 100?
- ❓ **Topics with 0 relevant docs in qrels:** Some train queries may have no relevant PMIDs in `biogen_2024_submissions.json`. Should they be excluded from MAP/NDCG calculation? (Standard TREC practice: yes, skip and log WARNING.)
- ❓ **Graded relevance threshold for P@k:** Binary P@k uses score ≥ 1 (supporting OR neutral). Should binary P@k use only `supporting` (score=2)? Clarify with professor.
- ❓ **RRF fusion components:** Guide says "BM25 + KNN" for RRF. Should we also include LM-Dir or LM-JM in the fusion for potentially better results?
- ❓ **NDCG cut-off:** Should we compute NDCG@10 or NDCG@100? Both? What is the required cut-off for the report?
- ❓ **PR curves for individual queries:** The guide requires 3 specific PR curves discussed in depth. Do they need to be on train or test set?

### Phase 2
- ❓ **Answer length limit:** Guide v1 said "250 words", new guide says "2500 words". Confirm with professor which is correct.
- ❓ **Cross-encoder model:** MedCPT vs ms-marco vs BioBERT — does it matter which we use as primary? Is comparison between models required or optional?
- ❓ **Sentence selection:** "Top-3 reference sentences per article" — top-3 across ALL articles (giving 300 sentences if 100 docs) or top-3 per article for only the top-N (e.g. top-10) retrieved articles?
- ❓ **IAedu API quota:** Is there a rate limit or cost limit on GPT-4o calls via IAedu? How many topics can we judge before hitting limits?
- ❓ **Judge prompt calibration:** Should the calibration examples (5-10 manual labels) be included in the report as an appendix?
- ❓ **What "good" judge agreement looks like:** What precision/recall of the LLM judge vs manual labels is acceptable for this project?

### Phase 3
- ❓ **ReAct action format:** XML vs JSON? Is there a preference? Must it be strict structured output or is the agent allowed some free-form reasoning?
- ❓ **Number of sub-topics:** How many sub-topics should the planner decompose a topic into? 2–5? More? Any guidance?
- ❓ **Agent evaluation metric:** The guide says "more details will be provided." What is the expected formal metric for Phase 3? (Judge score? Coverage? Something else?)
- ❓ **Does Phase 3 build on Phase 2 output?** I.e. does the agent use the cross-encoder reranker from Phase 2, or just raw Phase 1 retrieval?

### General / Report
- ❓ **Report format:** Is there a page limit? LaTeX or PDF or notebook format? Can figures be in the notebook or must they be exported to a separate document?
- ❓ **Code submission:** Do we submit the GitHub repo link or zip? Are the Jupyter notebooks the main deliverable?
- ❓ **Run file format:** Is the TREC run format (text) required, or is JSON fine for submission?
- ❓ **Shared OpenSearch index:** If multiple students use the same server, could index name conflicts occur? Is cleanup needed after the course?
- ❓ **vLLM model name:** The server's available models are queried at startup. What if the model changes between Phase 2 and Phase 3? Should we hardcode a model version or always query?
