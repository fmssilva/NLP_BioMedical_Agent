# Project Plan — BioMedical NLP Agent (TREC BioGen 2025/2026)

---

## Big Picture Intuition

We are building a **biomedical information retrieval and answer generation agent**, modelled on the TREC BioGen shared task. The problem being solved is this: a doctor or researcher asks a biomedical question (e.g., *"Are there natural treatments for sleep apnea?"*) and our system must (1) find the most relevant PubMed abstracts, (2) generate a grounded, evidence-citing answer, and (3) eventually orchestrate an autonomous research agent that plans sub-topics, browses evidence, and synthesises a nuanced final report — all while minimising hallucination.

The project is built in **three sequential phases**, each one building on the previous. Each phase delivers both code (a Jupyter notebook) and a growing project report.

---

## The Dataset

- **Corpus:** ~100k PubMed abstracts (title + abstract + PMID), stored in a JSONL file, indexed in OpenSearch.
- **Query-topics:** Biomedical questions structured as `{id, topic, question, narrative}`. The query can use any combination of these three fields.
- **Ground truth:** `citation_assessment` labels (`required`, `supporting`, `borderline`) per (query, PMID) pair, mapped to numeric relevance grades for nDCG. Split into training topics (odd IDs) and test topics (even IDs).
- **Reference answer submissions from BioGen 2024** are also available as a gold reference for judging.

---

## Phase 1 — Search and Evaluation  *(deadline: April 13, 2026 — DONE)*

**Goal:** Given a query topic, retrieve the most relevant PubMed abstracts.

**What we do:**
- Parse and index all PubMed abstracts into OpenSearch with multiple similarity models configured per-field: BM25, LM Jelinek-Mercer, LM Dirichlet, and dense KNN with sentence embeddings.
- Implement four retrieval strategies (BM25, LM-JM, LM-Dir, dense KNN) and a fusion strategy (RRF — Reciprocal Rank Fusion) that combines any pair of them.
- Tune hyperparameters (BM25 k1/b, LM-JM lambda, LM-Dir mu, dense encoder choice, RRF pair selection) on the **training set** using cross-validation.
- Evaluate on the **test set** using: P@10, R@100, nDCG, mAP, precision-recall curves (including per-query curves for best/worst/one-more queries).
- Lock down the best retrieval configuration to be reused in Phase 2.

**Key insight:** BM25 is the strong sparse baseline; dense KNN captures semantic similarity; RRF fusion combines their complementary strengths. Language model smoothing methods (JM, Dirichlet) control how much to fall back to corpus-wide term frequencies — important for short biomedical queries.

---

## Phase 2 — Factually Grounded RAG  *(deadline: May 4, 2026 — CURRENT)*

**Goal:** Given the retrieved documents from Phase 1, generate a factually grounded answer with cited PMIDs.

**What we do:**

### 2a — Reference Sentence Selection (Cross-Encoder Re-ranking)
Each retrieved abstract contains multiple sentences. We use a **cross-encoder** (MedCPT-Cross-Encoder or BioBERT fine-tuned on biomedical NLI) to score each (query, sentence) pair and select the **top-3 most relevant sentences per abstract**. This filters noise and focuses generation on the highest-signal evidence.

### 2b — Embedding and Attention Visualisation (Analysis Exercise)
We perform an in-depth analysis of the transformer model used for encoding:
- **Positional embeddings:** Visualise how position information is encoded — distance from token 0, and pairwise distance matrix.
- **Contextual embeddings:** Observe how word representations evolve across layers 0–11, from syntactic to semantic.
- **Self-attention:** Examine the attention heads of a cross-encoder — which tokens attend to which, and what patterns emerge.

This is both an analysis exercise and a tool for understanding *why* the cross-encoder makes certain re-ranking decisions.

### 2c — Answer Generation (RAG with Attribution)
Using the selected reference sentences (top-3 per doc, top-k docs from Phase 1 retrieval), we prompt a local vLLM (served at `amalia.novasearch.org`) to generate a **≤250-word answer** where each sentence is attributed to ≤3 PMIDs from the valid corpus. The prompt enforces factual grounding and citation format.

### 2d — LLM-as-a-Judge Evaluation
We use **GPT-4o** (via the IAedu API) as an automated evaluator because human annotation is too expensive at scale. Two evaluation axes:
- **Sentence alignment:** Is each selected reference sentence relevant to the query? (Scored per sentence)
- **Answer entailment:** Is the generated answer fully entailed by its cited reference sentences? (Scored per answer)

We manually verify and adapt the judge prompts on a 5–10 example sample to calibrate them for the biomedical domain before running at scale.

**Key insight:** RAG (Retrieval-Augmented Generation) grounds the LLM in evidence, reducing hallucination. The cross-encoder re-ranking step is critical — it selects the most relevant sentences as generation context. The LLM judge closes the evaluation loop without requiring human annotation.

---

## Phase 3 — Deep Research Agent  *(deadline: June 1, 2026)*

**Goal:** Build an autonomous agent that plans, retrieves, and synthesises a nuanced research report.

**What we do:**
- Implement a **ReAct-style agent** (Reason + Act loop) that takes a complex biomedical topic and:
  1. **Plans** sub-topics (e.g., "Weight loss", "Side effects", "Long-term outcomes").
  2. **Browses** — executes Phase 1+2 cycles for each sub-topic independently.
  3. **Synthesises** — aggregates conflicting evidence into a final structured report with proper citations.
- Evaluate the agent's outputs with the same LLM-judge framework from Phase 2, extended to cover multi-step reasoning quality and report coherence.

**Key insight:** A single RAG call cannot handle complex multi-faceted questions well. By decomposing the question into sub-topics and running independent evidence gathering per sub-topic, the agent produces more complete and nuanced answers — at the cost of higher compute and latency. The planning step is the hardest part (it determines quality).

---

## Important Technical Decisions (Fixed Across All Phases)

| Decision                | Choice                                                               | Reason                                                                           |
| ----------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Search engine           | OpenSearch (hosted)                                                  | Course infrastructure, supports BM25/LM/KNN natively                             |
| Evaluation library      | `ranx`                                                               | Standard TREC-compatible, supports all required metrics                          |
| Dense encoder           | MedCPT-Article-Encoder (locked from Phase 1 tuning)                  | Best nDCG on biomedical topics in our CV sweep                                   |
| Cross-encoder (Phase 2) | MedCPT-Cross-Encoder                                                 | Same family, NCBI-trained on PubMed, specifically tuned for biomedical retrieval |
| Generation LLM          | vLLM at `amalia.novasearch.org` (local)                              | Low latency, no API cost for generation at scale                                 |
| Judge LLM               | GPT-4o via IAedu API                                                 | Frontier capability needed for entailment judgement                              |
| Query field             | `topic + question + narrative` combined (locked from Phase 1 tuning) | Showed best retrieval performance                                                |
| Train/test split        | Odd IDs = train, Even IDs = test                                     | As per project specification                                                     |

---

## Report Structure (Incremental, 15 pages max)

The report grows each phase:
- **Cover + Introduction** (1 page)
- **BioGen NL Agent** (3 pages: 1 per phase)
- **Evaluation** (9 pages: setup + results per phase)
- **Conclusion** (2 pages: achievements + limitations)

---

## Deliverables per Phase

| Phase | Code                                         | Report Sections Added                                                |
| ----- | -------------------------------------------- | -------------------------------------------------------------------- |
| 1     | `tasks/phase1_search.ipynb` + `src/` modules | Intro, Phase 1 system description, Phase 1 eval setup + results      |
| 2     | `tasks/phase2_rag.ipynb` + `src/` modules    | Phase 2 system description, Phase 2 eval setup + results             |
| 3     | `tasks/phase3_agent.ipynb` + `src/` modules  | Phase 3 system description, Phase 3 eval setup + results, Conclusion |
