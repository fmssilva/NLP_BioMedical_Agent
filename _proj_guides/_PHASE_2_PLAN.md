# Phase 2 Plan — Factually Grounded RAG
### Notebook: `tasks/phase2_rag.ipynb`

---

## Overview

Phase 2 takes the **locked retrieval configuration from Phase 1** and builds a full
Retrieval-Augmented Generation (RAG) pipeline on top of it: re-rank retrieved sentences
with a cross-encoder, generate a ≤250-word grounded answer, and evaluate both the
sentence selection and the final answer with a GPT-4o LLM-judge.

There is also a standalone **analysis exercise** on transformer internals (positional
embeddings, contextual embeddings, self-attention) that demonstrates *why* cross-encoders
work the way they do.

---

## !!!!! DOUBTS — RESOLVED

**D1 — Cross-encoder model ✅ RESOLVED**
> **Decision:** `ncbi/MedCPT-Cross-Encoder` as primary model (biomedical domain match).
> Include `cross-encoder/ms-marco-MiniLM-L-6-v2` as a comparison baseline in the Section 1.7 ablation only.
> **Note:** Verify output shape on first load — some HuggingFace cross-encoders output 2-class logits `[irrelevant, relevant]` and you need `logits[:, 1]` (positive class), not `logits[:, 0]`. Add assertion in smoke test.

**D2 — Sentence segmentation strategy ✅ RESOLVED**
> **Decision:** Use `nltk.sent_tokenize` (punkt). Simple, fast, no heavy dependencies.
> scispaCy (`en_core_sci_sm`) adds a 500MB+ model download for marginal improvement on PubMed text.
> Note scispaCy as a potential improvement in the report discussion section.
> **Note:** NLTK ≥3.9 renamed the resource from `punkt` to `punkt_tab`. Download both to be safe:
> `nltk.download('punkt'); nltk.download('punkt_tab')`

**D3 — Number of retrieved docs passed into RAG ✅ RESOLVED**
> **Decision:** Make `top_k` a configurable argument (not hardcoded). Default to 10 for Phase 2.
> In Phase 3, can sweep different values (e.g., 5, 10, 20, 30) to confirm optimal k.
> This is easy to parameterise since the retriever already supports `size=k`.

**D4 — Generation LLM ✅ RESOLVED**
> **Decision:** Use GPT-4o via IAedu API for both generation AND judging.
> vLLM at `amalia.novasearch.org` kept as a secondary option for the optional comparison in §5.3.
> IAedu API credentials provided by user (see `.env` for keys).

**D5 — Answer generation prompt template ✅ RESOLVED**
> **Decision:** Start with zero-shot (Option A). Design the function so it's flexible —
> easy to add few-shot examples later by extending the `messages` list.
> Can also compare zero-shot vs few-shot in Phase 3 to confirm best results.

---

## Notebook Structure

```
tasks/phase2_rag.ipynb
```

The notebook is organised into **5 top-level sections**, each self-contained and
narrated as a learning tool. Every section starts with a markdown cell explaining the
concept, then the code cells, then a markdown analysis cell.

---

---
# **0. Project Set Up** [DONE]
*Infrastructure, constants, and data loading — same pattern as Phase 1.*

---
## 0.1 Constants & Config [DONE]

**[MARKDOWN]** Brief description of what constants we define and why (paths, model names, k values, API endpoints).

**[CODE]** Define all constants:
- Paths: corpus, topics, qrels, results dir, phase1 run files.
- Phase 1 locked config: best retriever name, best encoder name, top-k docs.
- Phase 2 config: cross-encoder model name, top-n sentences per doc, max answer words, max PMIDs per sentence.
- API endpoints: vLLM base URL, IAedu GPT-4o base URL + API key.

> Priority: **10** — Without this nothing else runs. Every constant in one place.

---
## 0.2 Imports & Environment [DONE]

**[CODE]** All imports: `opensearch-py`, `transformers`, `torch`, `openai`, `nltk`, `bertviz` (for attention viz), `matplotlib`, `seaborn`, `pandas`, standard libs.
Check GPU availability (`torch.cuda.is_available()`), print device info.
Download NLTK tokenizer data: `nltk.download('punkt'); nltk.download('punkt_tab')` (both needed for NLTK ≥3.9 compatibility).

> Priority: **10** — Sanity check that all packages are installed before spending time running cells.

---
## 0.3 Data Loading [DONE]

**[MARKDOWN]** Brief recap of the dataset: corpus structure, topics structure, qrels. Remind the reader we reuse Phase 1 data loading — no new data.

**[CODE]** Load:
- `corpus`: list of `{pmid, title, abstract}` dicts — use `src.data.loader.load_corpus`.
- `corpus_lookup`: dict `{pmid: contents}` for O(1) abstract access — build as `{doc["id"]: doc["contents"] for doc in corpus}`.
- `valid_pmids`: set of all corpus PMIDs — build as `{doc["id"] for doc in corpus}`.
- `topics`: train + test splits — use `src.data.loader.load_topics`.
- `qrels`: train + test — use `src.data.qrels_builder`.
- **Phase 1 best run file** (the locked top-k retrieval results per query to feed into Phase 2).
Print: corpus size, topic counts, qrels stats.

> Priority: **10** — Everything downstream depends on this.

---
## 0.4 Phase 1 Retrieval Recap (Locked Config) [DONE]

**[MARKDOWN]** One short paragraph: what was the best retrieval configuration from Phase 1 (model, parameters, nDCG score). This anchors Phase 2 to Phase 1 results. Add a small table: Phase 1 best metrics (P@10, R@100, nDCG).

**[CODE]** Load Phase 1 best run file (JSON). Show: first 3 query results as a quick sanity check (query text → top-3 PMIDs + scores).

> Priority: **8** — Sets context for why we are using this particular retrieval output. Important for the report and for the reader.

---

---
# **1. Cross-Encoder Re-Ranking** [DONE]
*Given retrieved abstracts, score each sentence for relevance to the query using a cross-encoder. Select the top-3 sentences per document as reference evidence.*

---
## 1.1 Cross-Encoder: Concept & Model [DONE]

**[MARKDOWN]** Explain the difference between bi-encoders (Phase 1 KNN — encode query and doc separately, compare with dot product) vs cross-encoders (encode query+doc *jointly*, allowing full attention across both — slower but more accurate relevance score). Show the formula for cross-encoder scoring. Explain why MedCPT-Cross-Encoder was trained on PubMed and why that matters for biomedical re-ranking.

> Priority: **9** — This is core conceptual content for the notebook-as-learning-tool goal. Reader needs to understand *why* we switched from bi-encoder to cross-encoder here.

---
## 1.2 Load Cross-Encoder Model [DONE]

**[CODE]** Load `ncbi/MedCPT-Cross-Encoder` (or alternative, see D1):
- Use `AutoTokenizer` + `AutoModelForSequenceClassification`.
- Print model architecture summary (num layers, hidden dim, num attention heads).
- Quick smoke test: score one (query, sentence) pair, confirm output is a scalar logit.

> Priority: **10** — Foundation for all sentence re-ranking.

---
## 1.3 Sentence Segmentation [DONE]

**[MARKDOWN]** Explain the motivation: abstracts are multi-sentence, not every sentence is relevant to the query. We need to split and score individually. Note the challenge: biomedical abbreviations ("i.v.", "mg/dL", "et al.") trip up naive sentence splitters.

**[CODE]**
- Implement `split_into_sentences(text: str) -> list[str]` using `nltk.sent_tokenize` (punkt).
- Show example: take 1 abstract, display its sentences with indices.
- Show edge cases: abbreviations, numeric values, parenthetical references — does the splitter handle them correctly?
- **Edge case:** if an abstract has 0 sentences after splitting (very short or title-only docs), return an empty list. These docs are skipped silently in the bulk pipeline.

> Priority: **8** — Sentence quality directly impacts re-ranking quality. Worth a brief analysis.

---
## 1.4 Sentence Scoring with Cross-Encoder [DONE]

**[MARKDOWN]** Describe the scoring pipeline: for each retrieved doc → split abstract into sentences → score each (query, sentence) pair with cross-encoder → keep top-3 sentences per doc.

**[CODE]** Implement `score_sentences(query: str, sentences: list[str], model, tokenizer, batch_size=16) -> list[float]`:
- Tokenise (query, sentence) pairs, forward pass, extract logit score.
- Handle batching for efficiency.
- Return scores aligned to input sentences list.

Implement `select_top_sentences(query, abstract, top_n=3, ...) -> list[tuple[str, float]]`.

> Priority: **10** — Core Phase 2 function. Everything downstream uses this.

---
## 1.5 Demo: Sentence Selection on Sample Queries [DONE]

**[MARKDOWN]** Walk through the full pipeline on 2-3 example queries from the training set. Show: query text → top-k retrieved docs → for each doc, all sentences with their cross-encoder scores → top-3 selected sentences highlighted.

**[CODE]**
- Pick query with high Phase 1 AP (best retrieval case) and query with low Phase 1 AP (hard case).
- For each: show the selected sentences in a nice formatted output (query, PMID, sentence, score).
- Add a quick assert: top-3 scores are always ≥ all other scores (sanity check on the selection logic).

> Priority: **8** — Great for the learning-tool goal. Reader sees the pipeline in action with real examples.

---
## 1.6 Cross-Encoder Score Distribution Analysis [DONE]

**[MARKDOWN]** What does the cross-encoder score distribution look like? Are relevant sentences clearly separated from irrelevant ones? This tells us how "confident" the model is.

**[CODE]**
- Run sentence scoring on a random sample of 10-20 queries (train set).
- **Plot A:** Histogram of all cross-encoder scores for sentences from relevant docs vs sentences from non-relevant docs (two overlapping histograms). Do scores separate? What is the decision threshold?
- **Plot B:** Box plot — score distribution per relevance label (`required`, `supporting`, `borderline`, none).
- **Table 1:** Mean and std of scores per relevance label.

> Priority: **7** — Diagnostic/analysis value. Shows the model is actually discriminating. Good for report.

---
## 1.7 Baseline Comparison: Random vs Cross-Encoder Sentence Selection [DONE]

**[MARKDOWN]** Quick ablation: what happens if we select sentences randomly instead of using the cross-encoder? This justifies the cross-encoder step.

**[CODE]**
- Implement `random_sentence_selection(abstract, top_n=3) -> list[str]` as baseline.
- For a small sample of queries (10-20 train), compute a simple proxy metric: what fraction of selected sentences are from relevant docs (by PMID) under each strategy?
- **Table 2:** Cross-encoder top-3 vs Random top-3 — % sentences from relevant docs, mean cross-encoder score of selected sentences.

> Priority: **6** — Nice ablation, justifies the design choice. Not critical but adds rigor.

---

---
# **2. Transformer Internals: Embeddings & Attention** [DONE]
*Analysis exercise: understand what happens inside the transformer encoder — positional embeddings, contextual evolution across layers, and self-attention patterns.*

---
## 2.1 Introduction to Transformer Internals [DONE]

**[MARKDOWN]** Explain the three components we will analyse:
1. **Positional embeddings** — how the model knows token order without recurrence.
2. **Contextual embeddings** — how word representations change as they pass through attention layers (from syntactic surface form to semantic meaning).
3. **Self-attention** — which tokens attend to which, and what patterns emerge (syntactic, semantic, positional).

Include a simple diagram of a transformer block (text/ASCII art is fine). Mention that we use a BERT-family model (MedCPT uses BERT architecture) and the visualisations apply directly to understanding the cross-encoder we use in Section 1.

> Priority: **9** — This section has the most conceptual content. The guide explicitly requires these visualisations and a "critical analysis".

---
## 2.2 Model Setup for Analysis [DONE]

**[CODE]** Load a BERT-family model for the analysis (can reuse the cross-encoder from Section 1, or load `bert-base-uncased` for cleaner demos). Configure `output_hidden_states=True` and `output_attentions=True` in the model config.

Write a helper `get_model_outputs(text, model, tokenizer) -> (tokens, hidden_states, attentions)` that returns:
- `tokens`: list of token strings (decoded from tokenizer).
- `hidden_states`: tensor `(num_layers+1, seq_len, hidden_dim)` — layer 0 is embedding layer.
- `attentions`: tensor `(num_layers, num_heads, seq_len, seq_len)`.

> Priority: **10** — All downstream analysis cells depend on this helper.

---
## 2.3 Positional Embeddings — Distance from Token 0 [DONE]

**[MARKDOWN]** Theory: In BERT, positional embeddings are learned (not sinusoidal). Each position gets a unique learned vector added to the word embedding. If we feed the same word repeated N times, the only variation between tokens is position. This isolates positional structure.

**[CODE]** 
- Build input: a common neutral word (e.g., `"the"`) repeated 200 times, truncated to model max_length.
- Extract layer-0 hidden states (pure embeddings, no attention yet) → tensor `(200, 768)`.
- Compute cosine distance from token 0 to all other tokens → vector `(200,)`.
- **Plot C:** 2D scatter of all 200 token embeddings via PCA/UMAP, color-coded by distance from token 0. Do nearby positions cluster? Is there a gradient?

> Priority: **9** — Directly required by the project guide.

---
## 2.4 Positional Embeddings — Pairwise Distance Matrix [DONE]

**[CODE]**
- Using the same 200-token input from 2.3: compute pairwise cosine distance matrix `(200, 200)`.
- **Plot D:** Heatmap of the pairwise distance matrix (seaborn `heatmap`, RdYlGn or viridis colormap). Does it look banded (nearby positions more similar)? Are there periodic patterns?

**[MARKDOWN]** Analysis: describe what the matrix tells us about how BERT encodes position. Compare to the expected pattern from sinusoidal embeddings (which have a specific correlation structure).

> Priority: **9** — Directly required by the project guide.

---
## 2.5 Contextual Embeddings — Layer-by-Layer Evolution [DONE]

**[MARKDOWN]** Theory: in layer 0 (after embedding, before attention), tokens are contextualised only by position. By layer 11, each token has attended to all others 12 times — the representation captures deep semantic and syntactic context. Visualising how a word's embedding changes layer by layer shows the "context injection" process.

**[CODE]**
- Pick a short biomedical sentence with polysemous words if possible (e.g., *"The drug treatment showed significant side effects on the trial."*).
- Extract hidden states for all 12 layers (indices 1-12 from `hidden_states` tuple).
- For each layer, project all token embeddings to 2D via PCA (fit PCA on layer 12 for a common space).
- **Plot E:** Grid of 12 small scatter plots (3×4 or 4×3), one per layer, showing token positions in 2D PCA space with token labels. Do tokens with similar semantic roles cluster together in deeper layers?

**[MARKDOWN]** Analysis: describe how representations evolve from surface-level (syntactic) in early layers to semantic in deeper layers.

> Priority: **9** — Required by the guide. Also a great visual for the report.

---
## 2.6 Self-Attention Analysis [DONE]

**[MARKDOWN]** Theory: self-attention computes, for each token, a weighted sum over all other tokens. The weights (attention scores) reflect "how much this token looks at that token". Different heads specialise: some track syntax (subject-verb), some track co-reference, some track positional adjacency. For a **cross-encoder**, attention flows between query tokens and document tokens — this is where relevance is computed.

**[CODE]**
- Pick a (query, sentence) pair where the cross-encoder gave a HIGH score (highly relevant) and another where it gave a LOW score.
- Use `bertviz` (`head_view` or `model_view`) to visualise attention for both:
  - **Plot F:** Attention head visualisation for the high-relevance pair.
  - **Plot G:** Attention head visualisation for the low-relevance pair.
- Alternative (if bertviz has rendering issues in the notebook): use matplotlib to plot the mean attention matrix `(seq_len, seq_len)` averaged across heads for layer 6 and layer 11.

**[MARKDOWN]** Critical analysis:
- Which heads seem to attend to query keywords from the sentence side?
- Is there a clear difference in attention patterns between the high-score vs low-score pair?
- What does this tell us about how the cross-encoder "decides" relevance?

> Priority: **8** — Required analysis. The "critical analysis" part is what earns the marks here — the code is secondary.

---

---
# **3. Answer Generation**
*Given the selected reference sentences, prompt the LLM to generate a ≤250-word grounded answer with cited PMIDs.*

---
## 3.1 RAG Pipeline: Concept & Design Decisions [DONE]

**[MARKDOWN]** Explain the RAG pipeline: retrieve → re-rank sentences → format context → prompt LLM → parse answer + citations. Walk through the constraints:
- ≤250 words total answer.
- ≤3 PMIDs per sentence.
- PMIDs must come from the valid corpus (no hallucinated PMIDs).
Include a diagram of the full pipeline (text diagram is fine):
```
query → Phase1 retrieval → top-k docs → sentence split → cross-encoder score → top-3/doc → format prompt → LLM → answer + PMIDs
```

> Priority: **9** — The reader needs to understand the architecture before seeing code. Also good for the report.

---
## 3.2 LLM Client Setup [DONE]

**[CODE]** Set up LLM clients:
- GPT-4o client (IAedu API) for both generation and judging — primary LLM.
- vLLM client (`openai.OpenAI(base_url=VLLM_URL, api_key=...)`) as optional secondary generator for comparison.
Test both with a simple ping (list models or single completion call). Print available model names.

> Priority: **10** — Both need to work before any generation or evaluation.

---
## 3.3 Context Formatting [DONE]

**[MARKDOWN]** How we format the reference sentences into a prompt context. Each piece of evidence should be attributed to its PMID. Show the format we use:
```
[PMID 12345678] Sentence 1 text here.
[PMID 23456789] Sentence 2 text here.
...
```

**[CODE]** Implement `format_context(selected_sentences: list[dict]) -> str`:
- Input: list of `{pmid, sentence, score}` dicts.
- Output: formatted context string with PMID labels.
- Include a check: warn if any PMID is not in the valid corpus set.

> Priority: **9** — Determines the structure the LLM sees. Critical for citation accuracy.

---
## 3.4 Generation Prompt Design [DONE]

**[MARKDOWN]** Walk through the system prompt and user prompt structure. Explain the design choices:
- Why we use a system prompt to establish the role (biomedical expert, grounded answers only).
- Why we specify the output format explicitly (numbered sentences + citations).
- Temperature=0.0 or low for factual tasks (less creative variation, more consistent citations).

Show the full prompt template with `{question}` and `{context}` placeholders filled in.

> Priority: **9** — The prompt is the most important design decision in this section. Show it clearly.

---
## 3.5 Answer Generation Function [DONE]

**[CODE]** Implement `generate_answer(question: str, context: str, client, model_name: str, ...) -> str`:
- Calls the LLM (GPT-4o by default) with the formatted prompt.
- Returns raw answer string.
- Designed as zero-shot; flexible to add few-shot examples later by extending messages list.

Implement `parse_answer(raw_answer: str) -> dict`:
- Extracts: `{answer_text, sentences: [{text, pmids}], total_words, pmid_list}`.
- Validates: word count ≤250, ≤3 PMIDs per sentence, all PMIDs in valid corpus.
- Returns a structured dict for easy downstream use.

> Priority: **10** — Core generation + parsing logic.

---
## 3.6 Demo: Generate Answers for Sample Queries [DONE]

**[MARKDOWN]** Show end-to-end pipeline for 3 example queries (same ones used in Section 1.5 for continuity): query → retrieved docs → selected sentences → generated answer.

**[CODE]**
- Run the full pipeline for 3 training queries.
- Display output in a readable format: question, then answer with PMIDs inline.
- Run `parse_answer` and confirm constraints are met (print green/red pass/fail for each constraint).

> Priority: **8** — Great for the learning tool. Shows the system working end-to-end before bulk generation.

---
## 3.7 Bulk Answer Generation — All Training Queries [DONE]

**[MARKDOWN]** Now run the full pipeline on all training queries. Note: this may take several minutes — estimated time. Show progress bar.

**[CODE]**
- Loop over all train queries, run retrieval → sentence selection → generation → parse.
- Save results to `results/phase2_train_answers.json`: `{query_id, question, selected_sentences, answer, parsed}`.
- Print: total queries processed, % that passed all constraints (word count, PMID count, valid PMIDs).

> Priority: **10** — Required for evaluation. Must complete without errors.

---
## 3.8 Constraint Violation Analysis [DONE]

**[MARKDOWN]** How often do constraints get violated? Are any systematic (e.g., the model always goes over 250 words for complex topics)?

**[CODE]**
- **Table 3:** Constraint compliance table — % queries meeting each constraint: word count ≤250, max PMIDs per sentence ≤3, all PMIDs valid.
- **Plot H:** Histogram of answer word counts across all generated answers. Mark the 250-word limit.
- **Plot I:** Distribution of PMIDs per answer sentence (bar chart).

> Priority: **7** — Diagnostic. If many violations exist, we need to fix the prompt before evaluation.

---

---
# **4. LLM-as-a-Judge Evaluation**
*Use GPT-4o as an automated evaluator to measure (1) sentence relevance alignment and (2) answer entailment against reference sentences.*

---
## 4.1 LLM-Judge: Concept & Motivation [DONE]

**[MARKDOWN]** Explain the motivation: human annotation is gold standard but expensive. LLM judges (especially frontier models like GPT-4o) have shown strong correlation with human judgements on biomedical NLP tasks. Two-level evaluation:
1. **Sentence alignment** — is each selected reference sentence relevant to the query?
2. **Answer entailment** — is the generated answer fully supported by (entailed by) its cited reference sentences?

Explain why entailment is distinct from relevance: a sentence can be relevant to the topic but the generated answer might make a stronger claim than the sentence actually supports — that's a hallucination.

> Priority: **9** — Conceptual foundation for the whole evaluation. Must be well-explained.

---
## 4.2 Judge Prompt Design — Sentence Alignment [DONE]

**[MARKDOWN]** Show the sentence alignment prompt (from the TREC BioGen project guide). Explain the scoring criteria using the guide's exact labels:
- **Required:** The answer sentence is necessary for completeness of the answer.
- **Unnecessary:** Not required — information overload, trivial, or not relevant to the question.
- **Borderline:** Relevant and possibly "good to know" but not required.
- **Inappropriate:** May harm the patient (e.g., contradictory advice).

Explain that we start with the guide's exact prompt and then calibrate on 5–10 examples.

**[CODE]** Implement `judge_sentence_alignment(query: str, sentence: str, pmid: str, gpt4o_client, ...) -> dict`:
- Returns `{label: str, pmid: str}` where label ∈ {Required, Unnecessary, Borderline, Inappropriate}.
- Uses the TREC BioGen prompt template (guide labels, "Respond ONLY with the label no explanation").
- Uses `response_format={"type": "json_object"}` for reliable parsing.

> Priority: **10** — Core evaluation function.

---
## 4.3 Judge Prompt Design — Answer Entailment [DONE]

**[MARKDOWN]** Show the answer entailment prompt (from the TREC BioGen project guide). Explain the entailment criteria using the guide's exact labels:
- **Supported:** The answer is completely supported by the reference sentences.
- **Partially Supported:** The answer is relevant and partially supported, possibly "good to know".
- **Unsupported:** The answer is not supported by the provided sentences and may harm the patient.

**[CODE]** Implement `judge_answer_entailment(question: str, answer: str, reference_sentences: list[str], gpt4o_client, ...) -> dict`:
- Returns `{label: str, unsupported_claims: list[str]}` where label ∈ {Supported, Partially Supported, Unsupported}.
- Uses the TREC BioGen prompt template (guide labels, "Respond ONLY with the label no explanation").
- Uses JSON structured output.

> Priority: **10** — Core evaluation function.

---
## 4.4 Prompt Calibration on Manual Sample [DONE]

**[MARKDOWN]** The guide says: "the prompts are uncalibrated — verify and adapt on 5–10 examples". This step is critical. Before running at scale, we manually inspect the judge's outputs on a small sample and correct any systematic errors (e.g., the judge is too lenient / too strict, misses biomedical-specific nuances).

**[CODE]**
- Randomly select 5 training query answers.
- Run both judge prompts on them.
- **Table 4:** Manual vs GPT-4o judge comparison table — for each of the 5 examples, show: our manual label (you decide while reviewing), GPT-4o label, match/mismatch.
- Identify any systematic issues (e.g., GPT-4o never gives score 0 for sentence alignment).

**[MARKDOWN]** Analysis: document any prompt adjustments made and the rationale. This is important for reproducibility and should go in the report.

> Priority: **9** — The guide explicitly requires this calibration step. Skipping it is a grading risk.

---
## 4.5 Bulk Evaluation — Sentence Alignment (Training Set) [DONE]

**[MARKDOWN]** Run sentence alignment judge on all selected sentences across all training queries. Note: this hits the GPT-4o API — estimated cost and time shown.

**[CODE]**
- Run `judge_sentence_alignment` for all (query, selected_sentence) pairs in train set.
- Save to `results/phase2_sentence_alignment_train.json`.
- **Table 5:** Mean alignment label distribution per query (top-5 and bottom-5 queries by % Required).
- **Plot J:** Distribution of alignment labels (Required/Unnecessary/Borderline/Inappropriate) across all evaluated sentences — stacked bar chart.

> Priority: **9** — Required by the guide. Core evaluation metric.

---
## 4.6 Bulk Evaluation — Answer Entailment (Training Set) [DONE]

**[CODE]**
- Run `judge_answer_entailment` for all generated answers in train set.
- Save to `results/phase2_entailment_train.json`.
- **Table 6:** Entailment label distribution — % Supported / Partially Supported / Unsupported.
- **Plot K:** Pie chart or bar chart of entailment label distribution.

> Priority: **9** — Required by the guide. Core evaluation metric.

---
## 4.7 Correlation: Retrieval Quality → Answer Quality [DONE]

**[MARKDOWN]** Do queries where Phase 1 retrieved better results (higher nDCG) also get better Phase 2 entailment scores? This tests whether better retrieval actually leads to better generation.

**[CODE]**
- Join Phase 1 per-query nDCG scores with Phase 2 per-query entailment scores.
- **Plot L:** Scatter plot — Phase 1 nDCG (x-axis) vs mean answer entailment score (y-axis). Add regression line. Compute Pearson/Spearman correlation.
- **Plot M:** Scatter plot — mean sentence alignment score (x-axis) vs answer entailment (y-axis). Is sentence quality predictive of answer quality?

**[MARKDOWN]** Analysis: does better retrieval → better sentence selection → better answers? Or does the cross-encoder compensate for weaker retrieval? This is one of the most interesting findings to discuss in the report.

> Priority: **8** — Ties the entire project together (Phase 1 retrieval quality → Phase 2 generation quality). One of the most intellectually interesting results.

---
## 4.8 Error Analysis — Failure Cases [DONE]

**[MARKDOWN]** Qualitative deep-dive into failures. Select:
- 2 answers with entailment score 0 (hallucinated/unsupported).
- 2 answers with entailment score 2 (fully grounded).
For each, display the question, reference sentences, generated answer, and the judge's reasoning.

**[CODE]** Display formatted cards for each of the 4 selected examples. No complex code needed here — just structured display.

**[MARKDOWN]** Analysis: what patterns appear in failures? Common failure modes:
- No relevant sentences retrieved (Phase 1 failure propagates).
- Cross-encoder selected tangentially related sentences.
- LLM added "commonsense" claims not in the evidence.

> Priority: **7** — Qualitative analysis is always valuable for reports. Shows deep understanding.

---

---
# **5. Test Set Evaluation & Summary** [DONE]
*Re-run the full pipeline on the test queries and report final results.*

---
## 5.1 Test Set Generation [DONE]

**[MARKDOWN]** We now run the pipeline on the held-out test set (even-ID topics). Remind the reader: we do NOT tune anything based on test set results. Everything was fixed on the training set.

**[CODE]**
- Run full pipeline (retrieval → sentence selection → generation → parsing) on all test queries.
- Save to `results/phase2_test_answers.json`.
- Run both LLM judges on test set answers.
- Save to `results/phase2_sentence_alignment_test.json` and `results/phase2_entailment_test.json`.

> Priority: **10** — Final deliverable. Required.

---
## 5.2 Final Results Tables [DONE]

**[MARKDOWN]** Present Phase 2 results in the format expected for the report.

**[CODE]**
- **Table 7:** Main results table — train vs test for both metrics + constraint compliance:

| Split | % Required (Alignment) | % Supported (Entailment) | % Partially Supported | % Unsupported | % All Constraints Met |
| ----- | ---------------------- | ------------------------ | --------------------- | ------------- | --------------------- |
| Train | ...                    | ...                      | ...                   | ...           | ...                   |
| Test  | ...                    | ...                      | ...                   | ...           | ...                   |

- **Table 8:** Per-query breakdown (test set) — top-5 and bottom-5 queries by entailment score, with Phase 1 nDCG for comparison.

> Priority: **10** — Core result reporting. This goes directly into the report.

---
## 5.3 Comparison: vLLM vs GPT-4o Generator (Optional) [DONE — skipped, vLLM unavailable]

**[MARKDOWN]** *(Only if time/API budget allows — mark as optional.)*
Since we use GPT-4o as the primary generator, run a small sample (5-10 test queries) with vLLM as the generator instead. Compare answer entailment labels: does the frontier model produce more grounded answers than the local model?

**[CODE]**
- **Table 9:** GPT-4o vs vLLM generator — entailment label comparison on 10 test queries.

> Priority: **4** — Nice to have if budget allows. Shows awareness of generator quality impact.

---
## 5.4 Phase 2 Summary & Handoff to Phase 3 [DONE]

**[MARKDOWN]** Structured summary of Phase 2:
- What we built: cross-encoder re-ranker → RAG generator → LLM judge evaluator.
- Locked configuration for Phase 3: which cross-encoder, which generator, top-k docs, top-n sentences.
- Key findings: best alignment score, entailment rate, correlation with Phase 1 retrieval.
- Limitations: what went wrong, what could be improved.

**[CODE]** Print the locked Phase 2 configuration as a Python dict (so Phase 3 can import it directly).

> Priority: **8** — Closes the loop. Good academic practice to summarise findings and hand off cleanly.

---

---

## Appendix: Implementation Notes (not a notebook section — for developer reference)

### New `src/` modules needed for Phase 2:
```
src/
  reranking/
    __init__.py
    cross_encoder.py       # CrossEncoder class: load model, score sentences
    sentence_splitter.py   # split_into_sentences(), handle biomedical abbreviations
    __cross_encoder_test.py
  generation/
    __init__.py
    context_formatter.py   # format_context() -> prompt-ready string with [PMID X] labels
    answer_generator.py    # generate_answer() calling vLLM
    answer_parser.py       # parse_answer() -> structured dict, validate constraints
    __generation_test.py
  evaluation/
    llm_judge.py           # judge_sentence_alignment(), judge_answer_entailment() via GPT-4o
    (add to existing evaluator.py or new file)
  analysis/
    __init__.py
    attention_viz.py       # get_model_outputs(), plot helpers for attention + embeddings
```

### Test checklist (to run before notebook cells):
- `__cross_encoder_test.py`: model loads, single-pair scoring works, batching works, output is scalar, top-3 selection is sorted.
- `__generation_test.py`: answer ≤250 words, ≤3 PMIDs per sentence, all PMIDs in corpus, parse_answer handles edge cases (no citations, extra whitespace, malformed output).
- `llm_judge.py` test: mock GPT-4o call (or use 1 real call) returns expected JSON schema.

### Key API details:
- **vLLM:** `base_url = "https://amalia.novasearch.org/vlm/v1"`, `api_key = "amalia012026"`, model = first model from `client.models.list()`. (Optional secondary generator)
- **GPT-4o (IAedu):** Primary LLM for both generation and judging. Get API key from iaedu.pt → API info. Use `openai.OpenAI(base_url=IAEDU_URL, api_key=IAEDU_KEY)`.
  - **NOTE:** The IAedu API endpoint provided by user is a streaming agent-chat format. Verify whether there is also a standard OpenAI-compatible `/v1/chat/completions` endpoint, or if the `openai` library needs a custom adapter for the streaming format.

### Prompt template draft (sentence alignment judge — from TREC BioGen guide):
```
System: You are an expert annotator. Given a question and an answer sentence, your task is to assign
        a single label from the following list: ['Required', 'Unnecessary', 'Borderline', 'Inappropriate'].
        The label definitions are as follows:
        Required: The answer sentence is necessary to have in the generated answer for
        completeness of the answers.
        Unnecessary: The answer sentence is not required to have included in the generated answer.
        An answer sentence may be unnecessary for several reasons:
        (a) If including it would cause information overload if it is added to the answer;
        (b) If it is trivial, e.g., stating that many treatment options exist.
        (c) If it consists entirely of a recommendation to see a health professional.
        (d) If it is not relevant to the answer, e.g., describing the causes of a disease when the
        question is about treatments.
        Borderline: If an answer sentence is relevant, possibly even "good to know," but not required,
        the answer sentence may be marked borderline.
        Inappropriate: The assertion may harm the patient, e.g., if according to the answer, physical
        therapy reduces the pain level, but the patient experiences more pain due to hip mobilization,
        the patient may start doubting they are receiving adequate treatment.
        Do not generate anything else.
        Respond ONLY with the label no explanation.

User: Question: {query}
      Answer sentence (PMID {pmid}): {sentence}
```

### Prompt template draft (answer entailment judge — from TREC BioGen guide):
```
System: You are an expert annotator. Given a question and a complete answer and its reference
        sentences, your task is to assign a single label from the following list: ['Unsupported',
        'Partially Supported', 'Supported'].
        The label definitions are as follows:
        Supported: The answer is completely supported by the reference sentences.
        Partially Supported: The answer is relevant and partially supported by the reference
        sentences, possibly even "good to know".
        Unsupported: The answer is not supported by the provided sentences and may harm the
        patient.
        Do not generate anything else.
        Respond ONLY with the label no explanation.

User: Question: {question}
      Reference Sentences:
      {reference_sentences_formatted}
      Generated Answer:
      {answer}
```

> **NOTE:** These are the uncalibrated starting prompts from the TREC BioGen guide. During
> prompt calibration (Section 4.4) we manually inspect the judge outputs on 5–10 examples
> and adapt the prompts to improve precision for our specific biomedical domain. Any changes
> are documented in the calibration analysis cell.

---

## Priority Summary (sorted by importance)

| #   | Task                             | Priority | Why                                         |
| --- | -------------------------------- | -------- | ------------------------------------------- |
| 0.1 | Constants & Config               | 10       | Foundation                                  |
| 0.2 | Imports & Env                    | 10       | Foundation                                  |
| 0.3 | Data Loading                     | 10       | Foundation                                  |
| 1.2 | Load Cross-Encoder               | 10       | Core model                                  |
| 1.4 | Sentence Scoring                 | 10       | Core Phase 2 function                       |
| 3.2 | LLM Client Setup                 | 10       | Required for generation + judging           |
| 3.5 | Generation Function              | 10       | Core generation                             |
| 3.7 | Bulk Train Generation            | 10       | Required for evaluation                     |
| 4.2 | Sentence Alignment Judge         | 10       | Core evaluation                             |
| 4.3 | Answer Entailment Judge          | 10       | Core evaluation                             |
| 5.1 | Test Set Generation              | 10       | Final deliverable                           |
| 5.2 | Final Results Tables             | 10       | Report output                               |
| 2.2 | Model Setup for Analysis         | 10       | Analysis section foundation                 |
| 1.1 | Cross-Encoder Concept (MD)       | 9        | Learning tool quality                       |
| 2.1 | Transformer Internals Intro (MD) | 9        | Learning tool quality                       |
| 2.3 | Positional Embeddings Plot       | 9        | Required by guide                           |
| 2.4 | Pairwise Distance Matrix Plot    | 9        | Required by guide                           |
| 2.5 | Contextual Embeddings Plot       | 9        | Required by guide                           |
| 3.1 | RAG Pipeline Design (MD)         | 9        | Learning tool quality                       |
| 3.3 | Context Formatting               | 9        | Determines citation accuracy                |
| 3.4 | Prompt Design (MD)               | 9        | Most important design decision              |
| 4.1 | LLM-Judge Concept (MD)           | 9        | Required conceptual content                 |
| 4.4 | Prompt Calibration (manual)      | 9        | Required by guide — grading risk if skipped |
| 4.5 | Bulk Sentence Alignment Eval     | 9        | Required evaluation                         |
| 4.6 | Bulk Entailment Eval             | 9        | Required evaluation                         |
| 0.4 | Phase 1 Recap                    | 8        | Context for reader, report                  |
| 1.3 | Sentence Segmentation            | 8        | Affects re-ranking quality                  |
| 1.5 | Demo Sentence Selection          | 8        | Learning tool                               |
| 2.6 | Self-Attention Analysis          | 8        | Required analysis + critical discussion     |
| 3.6 | Demo Generation 3 Queries        | 8        | Learning tool                               |
| 5.4 | Phase 2 Summary                  | 8        | Report handoff                              |
| 1.6 | Score Distribution Analysis      | 7        | Diagnostic                                  |
| 4.7 | Retrieval→Answer Correlation     | 8        | Ties entire project together                |
| 4.8 | Error Analysis Failure Cases     | 7        | Qualitative depth                           |
| 3.8 | Constraint Violation Analysis    | 7        | Diagnostic                                  |
| 1.7 | Random vs Cross-Enc Ablation     | 6        | Justification                               |
| 5.3 | vLLM vs GPT-4o Generator         | 4        | Nice to have                                |
