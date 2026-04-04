# Notebook Cell Index — BioMedical NL Agents

**3 notebooks, one per phase. Each is a self-contained learning artifact:**
- Markdown cells teach the concept *before* the code runs it
- Code cells produce visible output (tables, plots, printed results)
- The reader learns *why* → *how* → *what does the result mean*

> **Legend:** `[MD]` = Markdown cell · `[PY]` = Python code cell · `(✅ exists)` = already in phase1_search.ipynb

---

---
# 📓 NOTEBOOK 1 — `phase1_search.ipynb`
**Topic:** Data · Indexing · Retrieval · Tuning · Evaluation
**File:** `tasks/phase1_search.ipynb`

---

## § 0 — Cover & Index
| #   | Type | Title / Purpose                                                                                                                           |
| --- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1 | [MD] | **Cover cell** — project title, phase overview, deadlines, AI disclosure, notebook structure note (`src/` holds logic, notebook calls it) |
| 0.2 | [MD] | **Table of Contents** — numbered list of all sections with one-line descriptions so the reader knows what to expect                       |

---

## § 1 — Infrastructure
| #   | Type | Title / Purpose                                                                                                                                                            |
| --- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | [MD] | **Setup & Environment** — explains local vs Colab duality, `.env` file, Colab Secrets, why we need autoreload *(✅ exists)*                                                 |
| 1.2 | [PY] | **Environment bootstrap** — `IN_COLAB` detect, git clone/pull, `dotenv`, autoreload, `sys.path`, connect to OpenSearch, `check_index()` *(✅ exists)*                       |
| 1.3 | [MD] | **Constants & Configuration guide** — explains each constant group: flags, paths, retrieval size, index settings, HNSW params, encoder choices, tuning grid *(✅ exists)*   |
| 1.4 | [PY] | **Constants cell** — all `FORCE_*`, paths, `RETRIEVAL_SIZE`, `GRADED_SCORE`, `BINARY_THRESHOLD`, `INDEX_*`, `ENCODER_*`, tuning grids, `BEST_*` locked params *(✅ exists)* |

---

## § 2 — Data: Corpus, Topics, Splits, Qrels
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                                             |
| --- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | [MD] | **Dataset overview** — describe corpus (4194 PubMed abstracts, JSONL, PMID + contents), topics (65 BioGen queries with `topic`/`question`/`narrative`), and ground-truth (`biogen_2024_submissions.json`). Show real JSON examples. Explain *why* the ground truth is derived from citation assessments, not human judges — and why that matters *(✅ exists, well written)* |
| 2.2 | [PY] | **Load corpus & topics** — `load_corpus()`, `load_topics()`, print counts and sample entries *(✅ exists)*                                                                                                                                                                                                                                                                   |
| 2.3 | [MD] | **Train / Test split rationale** — odd IDs → train, even IDs → test; split is on *queries only* (corpus is shared); never re-derive inline; importance of not touching test until final eval                                                                                                                                                                                |
| 2.4 | [PY] | **Run splitter** — `run_splitter()`, load results, print: train count, test count, confirm no ID overlap, show 3 sample train + 3 sample test topics                                                                                                                                                                                                                        |
| 2.5 | [MD] | **Qrels: how ground truth becomes evaluation labels** — walk through the `evidence_relation` field values, explain binary vs graded scoring, show the mapping table (supporting=2, neutral=1, else=0), explain why two files are kept on disk, note the limitation (automated citations ≠ human judges) *(✅ exists, well written)*                                          |
| 2.6 | [PY] | **Build qrels** — `run_qrels_builder()`, load both files, print: total (topic, pmid) pairs, distribution of evidence_relation values (bar chart), number of topics with 0 relevant docs and their IDs (edge cases) *(✅ exists)*                                                                                                                                             |

---

## § 3 — OpenSearch Index: Theory & Creation
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                         |
| --- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | [MD] | **Why OpenSearch? Why multiple fields?** — explain that similarity metrics are baked at index time (inverted index per field), so we need one field per (model, params) combination. No re-indexing for tuning — params are pre-baked. One index, multiple fields vs one index per strategy. *(✅ exists, well written)* |
| 3.2 | [MD] | **Dense embeddings: Mean Pooling + L2 norm** — formula for mean pooling, explain why CLS token is suboptimal, explain L2 normalisation and why it makes inner product = cosine. *(✅ exists)*                                                                                                                            |
| 3.3 | [MD] | **HNSW: Approximate Nearest Neighbour index** — explain graph structure, ef_search/ef_construct/M params and their tradeoffs, link to original paper. *(✅ exists, well written)*                                                                                                                                        |
| 3.4 | [PY] | **Baseline index creation** — encode corpus with default msmarco encoder, `create_or_update_index()` with baseline fields only, `index_documents()`, print live fields and doc count *(✅ exists)*                                                                                                                       |

---

## § 4 — Retrieval Baselines: Theory & Demo
| #    | Type | Title / Purpose                                                                                                                                                           |
| ---- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1  | [MD] | **Section intro** — purpose: sanity check + build intuition. Introduce the 5 strategies table, explain the demo query, note that formal evaluation is in § 8 *(✅ exists)* |
| 4.2  | [PY] | **Helper: `print_results()`** — shows rank, PMID, score, abstract snippet for top-5; runs sanity checks (size, monotone scores, no duplicate PMIDs) *(✅ exists)*          |
| 4.3  | [MD] | **BM25 theory** — full formula with k1 and b explanation, IDF formula, score scale note (unbounded positive, not comparable across models) *(✅ exists, excellent)*        |
| 4.4  | [PY] | **BM25 demo** — instantiate `BM25Retriever`, search demo query, `print_results()` *(✅ exists)*                                                                            |
| 4.5  | [MD] | **LM Jelinek-Mercer theory** — smoothed LM formula, λ intuition (low=short queries, high=long queries), score scale note *(✅ exists)*                                     |
| 4.6  | [PY] | **LM-JM demo** — `LMJMRetriever(λ=0.7)`, search, `print_results()` *(✅ exists)*                                                                                           |
| 4.7  | [MD] | **LM Dirichlet theory** — Bayesian prior formula, document-length adaptive smoothing, key difference vs LM-JM *(✅ exists)*                                                |
| 4.8  | [PY] | **LM-Dir demo** — `LMDirichletRetriever(μ=75)`, search, `print_results()` *(✅ exists)*                                                                                    |
| 4.9  | [MD] | **Dense KNN theory** — cosine similarity formula, role of mean pooling + L2 norm, HNSW search, score range note ([0,2] due to OpenSearch offset) *(✅ exists)*             |
| 4.10 | [PY] | **Dense KNN demo** — `Encoder(msmarco)`, `KNNRetriever`, search, `print_results()` *(✅ exists)*                                                                           |
| 4.11 | [MD] | **RRF theory** — formula, k=60 rationale (Cormack 2009), why fusion beats individual models (lexical + semantic), score range note *(✅ exists)*                           |
| 4.12 | [PY] | **RRF demo** — `RRFRetriever(bm25, knn, k=60)`, search, `print_results()`, show where RRF top-1 doc ranked in each individual list *(✅ exists)*                           |
| 4.13 | [MD] | **Cross-model score comparison table** — summary of all 5 score ranges, state clearly scores are not comparable across models, only ranking matters *(✅ exists)*          |

---

## § 5 — Evaluation Metrics: Theory & Reference Card
| #    | Type | Title / Purpose                                                                                                                                                                                                              |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1  | [MD] | **Metric overview intro** — binary vs graded qrels, table of which metric uses which flavour *(✅ exists, excellent)*                                                                                                         |
| 5.2  | [MD] | **NDCG@100** — DCG formula, IDCG, normalisation, graded relevance intuition (2^r − 1 gain), rank discount intuition, how-to-read scale, why @100, edge case (0 relevant → skip) *(✅ exists, excellent)*                      |
| 5.3  | [MD] | **MAP** — AP formula, MAP formula, intuition (area under PR curve), how missing an early relevant doc kills AP, how-to-read scale *(✅ exists, excellent)*                                                                    |
| 5.4  | [MD] | **P@10** — formula, intuition ("first page"), how-to-read scale, limitation (not recall-sensitive, ignores within-top-10 ranking) *(✅ exists)*                                                                               |
| 5.5  | [MD] | **R@100** — formula, intuition (coverage of evidence pool for Phase 2 LLM), how-to-read scale, why critical for our pipeline *(✅ exists)*                                                                                    |
| 5.6  | [MD] | **MRR** — formula, intuition (how deep to scroll for first hit), how-to-read scale, limitation (ignores everything after first hit) *(✅ exists)*                                                                             |
| 5.7  | [MD] | **PR Curves** — what the plot shows, 3 canonical shapes (perfect/good/random), 11-point interpolation explained, how to read our plots *(✅ exists)*                                                                          |
| 5.8  | [MD] | **Metric relationships table** — 5×5 table: which metric captures precision-early / full-recall / ranking-quality / graded-rel / first-hit. "No single metric is complete." *(✅ exists)*                                     |
| 5.9  | [MD] | **Good/Bad reference card** — thresholds table (Bad/Average/Good/Excellent) for each metric calibrated to our task *(✅ exists)*                                                                                              |
| 5.10 | [PY] | **Metric sanity check** — reproduce Lab03 toy example: two ranked lists A and B, verify AP(A)=1.0, AP(B)≈0.7095, verify NDCG on a known example. Print PASS/FAIL. Gives the reader confidence the implementation is correct. |

---

## § 6 — Hyperparameter Tuning: Full Index + Grid Search
| #    | Type | Title / Purpose                                                                                                                                                                                                |
| ---- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1  | [MD] | **Tuning protocol** — 5-fold CV on 32 train queries, why NDCG@100 drives selection, why test set is never touched, what parameters are tuned and the search ranges *(✅ exists)*                                |
| 6.2  | [PY] | **Build full tuning index** — encode all 3 encoders, `create_or_update_index()` with full grids (all k1/b pairs, all λ, all μ, all encoders), `index_documents()`, print all live fields *(✅ exists)*          |
| 6.3  | [MD] | **Query field ablation: which topic field to use?** — explain the 6 variants (topic, question, narrative, topic+question, topic+narrative, all-3 concatenated), motivate why this matters before tuning models |
| 6.4  | [PY] | **Run query field ablation** — 5-fold CV with BM25-default on all 6 query formulations, bar chart of NDCG@100 per formulation, print winner, **lock `BEST_QUERY_FIELD`**                                       |
| 6.5  | [MD] | **BM25 k1/b tuning** — recap of what k1 and b do, show the 5×4 grid we sweep                                                                                                                                   |
| 6.6  | [PY] | **BM25 grid search** — 5-fold CV over all (k1, b) pairs, heatmap plot (rows=k1, cols=b, color=NDCG@100), print top-5 configs, **lock `BM25_K1_B_BEST`**                                                        |
| 6.7  | [MD] | **LM-JM λ tuning** — recap: two values pre-baked in index (λ=0.1 and λ=0.7), pick winner                                                                                                                       |
| 6.8  | [PY] | **LM-JM λ comparison** — 5-fold CV for λ=0.1 vs λ=0.7, bar chart, print MAP and NDCG for both, **lock `LMJM_LAMBDA_BEST`**                                                                                     |
| 6.9  | [MD] | **LM-Dir μ tuning** — recap: Dirichlet pseudo-count, what small μ vs large μ does for biomedical abstracts                                                                                                     |
| 6.10 | [PY] | **LM-Dir μ sweep** — 5-fold CV over all μ values, line plot (x=μ, y=NDCG@100 with error bars), **lock `LMDIR_MU_BEST`**                                                                                        |
| 6.11 | [MD] | **Encoder comparison: msmarco vs MedCPT vs multi-qa** — explain domain mismatch of msmarco (trained on Bing), domain match of MedCPT (trained on PubMed click data), expected ranking                          |
| 6.12 | [PY] | **Encoder sweep** — 5-fold CV for all 3 encoders, grouped bar chart, print MAP/NDCG/P@10 per encoder, **lock `ENCODER_BEST`**                                                                                  |
| 6.13 | [MD] | **Tuning summary** — recap all locked params in one table: param → winner → train NDCG gain vs baseline. Explain that all subsequent evaluation uses these fixed params.                                       |

---

## § 7 — Run Generation: Save All Run Files
| #   | Type | Title / Purpose                                                                                                                                                                         |
| --- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.1 | [MD] | **What a "run file" is** — IR terminology: a run file maps each query to a ranked list of (docid, score) pairs. JSON format. Used by evaluation code and comparable across systems.     |
| 7.2 | [PY] | **Generate all 5 run files (train set)** — for each of the 5 strategies with best params, run all train queries, save to `results/phase1/*_train_run.json`. Print timing per strategy.  |
| 7.3 | [PY] | **Generate all 5 run files (test set)** — same but test queries. Save to `results/phase1/*_run.json`. **Test set touched here for the first and only time.** Print counts per run file. |

---

## § 8 — Final Evaluation on Test Set
| #    | Type | Title / Purpose                                                                                                                                                                                                                   |
| ---- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.1  | [MD] | **Evaluation protocol** — 33 test queries, binary qrels for MAP/P@10/R@100/MRR, graded qrels for NDCG@100. Results are final — no further tuning allowed after seeing these numbers.                                              |
| 8.2  | [PY] | **Compute all metrics for all 5 strategies** — load run files + qrels, compute full metric table, save `results/phase1/final_eval_summary.json`                                                                                   |
| 8.3  | [PY] | **Comparison table** — pretty-print DataFrame: rows=strategies, cols=NDCG@100/MAP/P@10/R@100/MRR, bold best per column                                                                                                            |
| 8.4  | [PY] | **Mean PR curves** — one plot, 5 curves (one per strategy), shaded ±1 std band, random baseline at corpus prevalence (~1.1%), legend, axis labels                                                                                 |
| 8.5  | [MD] | **Results discussion** — guide the reader through the table: which strategy wins each metric? Why does RRF not always win MAP? Why does KNN outperform sparse models in NDCG? When does BM25 beat dense?                          |
| 8.6  | [PY] | **AP distribution per strategy** — boxplot: x=strategy, y=per-query AP, shows variance across queries not just mean                                                                                                               |
| 8.7  | [MD] | **Three specific query analyses (required)** — introduce: highest-AP query, lowest-AP query, one additional "interesting" query                                                                                                   |
| 8.8  | [PY] | **Find the 3 queries** — auto-select: `argmax(AP)`, `argmin(AP)`, one mid-range query; print their topic text, AP per strategy                                                                                                    |
| 8.9  | [PY] | **Individual PR curves for the 3 queries** — 3×1 subplot grid: per-query PR curves, all 5 strategies on each plot, title shows topic text + AP values                                                                             |
| 8.10 | [MD] | **Analysis of the 3 curves** — explain *why* the best query is easy (many distinct terms, many relevant docs), why the worst is hard (vocabulary mismatch? few relevant docs? ambiguous topic?), what the mid-range curve teaches |

---

## § 9 — Error & Corpus Analysis
| #   | Type | Title / Purpose                                                                                                                                                                                                   |
| --- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9.1 | [MD] | **Error analysis goal** — understand *why* retrieval fails on certain queries. Two lenses: (a) qrel sparsity (too few relevant docs → MAP is noisy), (b) vocabulary mismatch (query terms don't appear in corpus) |
| 9.2 | [PY] | **Worst-performing queries table** — sort test queries by BM25 AP ascending, show bottom-10: query text, AP, number of relevant docs in qrels, number of relevant docs retrieved in top-100                       |
| 9.3 | [PY] | **IDF analysis** — for each query, tokenise and compute IDF of each term in corpus. Scatter plot: x=query term IDF, y=mean AP across strategies. Do rare terms (high IDF) correlate with better retrieval?        |
| 9.4 | [PY] | **Corpus vocabulary coverage** — for worst-5 queries: count what fraction of query words appear ≥1 time in corpus. Low coverage → vocabulary mismatch → dense retrieval may help                                  |
| 9.5 | [MD] | **Analysis conclusions** — synthesise: which failure mode dominates? sparse qrels? vocabulary mismatch? domain specificity? What does this suggest for Phase 2 (cross-encoder can still find relevant sentences)? |

---

## § 10 — Phase 1 Report Summary
| #    | Type | Title / Purpose                                                                                                                                                                                            |
| ---- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10.1 | [MD] | **Experimental Setup (report-ready text)** — datasets (corpus size, #topics, split), metrics used (definitions + formulas), evaluation protocol (5-fold CV on train, final test), locked hyperparams table |
| 10.2 | [MD] | **Results & Discussion (report-ready text)** — final comparison table (copy from §8.3), PR curve interpretation, 3-query analysis summary, key findings in bullet points                                   |
| 10.3 | [MD] | **Limitations (report-ready text)** — qrel quality (citation-derived, not human), corpus scope (4194 abstracts, not full PubMed), topics with 0 relevant docs, BM25 IDF at corpus scale vs full PubMed     |

---

---
# 📓 NOTEBOOK 2 — `phase2_rag.ipynb`
**Topic:** Cross-Encoder Reranking · BERT Visualization · Answer Generation · LLM-as-Judge
**File:** `tasks/phase2_rag.ipynb`

---

## § 0 — Cover & Index
| #   | Type | Title / Purpose                                                                                                                                                          |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0.1 | [MD] | **Cover cell** — Phase 2 title, goal ("move from retrieving documents to generating grounded answers"), what this phase builds on (Phase 1 run files + qrels), deadlines |
| 0.2 | [MD] | **Table of Contents** — numbered sections overview                                                                                                                       |

---

## § 1 — Infrastructure
| #   | Type | Title / Purpose                                                                                                                                                        |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | [MD] | **Setup notes** — same Colab/local pattern as Phase 1, additional API: IAedu GPT-4o for judge (separate from vLLM)                                                     |
| 1.2 | [PY] | **Bootstrap** — `IN_COLAB`, autoreload, dotenv, `sys.path`, connect OpenSearch, connect vLLM (`amalia.novasearch.org`), connect IAedu, print available vLLM model name |
| 1.3 | [PY] | **Constants** — paths to Phase 1 results, `TOP_K_RERANK`, `TOP_N_SENTENCES`, `MAX_WORDS`, `MAX_PMIDS_PER_SENTENCE`, model names, judge API config                      |
| 1.4 | [PY] | **Load Phase 1 artifacts** — load `test_queries.json`, `qrels.json`, `qrels_graded.json`, best Phase 1 run file (RRF or whichever won). Print counts.                  |

---

## § 2 — Cross-Encoder Reranking
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                              |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | [MD] | **Bi-encoder vs cross-encoder: the key difference** — bi-encoders encode query and document *independently* (fast, can pre-compute), cross-encoders encode them *jointly* (sees both at once → much richer interaction, but slower). Diagram: `[CLS] query [SEP] document [SEP]` → single relevance score. When to use each. |
| 2.2 | [MD] | **MedCPT Cross-Encoder** — why domain-specific matters (trained on PubMed user clicks), model card link, expected gain over general cross-encoders                                                                                                                                                                           |
| 2.3 | [PY] | **Load cross-encoder & demo on one (query, doc) pair** — show the raw relevance score before vs after reranking for a single example. Intuition: the cross-encoder sees the full abstract jointly with the query and produces a single float.                                                                                |
| 2.4 | [PY] | **Rerank top-100 → top-N for all test queries** — for each query: take Phase 1 top-100, cross-encode all (query, doc) pairs, re-sort, save `results/phase2/reranked_run.json`. Show progress bar.                                                                                                                            |
| 2.5 | [PY] | **Reranking evaluation: before vs after** — compute P@10, MAP, NDCG@100 for Phase 1 best run AND reranked run on same test queries. Print side-by-side table. Bar chart: metric × (before, after).                                                                                                                           |
| 2.6 | [MD] | **Reranking results discussion** — which metrics improve most? Why does P@10 improve more than MAP? When does reranking *hurt*? (When Phase 1 recall is low, there's nothing good to rerank to the top.)                                                                                                                     |

---

## § 3 — BERT: Positional & Contextual Embeddings (Required Exercise)
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 3.1 | [MD] | **Why study BERT internals?** — cross-encoders are fine-tuned BERT models. Understanding how BERT represents text (positionally, contextually, through attention) explains *why* they outperform bag-of-words. This section is a self-contained deep-dive into transformer representations.                                                                                                                                                                                                                              |
| 3.2 | [MD] | **Positional embeddings: theory** — BERT uses learned absolute positional embeddings (not sinusoidal). Each position 0..511 has a learnable vector added to the token embedding. Same word at position 0 vs position 100 gets a different positional offset. This is how the model knows where in the sequence each token is.                                                                                                                                                                                            |
| 3.3 | [PY] | **Positional embedding experiment** — insert same word ("the") repeated 200 times. Extract last-layer token embeddings (shape: 200×768). Compute L2 distance of each token to token-0. **Plot 1:** line plot x=position, y=distance-to-first — shows how representations drift with position despite identical word. **Plot 2:** full 200×200 pairwise distance matrix heatmap. Discuss: what pattern do you see? (band structure — nearby positions are closer)                                                         |
| 3.4 | [MD] | **Contextual embeddings: theory** — unlike static embeddings (word2vec), BERT computes a different vector for the same word depending on surrounding context. The same word "treatment" in "cancer treatment" vs "water treatment" gets different representations. This is the core of contextuality.                                                                                                                                                                                                                    |
| 3.5 | [PY] | **Contextual embedding experiment** — take a sentence with "bank" in two senses ("river bank" vs "savings bank"). Extract hidden states from all 12 layers (shape: 12 × seq_len × 768). Plot: for the "bank" token across both sentences, compute cosine similarity at each layer. **Layer 0:** both senses are similar (syntactic, positional). **Layer 11:** they diverge (semantic disambiguation). Also: t-SNE of layer-0 vs layer-11 token embeddings for a small paragraph — show how contextual geometry evolves. |
| 3.6 | [MD] | **Self-attention: theory** — attention allows each token to "look at" every other token. The attention weight $a_{ij}$ says "how much should token $i$ attend to token $j$ when computing its own new representation?" Multi-head attention runs H=12 parallel heads — each may specialise (syntax, coreference, proximity). Formula: $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$                                                                                                                            |
| 3.7 | [PY] | **Self-attention visualization in cross-encoder** — take one (query, doc) pair as input to the cross-encoder. Extract attention weights for all 12 layers × 12 heads. **Plot 1:** single head attention matrix heatmap — rows=query tokens, cols=doc tokens — show which doc words the query attends to. **Plot 2:** averaged across heads for one interesting layer. Critical analysis: does the model attend to medically relevant terms? Does "treatment" in the query attend to "therapy" in the document?           |
| 3.8 | [MD] | **Synthesis: what we observed** — connect the three experiments: (1) positional embeddings give the model sequential awareness; (2) contextual embeddings let the same word mean different things; (3) self-attention lets query terms find relevant evidence in the document regardless of position. *This is why cross-encoders outperform BM25 and bi-encoders.*                                                                                                                                                      |

---

## § 4 — Reference Sentence Selection
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                                                                     |
| --- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1 | [MD] | **From document retrieval to sentence retrieval** — the reranker gives us a ranked list of full abstracts. But an abstract has ~5-10 sentences; only 1-3 are directly relevant to the query. Feeding the full abstract to the LLM wastes context window and dilutes the signal. We select the top-3 sentences per abstract as *reference sentences* — the minimal factual grounding for generation. |
| 4.2 | [PY] | **Sentence splitting + cross-encoder scoring** — for top-K reranked abstracts: split each into sentences (spaCy or regex), score each (query, sentence) pair with the cross-encoder, take top-3 per abstract. Show example output for one query.                                                                                                                                                    |
| 4.3 | [PY] | **Visualise selected sentences for 3 test topics** — for each: show query, then the top-3 sentences selected from the top-3 documents. Reader can see what the LLM will use as evidence.                                                                                                                                                                                                            |
| 4.4 | [MD] | **Why this matters** — reference sentences are the *attribution units* for generation. Each answer sentence must be attributable to ≥1 reference sentence. This is the grounding chain: query → retrieved docs → selected sentences → generated answer → citations.                                                                                                                                 |

---

## § 5 — LLM Answer Generation
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1 | [MD] | **RAG pipeline overview** — diagram: query → retriever → cross-encoder → sentence selector → prompt builder → LLM → validator → answer with citations. Explain each step in one sentence.                                                                                                                                                      |
| 5.2 | [MD] | **Prompt design** — show the exact prompt template. Explain: system role, instruction (answer in ≤2500 words, cite PMIDs inline, ≤3 PMIDs per sentence), few-shot example (optional), and the context block (reference sentences with their PMIDs). Discuss: why few-shot helps, why constraints must be in the system prompt not user prompt. |
| 5.3 | [PY] | **Connect to vLLM** — query `amalia.novasearch.org` for available model name, print it, verify the API responds.                                                                                                                                                                                                                               |
| 5.4 | [PY] | **Generate answers for 3 demo topics** — call `answer_generator.py`, show raw generated text, validate with `answer_validator.py` (word count, PMID per sentence, valid PMIDs). Print PASS/FAIL for each constraint.                                                                                                                           |
| 5.5 | [PY] | **Generate answers for all test queries** — loop with progress bar, validate each answer, log any constraint violations, save `results/phase2/generated_answers.json`. Print stats: total answers, violation rate, avg word count, avg PMIDs cited.                                                                                            |
| 5.6 | [MD] | **Constraints discussion** — why ≤2500 words (comprehensive but concise), why ≤3 PMIDs per sentence (attribution clarity), why only valid PMIDs (hallucination prevention). Show one example of a violation that the validator catches.                                                                                                        |

---

## § 6 — LLM-as-Judge Evaluation
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                    |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 6.1 | [MD] | **Why LLM-as-judge?** — human annotation of every generated answer sentence is prohibitively expensive. Frontier LLMs (GPT-4o) can serve as automatic evaluators. Two judgment tasks from TREC BioGen: (1) sentence relevance, (2) citation entailment. Note: the judge is GPT-4o via IAedu — NOT the vLLM server. |
| 6.2 | [MD] | **Judge Task 1: Sentence Relevance** — does this answer sentence actually address the biomedical question? Show the exact prompt template. Output format: JSON `{"relevant": true/false, "reason": "..."}`                                                                                                         |
| 6.3 | [MD] | **Judge Task 2: Citation Entailment** — does the cited abstract's text logically support the answer sentence? Show the exact prompt template. Output: JSON `{"entailed": true/false, "reason": "..."}`                                                                                                             |
| 6.4 | [PY] | **Prompt calibration on 5 manual examples** — take 5 hand-picked (query, sentence, citation) triples where you know the correct judgment. Run GPT-4o judge. Print: judge output vs manual label, agreement rate. Adjust prompt if needed.                                                                          |
| 6.5 | [PY] | **Run judge on all generated answers** — for each test topic: judge each answer sentence for relevance, judge each (sentence, cited PMID) pair for entailment. Progress bar. Save `results/phase2/judge_labels.json`.                                                                                              |
| 6.6 | [PY] | **Judge results summary** — print: % sentences judged relevant, % citations judged entailed, breakdown per topic (bar chart), examples of high-entailment vs low-entailment answers.                                                                                                                               |
| 6.7 | [MD] | **Limitations & biases of LLM-as-judge** — self-preference bias (GPT-4o may favour GPT-like answers), prompt sensitivity, lack of domain calibration, cost. How to mitigate: multi-judge, human spot-check, calibration examples.                                                                                  |

---

## § 7 — Phase 2 Report Summary
| #   | Type | Title / Purpose                                                                                                           |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------- |
| 7.1 | [MD] | **Reranking section (report-ready)** — model choice rationale, before/after table, discussion                             |
| 7.2 | [MD] | **Generation section (report-ready)** — prompt design, constraint enforcement, sample outputs                             |
| 7.3 | [MD] | **Judge section (report-ready)** — judge tasks, calibration, aggregate results, limitations                               |
| 7.4 | [MD] | **BERT visualisation section (report-ready)** — connect the 3 visualisations to the cross-encoder's performance advantage |

---

---
# 📓 NOTEBOOK 3 — `phase3_agent.ipynb`
**Topic:** ReAct Agent · Planning · Exploration · Aggregation · Report Generation
**File:** `tasks/phase3_agent.ipynb`

---

## § 0 — Cover & Index
| #   | Type | Title / Purpose                                                                                                                                                       |
| --- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1 | [MD] | **Cover cell** — Phase 3 title, goal ("build a deep research agent that plans, retrieves, and synthesises a structured biomedical report"), how it extends Phases 1+2 |
| 0.2 | [MD] | **Table of Contents**                                                                                                                                                 |

---

## § 1 — Infrastructure
| #   | Type | Title / Purpose                                                                                                                                                  |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1 | [MD] | **Setup notes** — same Colab/local pattern, all Phase 1+2 components are reused (retriever, cross-encoder, vLLM). The agent is a new orchestration layer on top. |
| 1.2 | [PY] | **Bootstrap** — `IN_COLAB`, autoreload, dotenv, OpenSearch client, vLLM client, print available model                                                            |
| 1.3 | [PY] | **Constants** — `MAX_SUBTOPICS`, `MAX_REACT_ITERATIONS`, `MIN_EVIDENCE_DOCS`, paths, model config                                                                |
| 1.4 | [PY] | **Load shared artifacts** — load test queries, qrels, Phase 2 reranker (cross-encoder), Phase 1 best retriever                                                   |

---

## § 2 — Agentic Patterns: Theory
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                                                                |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1 | [MD] | **Why agents?** — a single-pass RAG (Phase 2) answers the query in one shot. But a complex biomedical question has multiple sub-aspects (e.g. "vitamin D deficiency" → bone health, cardiovascular risk, immune function, cancer risk). A single query retrieves some but misses others. An agent can *plan* which sub-aspects to explore and *iterate* until it has enough evidence for each. |
| 2.2 | [MD] | **The ReAct pattern (Reason + Act)** — diagram of the loop: `Reason → Act (SEARCH) → Observe → Reason → Act → ...`. Explain: at each step the LLM reasons about what it knows and what it still needs, then generates a search action, observes the results, and updates its reasoning state. Cite: Yao et al. 2022.                                                                           |
| 2.3 | [MD] | **Our agent architecture** — 4-component diagram: `Planner → Explorer (ReAct loop per sub-topic) → Aggregator → Report Writer`. One-sentence description of each component.                                                                                                                                                                                                                    |
| 2.4 | [MD] | **Structured action format** — show the XML format we use for LLM actions. Why strict format (not free text)? Parsing robustness: free text → regex fragility; XML/JSON → deterministic parsing. Show PASS/FAIL examples.                                                                                                                                                                      |

---

## § 3 — Planner
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                 |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3.1 | [MD] | **What the planner does** — takes a biomedical topic and decomposes it into N sub-topics for targeted evidence gathering. Example: "obstructive sleep apnea" → ["CPAP therapy outcomes", "cardiovascular comorbidities", "weight loss interventions", "surgical alternatives"]. |
| 3.2 | [PY] | **Planner demo on 3 topics** — call `planner.py` with 3 test topics, print: input topic + output sub-topic list for each. Show that sub-topics are specific and non-overlapping.                                                                                                |
| 3.3 | [PY] | **Planner robustness check** — run planner on all test topics, verify: ≥1 sub-topic per topic, all sub-topics are non-empty strings, no duplicates within a topic. Print stats: min/avg/max sub-topics per topic.                                                               |
| 3.4 | [MD] | **Planner quality discussion** — what makes a good sub-topic? (specific, actionable as a search query, covers a distinct aspect). What can go wrong? (too broad, redundant, unrelated to original topic).                                                                       |

---

## § 4 — Explorer: ReAct Loop
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                                  |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1 | [MD] | **Explorer design** — the explorer runs the ReAct loop for ONE sub-topic. Each iteration: (1) LLM reasons about what evidence it has and what it still needs; (2) LLM generates a search query; (3) retriever returns top-K docs; (4) cross-encoder selects top-3 sentences; (5) sentences added to evidence buffer; (6) repeat. Hard cap: MAX_REACT_ITERATIONS. |
| 4.2 | [MD] | **When does the explorer stop?** — two stopping conditions: (a) enough evidence (MIN_EVIDENCE_DOCS unique supporting docs), (b) iteration cap reached. Log which condition triggered. Discuss the tradeoff: too few iterations → shallow evidence; too many → redundant docs, slow, expensive API calls.                                                         |
| 4.3 | [PY] | **Explorer demo: full trace for 1 sub-topic** — run `explorer.py` on one sub-topic with verbose logging. Print each iteration: reason text, generated query, top-3 retrieved PMIDs + scores, selected sentences. Show the evidence accumulating over iterations.                                                                                                 |
| 4.4 | [PY] | **Explorer stats across all sub-topics (one topic)** — run full explorer for all sub-topics of one test topic. Print: iterations per sub-topic, unique PMIDs found, stopping condition.                                                                                                                                                                          |
| 4.5 | [MD] | **Explorer failure modes** — what if the retriever returns no relevant docs? What if the LLM keeps generating the same query? (Loop detection: if last 2 queries are identical → stop.) Show one real example of a hard sub-topic where the explorer struggles.                                                                                                  |

---

## § 5 — Aggregator
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                                                                                  |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 5.1 | [MD] | **What the aggregator does** — merges evidence buffers from all sub-topic explorers into a unified evidence pool. Key operations: (1) deduplicate PMIDs (same doc may support multiple sub-topics), (2) validate all PMIDs exist in corpus, (3) group evidence by sub-topic for report structuring.                                              |
| 5.2 | [PY] | **Run aggregator on one topic** — show input (N sub-topic evidence dicts) and output (unified evidence pool). Print: total PMIDs before dedup, after dedup, any invalid PMIDs removed.                                                                                                                                                           |
| 5.3 | [MD] | **Handling conflicting evidence** — what if two sub-topics retrieved the same PMID but for different reasons? (e.g. a paper about "CPAP for hypertension" is relevant to both the "CPAP therapy" and "cardiovascular" sub-topics). The aggregator keeps both associations — the report writer uses the sub-topic grouping for section structure. |

---

## § 6 — Report Writer
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 6.1 | [MD] | **Report structure design** — the final report has one section per sub-topic, each with: section header, 3-8 sentences with inline PMID citations, sourced from the aggregated evidence. Same constraints as Phase 2: ≤2500 words, ≤3 PMIDs/sentence, valid PMIDs only.        |
| 6.2 | [MD] | **Prompt design for report writing** — show the prompt template: system role (biomedical report writer), structured evidence input per section, constraints, output format instructions. Compare with Phase 2 single-answer prompt — note the structured multi-section output. |
| 6.3 | [PY] | **Generate report for 1 topic** — call `report_writer.py`, show full formatted output with section headers and inline citations. Validate constraints.                                                                                                                         |
| 6.4 | [PY] | **Generate reports for all test topics** — loop with progress bar, validate each report, save `results/phase3/agent_reports.json`. Print stats: valid reports, violation rate, avg word count, avg sections, avg PMIDs cited.                                                  |

---

## § 7 — Full Agent: End-to-End Demo
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                                      |
| --- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 7.1 | [MD] | **End-to-end pipeline recap** — diagram showing all 4 components in sequence for one topic. "This is the full agent — from a biomedical question to a structured, cited research report."                                                                                            |
| 7.2 | [PY] | **Full agent run on 1 topic with verbose trace** — run all 4 stages for one topic, print the intermediate output at each stage (sub-topics, exploration trace per sub-topic, aggregated evidence, final report). This is the "wow" cell — shows the complete agent thinking process. |
| 7.3 | [PY] | **Agent run stats across all test topics** — print table: topic → #sub-topics → #explorer iterations → #unique PMIDs → #report words → constraint status                                                                                                                             |

---

## § 8 — Agent Evaluation
| #   | Type | Title / Purpose                                                                                                                                                                                                                                                              |
| --- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8.1 | [MD] | **Evaluation strategy** — no single number captures "report quality". We use 3 lenses: (1) coverage (R@100: did the agent find the relevant PMIDs?), (2) LLM-judge sentence quality (from Phase 2), (3) structural completeness (all sub-topics addressed, constraints met). |
| 8.2 | [PY] | **Coverage evaluation** — for each test topic: treat agent's retrieved PMID set as a "run", compute R@100 against qrels. Compare vs Phase 1 single-query R@100. Bar chart: per-topic R@100 Agent vs Phase 1 Best.                                                            |
| 8.3 | [PY] | **LLM-judge on agent reports** — re-use Phase 2 judge (`llm_judge.py`) to assess: % sentences relevant, % citations entailed. Compare vs Phase 2 single-pass RAG results.                                                                                                    |
| 8.4 | [PY] | **Phase 2 vs Phase 3 comparison table** — rows: metrics (R@100, % relevant sentences, % entailed citations, avg words, avg PMIDs). Cols: Phase 2 RAG vs Phase 3 Agent.                                                                                                       |
| 8.5 | [MD] | **Evaluation discussion** — when does the agent clearly outperform single-pass RAG? (complex multi-aspect topics) When does it not? (narrow single-aspect topics where single-pass suffices) What does extra iteration cost in API calls? Is the quality gain worth it?      |

---

## § 9 — Phase 3 Report Summary
| #   | Type | Title / Purpose                                                                                                                                                                     |
| --- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9.1 | [MD] | **Agent architecture section (report-ready)** — planner + explorer + aggregator + writer diagram with explanations                                                                  |
| 9.2 | [MD] | **Example trace section (report-ready)** — pick 1 topic, show the full reasoning trace cleanly formatted (reason → search → observe → … → report)                                   |
| 9.3 | [MD] | **Evaluation section (report-ready)** — Phase 2 vs Phase 3 comparison table + discussion                                                                                            |
| 9.4 | [MD] | **Limitations & future work (report-ready)** — planner quality bottleneck, iteration cost, hard iteration cap tradeoff, lack of memory across topics, potential for sub-topic drift |

---

---
# 📊 Cell Count Summary

| Notebook                      | MD cells | PY cells | Total    |
| ----------------------------- | -------- | -------- | -------- |
| Phase 1 — Search & Evaluation | ~35      | ~25      | ~60      |
| Phase 2 — RAG & LLM-Judges    | ~25      | ~20      | ~45      |
| Phase 3 — Deep Research Agent | ~20      | ~15      | ~35      |
| **Total**                     | **~80**  | **~60**  | **~140** |

---

# 🧭 Design Principles for All Three Notebooks

1. **Teach before doing** — every `[PY]` cell is preceded by a `[MD]` cell that explains the concept, formula, and intuition. The reader should understand *why* before seeing *how*.
2. **Output tells a story** — every code cell prints something useful: a table, a plot, a confirmation, or a concrete example. No silent cells.
3. **Constants are a single source of truth** — all configurable values in the constants cell at the top. No magic numbers inline.
4. **Idempotent operations** — re-running any cell produces the same result. Use `FORCE_*` flags for the heavy one-time operations.
5. **Sanity checks before results** — always verify the output of a heavy operation (doc count, shape, score range, no duplicates) before using it downstream.
6. **Connect phases explicitly** — Phase 2 loads Phase 1 run files. Phase 3 loads Phase 2 reranker. The reader sees the pipeline as a whole, not isolated experiments.
7. **Report-ready markdown at the end** — each notebook ends with polished `[MD]` cells that can be copy-pasted into the final report document.
