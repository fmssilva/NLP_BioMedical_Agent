# Phase 1 -- Verification & Cleanup Notes

Status: All items addressed.

---

## 1. Binary vs Graded Qrels -- RESOLVED
- Binary: supporting=1, all others=0 (used for MAP, MRR, P@k, R@k)
- Graded: supporting=2, neutral=1, others=0 (used for NDCG)
- Both shown side-by-side in notebook section 14 and report
- Professor confirms: graded for NDCG, threshold for precision (binary). Current implementation correct.
- Explained in notebook section 16 ("Binary vs Graded Evaluation")
- MAP-based model selection is the IR standard for tuning.



## 2. Qrels & NDCG Plain-English Explanation -- DONE
- Added new markdown cell after section 1 (data loading) with full plain-English qrels/NDCG explanation.
- Report already has "Relevance Judgements (qrels)" section.

## 3. Notebook Cell Errors -- FIXED
- Cell 3 (setup) had ROOT path error: os.getcwd() returned tasks/phase1/ not project root.
- Fixed with parent directory navigation logic.

## 4. MedCPT Index & Vectorization -- CONFIRMED
- MedCPT embeddings computed separately (medcpt_docs.npy, 4194 x 768).
- embedding_medcpt field added to OpenSearch via final_eval.py --add-medcpt.
- Encoder comparison (section 13) uses pure-Python exact cosine (100% accurate, no HNSW loss).
- Final evaluation (section 14) uses HNSW via OpenSearch for both msmarco and MedCPT.
- HNSW loss: ~1-5% recall@100 for our config (ef_construction=256, m=48).
- All explained in section 3 markdown.

## 5. Mean Pooling vs CLS Token -- DONE
- Section 3 updated: "mean pooling (not CLS token)" with full explanation.
- encoder.py implements _mean_pooling() -- weighted mean of all token embeddings.

## 6. Code Simplicity & Duplication -- REVIEWED
- evaluator.py: baseline pipeline, binary metrics only (MAP/MRR/P@10).
- final_eval.py: adds R@100, NDCG@10, graded qrels, per-query detail.
- Overlap is partial and justified. Docstring clarification added.
- All src/ files have adequate inline comments and self-test sections.

## 7. Part A Duplication -- FIXED
- Duplicate header replaced with proper section 7 KNN Demo content.

## 8. Higher LM-JM Lambda -- DOCUMENTED
- Only lambda=0.1 and 0.7 index fields exist. Testing lambda>0.7 requires new fields.
- Added explanation to section 10: trend predicts <0.005 MAP gain from higher lambda.

## 9. Tuning Qrels -- CONFIRMED
- MAP-based selection (binary qrels) is IR standard. Rationale added to Part B protocol.

## 10. BM25 Heatmap -- FIXED
- Removed k1=1.8/2.0 from CSV (28->20 rows). Updated sweep script grid.
- Notebook cell has VALID_K1 safety filter.

## 11. Notebook Section 15 Reorganization -- DONE
- Added 15.1/15.2 subsection headers (PR interpretation demo vs final PR curves).

## 12. Notebook Section 18 Reorganization -- DONE
- Split monolithic analysis cell into 18.1-18.8 subsections.
- Each: markdown explanation -> code cell -> output.

## 13. Part B Table -- FIXED
- Updated k1 grid to {0.5, 0.8, 1.0, 1.2, 1.5} (removed 2.0).

## 14. Dense Encoders -- CONFIRMED
- msmarco-distilbert is from Sentence Transformers (Lab01 pattern).
- MedCPT is justified as domain-specific zero-shot encoder.

---

## Session 3 (2026-04-01)

## 15. Colab Setup & Credentials -- DONE
- Setup cell updated: local (.env walkup) vs Colab (google.colab.userdata Secrets).
- Colab: add OPENSEARCH_USER / OPENSEARCH_PASS / OPENSEARCH_INDEX in Secrets panel.
- Both paths inject into os.environ -> opensearch_client.py picks them up identically.
- Added setup explanation markdown cell before the code cell.

## 16. Can We Use a Single Graded Qrels File? -- ANALYSED
- Answer: technically yes. graded[score>=2] == binary for 100% of pairs (verified).
- Decision: keep two files on disk for clarity. One for binary (MAP/MRR/P@k/R@k), one for graded (NDCG).
- Both files are derived from the same raw submissions; graded is strict superset.
- Data confirmed: 2999 binary pairs, 3217 graded pairs (218 extra neutral at score=1).
- Documented in notebook §1.2 markdown, report qrels section, and qrels_builder_test.

## 17. Qrels Builder Tests -- DONE
- Created src/data/__qrels_builder_test.py with 11 comprehensive tests.
- Tests cover: structure, score values, binary/graded consistency, corpus membership,
  score distribution, no empty topics, full 65-topic coverage,
  synthetic unit test (no file I/O), rebuild determinism.
- All 11 tests pass as of 2026-04-01.
- Run: python -m src.data.__qrels_builder_test

## 18. Results Files in Colab -- CONFIRMED
- All notebook cells use ROOT-relative paths (ROOT / "results" / ...).
- results/ is NOT gitignored -> pre-computed run files available in Colab after git clone.
- embeddings/ IS gitignored -> Colab re-encodes (~3 min, handled by if EMB_PATH.exists() guard).
- PHASE1_DIR, TUNING_DIR, FIGS_DIR, EMB_PATH all ROOT-based.

---

## Summary of Changes

### Notebook (tasks/phase1/phase1_search.ipynb)
- Fixed ROOT path resolution in cell 3
- Added qrels/NDCG plain-English explanation after section 1
- Updated section 3: mean pooling, HNSW loss, MedCPT index details
- Fixed section 7: replaced duplicate with KNN demo content
- Updated section 10: lambda field limitations
- Updated Part B: fixed k1 grid, added MAP selection rationale
- Updated section 12 heatmap: VALID_K1 filter
- Added section 15 subsection headers
- Split section 18 into 18.1-18.8 subsections
- [Session 3] Updated setup cell: Colab Secrets + local .env walkup
- [Session 3] Added §1.1 markdown with Colab vs local setup table

### Source Code (src/)
- src/tuning/bm25_param_sweep.py: removed k1=1.8/2.0 from grid
- src/evaluation/final_eval.py: added relationship docstring
- results/phase1/tuning/bm25_param_sweep.csv: cleaned to 20 valid rows
- [Session 3] src/data/__qrels_builder_test.py: 11 qrels tests (all pass)
