# Project Status — Phase 1 Audit
**Date:** 2026-04-04
**Scope:** Compare current src/ and phase1_search.ipynb against TODO_PROJ_STRUCTURE.md and TODO_NOTEBOOK_CELLS.md

---

## TASKS TO DO: 
### What EXISTS in src/ that the plan did NOT mention
- `src/data/query_builder.py` — `build_query(topic, field)` function re-exported from evaluator.
  Plan showed it under evaluation/evaluator.py; it was correctly split into its own file. Good.
  » we will need this query builder for future phases yes? should it be in some more clear place? maybe in the retrievers, o or data or other or some new foldeer or file?? or leave it where it is now is clear enough?? 

- `src/evaluation/final_eval.py` — contains `PHASE_1_BEST_CONFIG` dict + `_BEST_PARAMS` imported
  from index_builder. Also has `add_medcpt_field()` and `populate_medcpt_embeddings()` which
  are index_builder concerns leaking into final_eval.
    »» about constants in the src folder, i don't want any constants there. i want all constants to come from the notebooks. what we can have yes is some constants to be used for local unit and integration tests or for some default fields for args (and in this cases only the minimum necessary, example lets not have defaults to force the creation of some index or some field or anything like that. if things are not sent so we don't create. lets be transparent)
    »» about this: `add_medcpt_field()` and `populate_medcpt_embeddings()` - we don't need to have these concrete and sort of hardcooded methods. we already have a indexing method and a create field method that does the embeding and creation of fields for any model given. so lets delete these functions ( `add_medcpt_field()` and `populate_medcpt_embeddings()`) and lest use the general functions we have and that work well 
  >>> GAP 1: `add_medcpt_field()` and `populate_medcpt_embeddings()` belong in
      `src/indexing/index_builder.py`, not in `final_eval.py`. The notebook cell imports them
      from final_eval (cell 58), which is a confusing import path. Not breaking, but misplaced.
      » lets remove these funcitons and use the general indexing and field creationg functions we have in the respective folders 
- `src/tuning/sweeper.py` has `field_ablation()`, `run_rrf_sweep()`, `run_encoder_sweep()`
  in addition to the BM25/LM-JM/LM-Dir sweeps the plan described. These are real and correct.
- `src/tuning/tuning_plots.py` has `plot_tuning_summary()` and `plot_pr_interpretation()`
  which were planned but also `plot_rrf_sweep()` and `plot_field_ablation()` which were not.
  These are correct additions.
- `src/indexing/index_builder.py` exports `_BEST_PARAMS` — a module-level dict of locked
  hyperparams. This is a reasonable choice but means best params are hardcoded inside a
  src/ file rather than the notebook constants. See GAP 2 below.
  »» again, these constants in src are not that relevant. they are just for local testing and strictly necesssary defaults, and if not really needed they should be deleted. 

### GAP 1 — Minor: misplaced functions in final_eval.py
`add_medcpt_field()` and `populate_medcpt_embeddings()` are index-building operations.
They live in `final_eval.py` but should be in `index_builder.py`.
Impact: low (nothing breaks). Worth fixing before Phase 2 if those functions are reused.
Action: move to index_builder.py, update imports in notebook cell 58 and final_eval.py.
» just delete them and lets use the general indexing and field creation functions we have in the respective folders

### GAP 2 — Minor: `_BEST_PARAMS` in index_builder.py creates dual sources of truth
The notebook constants cell (cell 6) defines `BM25_K1_B_BEST`, `LMDIR_MU_BEST`, etc.
But `src/indexing/index_builder.py` also has a `_BEST_PARAMS` dict with the same values.
And `final_eval.py` imports from `index_builder._BEST_PARAMS` to build `PHASE_1_BEST_CONFIG`.
So the locked values exist in: (a) the notebook constants cell and (b) index_builder.py.
If one is updated the other may diverge silently.
Impact: low for Phase 1 (values already match). Could cause confusion in Phase 2.
Action: not urgent — leave as-is for now. When implementing Phase 2, the notebook will
define its own constants and import `PHASE_1_BEST_CONFIG` from final_eval.py directly.
That import is the right pattern (single source) and already works.
» again, lets have constants in the src folder all starting with _CONSTANT_X   and lets have clear commeents that the constants that are source of truth are the ones in the notebooks and these are just for local testing and strictly necesssary defaults, and if not really needed they should be deleted.

---

## Notebook Audit — phase1_search.ipynb

### Section mapping: plan vs notebook

| Plan section                                | Notebook section                                                   | Status                                           |
| ------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------ |
| §0 Cover + ToC                              | Cell 1 (cover only; ToC says "TODO")                               | PARTIAL — ToC missing                            |
| §1 Infrastructure (setup + constants)       | Cells 1-6 (§1.1, §1.2 setup, §1.3 constants MD, §1.4 constants PY) | DONE — excellent                                 |
| §2 Data (corpus/topics/splits/qrels)        | Cells 7-8 (§1.3 data MD + §1.3 data PY)                            | DONE — very rich                                 |
| §3 Index creation (theory + code)           | Cells 9-10 (§1.4 index MD + code)                                  | DONE — excellent                                 |
| §4 Retrieval baselines                      | Cells 11-22 (§2 + 2.1-2.5)                                         | DONE — excellent                                 |
| §5 Evaluation metrics reference card        | Cells 23-24 (§3 intro + §3.0 metrics)                              | DONE — excellent                                 |
| §5.10 Metric sanity check PY cell           | NOT PRESENT                                                        | GAP 3                                            |
| §6 Tuning (query ablation + 4 model sweeps) | Cells 25-40 (§3.1-§3.8)                                            | DONE — goes deeper than plan                     |
| §6.13 Tuning summary MD cell                | Cell 37 (§3.7)                                                     | DONE                                             |
| §7 Run file generation                      | Cells 44-45 (§4.1)                                                 | DONE — generates 10 run files (baseline + tuned) |
| §8.1-8.3 Eval protocol + metric tables      | Cells 46-47 (§4.2)                                                 | DONE                                             |
| §8.4 Mean PR curves                         | Cells 51-55 (§4.3-§4.4)                                            | DONE — also has tuning gain chart                |
| §8.5 Results discussion MD                  | Cell 56 (§5)                                                       | DONE — very rich                                 |
| §8.6 AP distribution boxplot                | Cell 55 (inside §4.4)                                              | DONE                                             |
| §8.7-8.10 3-query PR curve analysis         | Cells 52-53 (§4.5)                                                 | DONE — best/median/worst                         |
| §9 Error & corpus analysis                  | NOT PRESENT                                                        | GAP 4                                            |
| §10 Report-ready summary cells              | NOT PRESENT                                                        | GAP 5                                            |

### What the notebook has that the plan did NOT describe
- **§3.8 RRF Pair Grid Search** (cells 40-42) — exhaustive sweep over 8 retriever pairs × 3
  RRF k values, including KNN × KNN pairs. Not planned but very well done. The analysis cell
  (§3.8.1) explains why no RRF pair beats MedCPT solo — this is a real finding.
- **§4.3 Baseline vs Tuned plots** (cell 49) — grouped bar chart across all 5 metrics +
  tuning-gain delta chart. Not in the plan; excellent addition for the report.
- **§4.6 Locked configuration table** (cells 57-58) — explicit locked-params table for
  Phase 2 handoff. Good engineering discipline.
- **PR interpretation guide** (cell 51) — a 6-panel "how to read PR curves" plot with
  calibration shapes before showing the real curves. Not planned; excellent pedagogical tool.
- **§5 Discussion & Findings** (cell 56) — a fully written results discussion section with
  a detailed analysis table, including binary vs graded evaluation rationale, hard/easy topics
  with actual topic IDs and AP scores, and a data quality note. Goes far beyond what the plan
  described for "§8.5 Results discussion".

### GAP 3 — Missing: Metric sanity check cell (§5.10 in plan)
The plan called for a PY cell that reproduces the Lab03 toy example (AP(A)=1.0, AP(B)=0.7095)
as a "PASS/FAIL" confidence check for the implementation.
The metrics ARE correct (confirmed in PROJECT_PLAN.md: "Lab03 toy example reproduced exactly"),
and tests in src/evaluation/metrics.py already do this.
But it is not shown in the notebook — a reader has no in-notebook confirmation the metrics impl is correct.
Impact: educational gap only. No correctness issue.
Action: add a short PY cell after §3.0 metrics that runs the Lab03 toy example inline.
»» so lets do this


### GAP 4 — Missing: §9 Error & Corpus Analysis
The plan described 4 cells: worst-performing queries table, IDF analysis scatter plot,
corpus vocabulary coverage for bottom-5 queries, and a conclusions MD cell.
These are NOT in the notebook.
Impact: moderate — these analyses are useful for the report discussion and explain
*why* certain queries fail. They would strengthen the Phase 1 evaluation section of the report.
Action: add after the current §5 Discussion, before the report-summary cells.
These are all inline notebook-only code (no new src/ functions needed except possibly
a term IDF calculation which is a few lines of numpy using the existing corpus).
»» LETS NOT IMPLEMENT THIS. CONFIRM IF THE PROJECT GUIDES EXPLICITLY SAY TO DO THIS: 
    C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\prof_guide_professor_notes.md
    C:\Users\franc\Desktop\NLP_Biomedical_Agent\_proj_guides\proj_guide_NLPS_-_TREC_2025_BioGen.md  
IF NOT, SO LETS NOT IMPLEMENT THIS


### GAP 5 — Missing: §10 Report-Ready Summary Cells
The plan described 3 MD cells with polished text ready to copy into the report:
- Experimental Setup (datasets, metrics, protocol, locked hyperparams table)
- Results & Discussion (comparison table + PR curve interpretation + 3-query analysis)
- Limitations (qrel quality, corpus scope, zero-relevant topics, IDF at small corpus scale)
None of these are present.
Impact: low for the notebook itself (§5 Discussion already covers the content informally),
but having report-ready cells would save time when writing the final report.
Action: add 3 MD cells at the end of the notebook. The content already exists
scattered across §3.0, §4.2, §4.3, §5 — these cells just structure it for copy-paste.
»» LETS NOT DO THIS. WE WILL WRITE THE REPORT DIRECTLY AFTER. 
»» LETS JUST HAVE SOME CONCISE NOTEBOOK CONCLUSIONS OF THE PHASE 1 FINDINGS IN THE DISCUSSION SECTION AND THEN WE WILL EXPAND ON THIS IN THE REPORT.

### GAP 6 — ToC placeholder still says "TODO"
Cell 1 has "## Index\nTODO" — the table of contents was never filled in.
Impact: low — purely cosmetic, but visible to the reader/professor.
Action: fill in the ToC (a simple numbered list pointing to each ## section).

»» LETS DO THIS LAST. AND LEST DO THE INDEX WITH ONLY THE # AND ## LEVELS SECTIONS SO WE HAVE A CLEAR HIGH AND SECOND LEVELS OF THE NOTEBOOK. 
---
