# src/tuning -- Hyperparameter sweep utilities
#
# Public API (import from here):
#   SweepResult         - dataclass wrapping sweep results (.to_dataframe(), .best, .baseline())
#   field_ablation      - query field selection (BM25, NDCG@100 criterion)
#   run_bm25_sweep      - BM25 (k1, b) grid via 5-fold CV
#   run_lmjm_sweep      - LM-JM lambda grid via 5-fold CV
#   run_lmdir_sweep     - LM-Dir mu grid via 5-fold CV
#   run_encoder_sweep   - dense encoder comparison (exact cosine, no OpenSearch)
#   run_rrf_sweep       - RRF pair × k grid via 5-fold CV
#   run_cv              - generic k-fold CV helper (used internally + in notebooks)
#   make_folds          - topic k-fold splitter
#
# All grids + constants come from the notebook (no hardcoded values here).

from src.tuning.sweeper import (
    SweepResult,
    field_ablation,
    run_bm25_sweep,
    run_lmjm_sweep,
    run_lmdir_sweep,
    run_encoder_sweep,
    run_rrf_sweep,
)
from src.tuning.cv_utils import run_cv, make_folds
