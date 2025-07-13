# Roadmap for chronus_causus Library

This document outlines the high-level implementation plan for the `chronus_causus` library, a Python package for timeseries causal discovery based on the paper "A Data-Driven Two-Phase Multi-Split Causal Ensemble Model for Time Series" and existing demonstration code.

## Core Objectives:
*   Develop a modular, extensible, and user-friendly library.
*   Incorporate scikit-learn-like API design principles for ease of use and integration.
*   Refactor and build upon the robust methodology presented in the research paper and `demostration/` codebase.

## Phased Implementation Plan:

**Phase 0: Project Setup & Core Structure**
1.  **Initialize Package Structure:**
    *   Create directories: `chronus_causus/core/`, `chronus_causus/methods/`, `chronus_causus/ensemble/`, `chronus_causus/utils/`, `chronus_causus/tests/`.
2.  **Basic Project Files:**
    *   `setup.py` or `pyproject.toml` for packaging and dependency management.
    *   `README.md` (project-level).
    *   `LICENSE` (Apache 2.0, already created within `chronus_causus/`).
3.  **Testing Framework:**
    *   Set up `pytest`.
4.  **Dependency Management:** (Completed: All listed dependencies added to `pyproject.toml`)
    *   Define dependencies (e.g., `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `tigramite`, `skccm`, `networkx`, `matplotlib`, `scipy`, `numba`).

**Phase 1: Base Causal Discovery Algorithms**
*   For each base method (CCM, NTE, PCMCI+, GC):
    1.  **Dedicated Module:** Create, e.g., `chronus_causus/methods/ccm.py`.
    2.  **Estimator Class:** Implement a class (e.g., `CCMDiscoverer`) with a scikit-learn-like API:
        *   `__init__(self, **params)`: Method-specific hyperparameters.
        *   `fit(self, X_partition, feature_names=None)`: Computes causal matrix for a single data partition. Stores result (e.g., `self.causal_matrix_`).
    3.  **Adapt Existing Code:** Refactor logic from `demostration/utilities_*.py` and `CPU_TE.py/GPU_TE.py`.
    4.  **Independent Usability:** Ensure classes can be used standalone.

**Phase 2: Ensemble Logic**
1.  **L1 Ensembler (`chronus_causus/ensemble/l1_ensemble.py`):**
    *   Develop a class/mechanism to apply a single base algorithm (from Phase 1) to multiple data partitions.
    *   Implement the GMM-based ensembling from `utilities_ensemble.py l1_ensemble` function.
    *   Output: L1 ensembled causal matrix and "accuracy index" for that algorithm.
2.  **L2 Ensembler / Main Ensemble Estimator (`chronus_causus/ensemble/causal_ensemble.py`):**
    *   Create the main `CausalEnsemble` class.
    *   `__init__(self, estimators: list, num_parts: int, l2_threshold: float, **kwargs)`:
        *   `estimators`: List of configured base algorithm instances, e.g., `[('ccm', CCMDiscoverer(lag=1)), ...]`.
        *   `num_parts`: Number of data partitions.
        *   Other L2 ensembling parameters.
    *   `fit(self, X, feature_names=None)`:
        1.  Implement data partitioning.
        2.  For each base estimator, compute its L1 ensembled matrix and accuracy index.
        3.  Apply L2 ensembling logic (from `utilities_ensemble.py l2_ensemble`).
        4.  Apply "optimization" step to remove indirect links.
        5.  Store final `self.causal_graph_` and `self.strength_matrix_`.
    *   `get_causal_graph()`: Method to return the final graph.

**Phase 3: Utilities & Supporting Modules**
1.  **Data Preprocessing (`chronus_causus/utils/preprocessing.py`):**
    *   Stationarity tests (ADF), differencing, data validation.
2.  **Partitioning (`chronus_causus/utils/partitioning.py`):**
    *   Implement the overlapping partitioning strategy from the paper.
    *   Consider future extensibility for other strategies.
3.  **Evaluation (`chronus_causus/utils/evaluation.py`):**
    *   Implement the evaluation index from the paper.
    *   Standard metrics (Precision, Recall, F1 if ground truth available).
4.  **Plotting (`chronus_causus/utils/plotting.py`):**
    *   Adapt `demostration/utilities_plotting.py`.

**Phase 4: Testing and Documentation**
1.  **Unit & Integration Tests:** Comprehensive tests for all modules.
2.  **Examples:** Develop example scripts/notebooks.
3.  **Documentation:** API documentation (docstrings) and usage guides.

## Potential Considerations:
*   **Full Paper Access:** Obtain and review the full PDF of arXiv:2403.04793 for complete methodological details.
*   **NTE Implementation Details:** Clarify best way to integrate or re-implement TE computation (Kraskov estimation).
*   **GPU Support:** Initially focus on CPU, with GPU as a potential later enhancement for NTE.

This roadmap will be updated as development progresses.
