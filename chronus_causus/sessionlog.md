# Chronus Causus Development Session Log

## Session: 2024-07-25

- **Start Time:** (Approximate time of this session's commencement)
- **User Request:** Implement the `chronus_causus` library based on `roadmap.md`, step-by-step. Maintain `todo.md`, `sessionlog.md`, and `AGENTS.md`.
- **Agent Actions:**
    - Confirmed understanding of the task (develop within the existing `chronus_causus` folder).
    - Set a new plan to incorporate file creation and roadmap implementation.
    - Created `chronus_causus/todo.md` with initial tasks.
    - Created `chronus_causus/sessionlog.md`.
    - Created `chronus_causus/AGENTS.md`.
    - Read `chronus_causus/pyproject.toml`.
    - Updated `chronus_causus/pyproject.toml` to include all dependencies from Phase 0, Item 4 of the roadmap.
    - Updated `chronus_causus/roadmap.md` to mark Phase 0, Item 4 as complete.
    - Reviewed other Phase 0 items (Package Structure, Basic Project Files, Testing Framework).
    - Updated `chronus_causus/todo.md` with findings: `chronus_causus/README.md` needs creation.
    - Created `chronus_causus/README.md` with placeholder content.
    - Updated `chronus_causus/todo.md` to mark Phase 0, Item 2 (Basic Project Files) as complete and Item 1 (Package Structure) as mostly complete.
- **Commit:** `Initialize chronus_causus development tracking and Phase 0 progress` (branch `chronus-causus-init`)

## Session: 2024-07-25 (Continued)
- **User Request:** Proceed with implementing the roadmap, starting with Phase 1 (CCM).
- **Agent Actions:**
    - Set new plan for initial CCM implementation.
    - Created `chronus_causus/methods/ccm.py` with `CCMDiscoverer` class structure.
    - Read and analyzed `demostration/utilities_ccm.py`.
    - Updated `chronus_causus/todo.md` with CCM sub-tasks.
- **Commit:** `feat: Initial structure for CCMDiscoverer` (branch `feat/ccm-discoverer-init`)

## Session: 2024-07-25 (Continued)
- **User Request:** Continue CCM implementation.
- **Agent Actions:**
    - Set new plan for updating `CCMDiscoverer.__init__` and starting `fit` method.
    - Updated `CCMDiscoverer.__init__` with new parameters (`split_percent`, `max_lib_size_iter`, `convergence_error_num`, `convergence_threshold`).
    - Implemented initial part of `CCMDiscoverer.fit()`: input validation, time series embedding, feature pair iteration, and train/test splitting.
    - Updated `chronus_causus/todo.md` to reflect progress.
- **Commit:** `feat: Update CCMDiscoverer params and initial fit structure` (branch `feat/ccm-fit-structure`)

## Session: 2024-07-25 (Continued)
- **User Request:** Continue CCM `fit` method implementation.
- **Agent Actions:**
    - Set new plan to implement core CCM logic, convergence, and matrix population.
    - Added `_error()` static method to `CCMDiscoverer`.
    - Implemented core `skccm` calls (`CCM()`, `fit()`, `predict()`, `score()`) in `CCMDiscoverer.fit()`.
    - Implemented convergence check logic for scores in `fit()`.
    - Implemented population of `self.causal_matrix_` in `fit()`.
    - Ensured `math` import.
    - Updated `chronus_causus/todo.md` to mark `fit()` method as complete.
- **Commit:** `feat: Implement CCMDiscoverer fit method core logic` (branch `feat/ccm-fit-logic-complete`)

## Session: 2024-07-25 (Continued)
- **User Request:** Add unit tests for `CCMDiscoverer`.
- **Agent Actions:**
    - Set new plan for creating `test_ccm.py` and adding initial test cases.
    - Created `chronus_causus/tests/test_ccm.py` with basic structure.
    - Implemented `test_instantiation`, `test_error_method`, and `test_validate_input`.
    - Implemented `test_fit_runs_basic` to check `fit` method execution and attribute setting.
    - Updated `chronus_causus/todo.md` to reflect unit test progress.
    - Attempted to run tests, failed due to missing dependencies (numpy).
    - Revised plan to include `pip install ./chronus_causus`.
    - `pip install` failed due to invalid URLs in `pyproject.toml`.
    - Updated `pyproject.toml` with example.com URLs.
    - `pip install` failed again due to invalid author email format in `pyproject.toml`.
    - Updated `pyproject.toml` with example author info.
    - `pip install` failed again due to `skccm>=0.1.3` not found.
    - Updated `pyproject.toml` to `skccm>=0.2.dev0`.
    - `pip install ./chronus_causus` succeeded.
    - Ran unit tests: `python -m unittest chronus_causus/tests/test_ccm.py` - All 4 tests passed.
