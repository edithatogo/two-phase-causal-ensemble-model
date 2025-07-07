# Chronus Causus Development TODO

- [X] **Phase 0: Project Setup & Core Structure**
    - [X] Item 4: Dependency Management - Ensure `pyproject.toml` includes all necessary dependencies. (Completed)
    - [X] Item 1: Initialize Package Structure - (`core/`, `methods/`, `ensemble/`, `utils/`, `tests/` exist. `lib/` also exists, TBD if needed for this project specifically or a remnant). (Status: Mostly complete, all specified dirs exist)
    - [X] Item 2: Basic Project Files - (`pyproject.toml` exists, `LICENSE` exists, `chronus_causus/README.md` created). (Status: Completed)
    - [ ] Item 3: Testing Framework setup - `pytest` in dev dependencies, `tests/` dir exists. Further configuration/validation when first tests for `chronus_causus` are written. (Status: Partially complete)
- [ ] **Phase 1: Base Causal Discovery Algorithms**
    - [ ] **CCM: Implement `CCMDiscoverer` (`chronus_causus/methods/ccm.py`)**
        - [X] Create `ccm.py` with class structure. (Completed)
        - [X] Analyze `demostration/utilities_ccm.py`. (Completed)
        - [X] Add identified parameters to `CCMDiscoverer.__init__`. (Completed)
        - [X] Implement `CCMDiscoverer.fit()` method based on `model_ccm` from `utilities_ccm.py`. (Completed: Core logic including skccm calls, convergence check, and causal matrix population).
        - [ ] Add unit tests for `CCMDiscoverer`. (In Progress: Basic tests for init, helpers, and fit structure created).
    - [ ] NTE: Implement `NTEDiscoverer` (`chronus_causus/methods/nte.py`)
    - [ ] PCMCI+: Implement `PCMCIDiscoverer`
    - [ ] GC: Implement `GCDiscoverer`
- [ ] **Phase 2: Ensemble Logic**
    - [ ] L1 Ensembler
    - [ ] L2 Ensembler / `CausalEnsemble` class
- [ ] **Phase 3: Utilities & Supporting Modules**
    - [ ] Data Preprocessing
    - [ ] Partitioning
    - [ ] Evaluation
    - [ ] Plotting
- [ ] **Phase 4: Testing and Documentation**
    - [ ] Unit & Integration Tests
    - [ ] Examples
    - [ ] Documentation

**Next Immediate Action:**
- Ensure `pyproject.toml` has all dependencies from Roadmap (Phase 0, Item 4).
