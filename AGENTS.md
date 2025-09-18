# Repository Guidelines

## Project Structure & Modules
- `alphagomoku/`: Core package
  - `env/` (Gymnasium Gomoku), `model/` (DW-ResNet‑SE), `mcts/`, `tss/`, `selfplay/`, `train/`, `eval/`, `utils/`.
- `scripts/`: Training and utility entry points (e.g., `train.py`).
- `tests/`: `unit/`, `integration/`, `performance/` with pytest markers.
- `docs/`: Technical specs (TSS, optimizations, system design).
- `configs/`, `data/`, `checkpoints/`, `runs/`: Config, datasets, models, logs.

## Build, Test, and Dev Commands
- Install: `pip install -r requirements.txt && pip install -e .`
- Quick tests: `python -m pytest tests/unit -v`
- Full tests: `pytest -v` or with markers, e.g., `pytest -m "not slow"`.
- Format: `black alphagomoku/ tests/ && isort alphagomoku/ tests/`
- Train (Makefile): `make train` or `make train-fast` (see `Makefile`).

## Coding Style & Naming
- Python 3.12; 4‑space indentation; type hints required for public APIs.
- Naming: modules/files `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`.
- Docstrings: concise, Google/NumPy style; include shapes/dtypes where relevant.
- Imports: standard → third‑party → local; enforce with `isort`.
- Formatting: run `black` before PRs; keep changes minimal and focused.

## Testing Guidelines
- Framework: `pytest` with markers: `unit`, `integration`, `performance`, `slow`, `mcts`, `tss`, `training` (see `pytest.ini`).
- Naming: files `test_*.py`; classes `Test*`; functions `test_*`.
- Run subsets: `pytest -m unit`, `pytest tests/integration -v`.
- Aim for meaningful coverage on new/changed code; add regression tests for bugs.

## Commit & Pull Requests
- Commits: clear, imperative subject (≤72 chars), scoped (e.g., "mcts: batch leaf eval fix").
- PRs: description of change, rationale, linked issue, screenshots/plots if applicable (e.g., training curves), and benchmarks for perf‑sensitive paths.
- Checklist: `black`/`isort` clean, tests added/updated, `pytest -v` passes locally, docs updated (`docs/` or README) when behavior or APIs change.

## Security & Configuration Tips
- macOS training: `export KMP_DUPLICATE_LIB_OK=TRUE`; MPS fallback: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Control CPU threads: `export OMP_NUM_THREADS=1` for reproducibility.
- LMDB/Zarr storage: configure via script flags (e.g., `--map-size-gb`); avoid committing large data/checkpoints.

## Architecture Overview (Brief)
- Unified search stack: Endgame solver (late game) → TSS (tactics) → MCTS (general play).
- Training loop: self‑play → replay buffer (augmented) → supervised updates (policy+value).
