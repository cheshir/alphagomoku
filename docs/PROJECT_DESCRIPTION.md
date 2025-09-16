# Technical Specification — Gomoku AI (AlphaZero-style)

## Project Goals
- Build a strong Gomoku bot for 15×15 (classic; Swap2 as a config).
- Single model, single backend for all difficulties.
- Difficulties: easy, medium, strong. Difficulty is chosen at game creation and cannot be changed mid-game.
- Inference target: arm64 Linux, 4 vCPU, 8 GB RAM, CPU-only by default. Optional GPU inference (CUDA/MPS/CoreML) via runtime provider switch without architectural forks.
- Training target: Apple Silicon M1 Pro, 16 GB RAM (MPS/Metal).

—

# A) Training Specification

## A.1 Core Approach
- AlphaZero pipeline: self-play → MCTS (PUCT) → collect (state, π, z) → train NN (policy+value).
- Hybrid modules for strength: Threat-Space Search (TSS), endgame solver (αβ), and opening book.
- Single CNN model. Difficulties are handled at inference runtime.

## A.2 Environment
- Gymnasium Env (gomoku-v0).
- Board size: configurable (default 15×15).
- Observation: current player’s stones=1, opponent=−1, empty=0.
- Action space: Discrete(N²) with legal action mask.
- Terminal conditions: 5 in a row or draw.
- Options: rule profile (classic / swap2).

## A.3 Network Architecture
- Backbone: DW-ResNet-SE (depthwise-separable ResNet + Squeeze-and-Excitation).
- 10–14 residual blocks, width 64, ~8–12M parameters.
- Inputs: own stones, opponent stones, last move, side-to-move, pattern maps (open-three, open-four, broken-four, double-three-four).
- Heads: 
  - Policy: logits for N² cells.
  - Value: scalar in [−1, 1].
- Training precision: FP32; MPS acceleration on macOS.

## A.4 MCTS (self-play)
- PUCT with cpuct≈1.6–2.0.
- Simulations: 256–512 on small boards; 800–1600 on 15×15.
- Batched leaf eval (64–256).
- Persistent tree, Zobrist hashing, TT.

## A.5 TSS & Endgame Solver (training)
- TSS depth 4–6 plies during self-play.
- Endgame αβ solver active when ≤ 20 empties.

## A.6 Data & Augmentation
- Self-play tuples (state, π, z).
- Augment with 8 symmetries.
- Hard example mining (10–20%).
- Disk-backed replay buffer (LMDB/Zarr) with 5–10M positions.

## A.7 Optimisation
- Loss: cross-entropy + MSE + L2.
- Optimiser: AdamW / SGD+Momentum.
- Batch size: 1–4k.
- Checkpoint: {model, optimizer, sched, config}.

## A.8 Curriculum & Promotion
- 9×9 → 13×13 → 15×15.
- Best-model promotion via evaluation matches.

## A.9 Evaluation
- Position test-suite (forced wins/blocks).
- Elo ladder vs heuristics and prior checkpoints.
- Human evaluation with fixed rules.

## A.10 Training Environment Setup
- macOS + PyTorch (MPS).
- Exports: PyTorch .pt, ONNX (FP32/FP16).

## A.11 Training Deliverables
- Source: env/, model/, mcts/, tss/, endgame/, selfplay/, train/, eval/.
- Scripts: selfplay.py, train.py, export_onnx.py.
- Checkpoints: model_best.pt, model_best.onnx (FP16).
- Docs: TRAINING.md (setup, hyperparams, curriculum).

—

# B) Inference Specification

## B.1 Target Platform
- Primary: arm64 Linux, 4 vCPU / 8 GB RAM, CPU-only.
- Latency goals (15×15):
  - Easy: 80–150 ms.
  - Medium: 350–800 ms.
  - Strong: 1–3 s (p50 ≤2.0 s).
- Memory budget: ≤4.5 GB RSS.

## B.2 Runtime & Model
- ONNX Runtime with providers: CPU (default), CUDA, CoreML/MPS.
- Model: model_best.onnx FP16 (fallback FP32).
- Threading tuned to vCPU count.

## B.3 Difficulties (immutable per game)
- Difficulty selected in POST /new, cannot be changed mid-game.

### Easy
- MCTS sims: 48.
- Root noise: α=0.5, ε=0.35.
- Temp: τ=1.2 (≤10 moves).
- Selection: top-p (p=0.9) + 3% blunder from top-k.
- Book depth: 2.
- TSS: off (or depth ≤2).
- Endgame solver: off.
- Time cap: 120 ms.

### Medium
- MCTS sims: 384.
- Root noise: α=0.35, ε=0.25.
- Temp: τ=0.8 (≤8).
- Selection: argmax.
- Book depth: 4.
- TSS: depth 3–4, 100 ms cap.
- Endgame solver: on at ≤14 empties.
- Time cap: 700 ms.

### Strong
- MCTS sims: 1600 (800–2000 adaptive).
- cpuct=1.8.
- Root noise: α=0.25, ε=0.25.
- Temp: τ=0.5 (≤6).
- Selection: argmax.
- Book depth: 6.
- TSS: depth 6–7, 300 ms cap, priority override.
- Endgame solver: on at ≤20 empties.
- Time cap: 2500–3000 ms.

## B.4 Search Stack
- Persistent MCTS tree with TT.
- TSS check before MCTS rollout.
- Endgame αβ solver for late game.
- Opening book for first plies.

## B.5 API
- POST /new: {board_size, rules, difficulty} → returns game_id.
- POST /move: {game_id, row, col} → returns {ai_move, status, stats}.
- GET /state/{game_id}.
- POST /resign.
- Errors: 404 unknown game, 409 difficulty change, 422 illegal move.

## B.6 Concurrency & State
- Async server (FastAPI + uvicorn).
- Per-game state: board + persistent tree.
- Global leaf eval queue for batching.

## B.7 Resource Budget
- Model + runtime: 0.2–0.5 GB.
- MCTS + TT: 1.0–2.2 GB.
- TSS/solver: 0.2–0.6 GB.
- Buffers/overheads: 0.5–1.0 GB.
- Total: 2.0–4.3 GB.

## B.8 Deployment
- Container: Debian slim, image <300 MB.
- Env flags: ORT providers, TT size, batch size.
- Healthcheck: /healthz warms model.
- Metrics: Prometheus (latency, sims, tt_hit_rate, RSS).

## B.9 Acceptance Criteria
- Easy: beats novices ≥70%, makes 5–10% designed blunders.
- Medium: ≥60% vs heuristics, ≥35–45% vs experienced humans.
- Strong: ≥85–90% vs experienced humans (50/50 first/second, ≤3s).
- Performance: meets latency/memory budgets.
- Stability: 24h run, no leaks (>1% drift/hr).

—

# C) Implementation Notes
- Move selection: temp-sample in opening, argmax after.
- TSS overrides MCTS if forced win/defence found.
- Endgame triggers at configured empties threshold.
- Logging: reason (book, tss_forced, mcts_best, endgame, easy_blunder).
- GPU inference optional via ORT provider switch, no separate backend.

—

# D) Deliverables

## Training
- Source code: modules for env/model/mcts/tss/endgame/selfplay/train/eval.
- Scripts: selfplay, train, export.
- Checkpoints: model_best.pt, model_best.onnx.
- Docs: TRAINING.md.

## Inference
- Service code: FastAPI server, Dockerfile, configs.
- Models: model_best.onnx, opening_book.json.
- Tests: position suite, load test, human eval protocol.
- Docs: DEPLOY.md, API.md, METRICS.md.
