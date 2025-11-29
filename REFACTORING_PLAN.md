# AlphaGomoku Refactoring and Optimization Plan

**Goal**: Refactor project to best practices, fix critical issues, optimize for fast/cheap training while maintaining strong play.

**Timeline**: Phased approach with testing at each step

---

## Phase 1: Project Audit and Documentation ✓ IN PROGRESS

### Current State Analysis

**Project Structure:**
```
alphagomoku/
├── alphagomoku/          # Core ML package (84 Python files)
│   ├── env/             # Gomoku environment
│   ├── model/           # Neural network
│   ├── mcts/            # Monte Carlo Tree Search
│   ├── tss/             # Threat-Space Search
│   ├── endgame/         # Endgame solver
│   ├── selfplay/        # Self-play workers
│   ├── train/           # Training pipeline
│   ├── eval/            # Evaluation (incomplete!)
│   ├── search/          # Unified search
│   └── utils/           # Utilities
├── apps/
│   ├── backend/         # FastAPI inference server
│   └── frontend/        # Vue.js UI
├── scripts/             # Training and test scripts
├── tests/               # Unit, integration, performance tests
├── docs/                # Technical documentation (25+ MD files)
└── notebooks/           # Jupyter notebooks (2)
```

**Key Findings:**

### ✅ Good
1. Solid architecture: DW-ResNet-SE, MCTS, TSS, endgame solver
2. Comprehensive documentation (25+ MD files)
3. Test coverage: unit, integration, performance
4. Inference server + UI working
5. Hardware auto-configuration
6. LMDB replay buffer with augmentation

### 🚨 Critical Issues
1. **No evaluation during training** - flying blind
2. **Model too large** (5M params) - slow training (3-4h/epoch)
3. **MCTS sims too low** (150) - noisy training data
4. **Replay buffer too small** (500K) - forgetting early lessons
5. **LR schedule suboptimal** - no warmup, min_lr too high
6. **Data filtering too aggressive** - limiting network learning
7. **No Elo tracking or win rate monitoring**
8. **Training restarts without explanation** (wasted compute)

### ⚠️ Code Quality Issues
1. Inconsistent error handling
2. Some hardcoded values (not in config)
3. Memory management issues (MPS cache cleanup)
4. No type hints in some modules
5. Duplicate functionality (multiple test scripts)

---

## Phase 2: Refactor Project Structure

### 2.1 Core Package Refactoring

**Goal**: Best practices, type hints, clear separation of concerns

#### Actions:
- [ ] Add comprehensive type hints to all modules
- [ ] Extract hardcoded values to config files
- [ ] Consolidate config management (single source of truth)
- [ ] Improve error handling and logging
- [ ] Remove dead code and unused files

#### Files to Refactor:
```python
alphagomoku/
├── config.py              # NEW: Central configuration
├── model/
│   └── network.py         # Add type hints, extract hyperparams
├── train/
│   ├── trainer.py         # Extract magic numbers, better logging
│   ├── data_buffer.py     # Improve error messages
│   └── data_filter.py     # Simplify filtering logic
└── mcts/
    └── mcts.py            # Better batch handling
```

### 2.2 Evaluation Framework (NEW)

**Goal**: Comprehensive evaluation during training

#### New Structure:
```python
alphagomoku/eval/
├── __init__.py
├── evaluator.py           # EXISTS but incomplete
├── baseline.py            # NEW: Fixed baseline opponents
├── elo_tracker.py         # NEW: Elo rating system
├── tactical_tests.py      # NEW: Tactical position suite
└── win_rate_tracker.py    # NEW: Win rate against checkpoints
```

### 2.3 Configuration Management

**Goal**: Single source of truth for all hyperparameters

#### New Files:
```python
configs/
├── model_configs.py       # Model architecture presets
│   ├── small (1.2M)       # 10 blocks, 96 channels
│   ├── medium (2.5M)      # 14 blocks, 128 channels
│   └── large (5M)         # 18 blocks, 192 channels
├── training_configs.py    # Training hyperparameters
│   ├── fast_iteration
│   └── production
└── inference_configs.py   # Inference settings by difficulty
```

### 2.4 Scripts Organization

**Goal**: Clear separation of training, evaluation, testing

```bash
scripts/
├── train.py               # Main training (KEEP, refactor)
├── evaluate.py            # NEW: Comprehensive evaluation
├── export_model.py        # NEW: ONNX export
├── test_tactical.py       # KEEP
└── benchmarks/            # NEW: Performance benchmarks
    ├── bench_mcts.py
    ├── bench_inference.py
    └── bench_selfplay.py
```

---

## Phase 3: Add Evaluation Framework

### 3.1 Baseline Opponents

Create fixed-strength opponents for consistent evaluation:

```python
# alphagomoku/eval/baseline.py

class BaselineOpponent:
    """Fixed-strength opponent for evaluation"""
    RANDOM = "random"          # Random legal moves
    HEURISTIC = "heuristic"    # Simple pattern-based
    MCTS_100 = "mcts_100"      # Pure MCTS, 100 sims
    MCTS_400 = "mcts_400"      # Pure MCTS, 400 sims
```

### 3.2 Elo Tracking

```python
# alphagomoku/eval/elo_tracker.py

class EloTracker:
    """Track model strength over epochs"""

    def __init__(self, initial_elo: int = 1500):
        self.elo_history = []
        self.current_elo = initial_elo

    def update(self, wins: int, losses: int, opponent_elo: int):
        """Update Elo after games"""
        ...
```

### 3.3 Tactical Test Suite

```python
# alphagomoku/eval/tactical_tests.py

TACTICAL_POSITIONS = [
    # Win in 1 moves
    {"board": [...], "solution": (7, 7), "type": "win_in_1"},

    # Defend immediate threats
    {"board": [...], "solution": (8, 8), "type": "defend"},

    # Double attack
    {"board": [...], "solution": (6, 9), "type": "double_attack"},
]
```

### 3.4 Integration with Training

```python
# In train.py, after each epoch:

if epoch % 5 == 0:
    # 1. Win rate vs baseline
    win_rate = evaluator.evaluate_vs_baseline(
        model, baseline="mcts_400", games=50
    )

    # 2. Tactical test performance
    tactics_score = evaluator.evaluate_tactics(
        model, tactical_positions
    )

    # 3. Elo update
    elo = elo_tracker.update(win_rate)

    # 4. Log to CSV and TensorBoard
    log_evaluation(epoch, win_rate, tactics_score, elo)
```

---

## Phase 4: Optimize Model Architecture

### 4.1 Model Size Presets

**Recommendation**: Start with SMALL config for fast iteration

```python
# configs/model_configs.py

MODEL_PRESETS = {
    "small": {
        "num_blocks": 10,
        "channels": 96,
        "params": "~1.2M",
        "use_case": "Fast iteration, 80% of large model strength"
    },
    "medium": {
        "num_blocks": 14,
        "channels": 128,
        "params": "~2.5M",
        "use_case": "Balanced training speed and strength"
    },
    "large": {
        "num_blocks": 18,
        "channels": 192,
        "params": "~5M",
        "use_case": "Maximum strength, slow training"
    }
}
```

### 4.2 Hardware Auto-Config Update

```python
# Update train.py:_get_hardware_config()

def _get_hardware_config(device: str, model_preset: str = "small") -> dict:
    """Get optimal config for device and chosen model size"""

    config = MODEL_PRESETS[model_preset].copy()

    # Adjust batch size and parallelization
    if device == 'cuda':
        if gpu_memory_gb >= 24:
            config['batch_size'] = 1024
            config['parallel_workers'] = 1
        elif gpu_memory_gb >= 12:
            config['batch_size'] = 512
            config['parallel_workers'] = 1

    elif device == 'mps':
        # Small model: use parallelization
        if model_preset == "small":
            config['batch_size'] = 256
            config['parallel_workers'] = 4
        else:
            config['batch_size'] = 512
            config['parallel_workers'] = 1

    return config
```

### 4.3 Model Comparison Tests

```python
# scripts/compare_models.py

def compare_model_sizes():
    """Compare small vs medium vs large models"""

    for preset in ["small", "medium", "large"]:
        model = GomokuNet.from_preset(preset)

        # Measure
        train_time = benchmark_training(model, epochs=5)
        strength = evaluate_strength(model, games=100)

        print(f"{preset}: {train_time:.1f}s/epoch, {strength:.1%} win rate")
```

---

## Phase 5: Fix Training Hyperparameters

### 5.1 MCTS Simulation Count

**Change**: Increase from 150 → 400-600 for training

```makefile
# Makefile

train:
	python scripts/train.py \
		--model-preset small \
		--mcts-simulations 400 \        # UP from 150
		--selfplay-games 100 \          # Fewer games, higher quality
		...
```

### 5.2 Learning Rate Schedule

```python
# Fix in train.py

parser.add_argument('--warmup-epochs', type=int, default=10)  # ADD warmup
parser.add_argument('--min-lr', type=float, default=1e-6)     # Lower min LR
```

### 5.3 Replay Buffer

```makefile
--buffer-max-size 5000000      # UP from 500K
--map-size-gb 32               # UP from 12GB
```

### 5.4 Data Filtering

**Simplify**: Remove aggressive tactical filtering, trust learning

```python
# alphagomoku/train/data_filter.py

def filter_only_illegal_moves(data):
    """Only filter truly illegal moves, let network learn the rest"""
    # Remove tactical pattern checking
    # Keep only: moves outside board, occupied squares
```

---

## Phase 6: Training Configuration Presets

### 6.1 Fast Iteration Config (Development)

```python
FAST_ITERATION = {
    "model_preset": "small",
    "mcts_simulations": 200,
    "selfplay_games": 50,
    "batch_size": 256,
    "epochs": 50,
    "eval_frequency": 5,
    "parallel_workers": 4,
    "expected_time_per_epoch": "10-15 min",
    "expected_total_time": "8-12 hours"
}
```

### 6.2 Production Config (Final Training)

```python
PRODUCTION = {
    "model_preset": "medium",
    "mcts_simulations": 600,
    "selfplay_games": 200,
    "batch_size": 512,
    "epochs": 200,
    "eval_frequency": 10,
    "parallel_workers": 1,
    "expected_time_per_epoch": "1-2 hours",
    "expected_total_time": "8-16 days"
}
```

---

## Phase 7: Testing Updates

### 7.1 Test Structure

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_model.py     # UPDATE: test all presets
│   ├── test_mcts.py      # UPDATE: test batch sizes
│   ├── test_eval.py      # NEW: test evaluation framework
│   └── ...
├── integration/          # End-to-end tests
│   ├── test_training.py  # NEW: test full training loop
│   └── test_eval.py      # NEW: test evaluation integration
└── benchmarks/           # Performance tests
    ├── test_inference_speed.py
    └── test_training_speed.py
```

### 7.2 Key Test Updates

```python
# tests/unit/test_model.py

def test_model_presets():
    """Test all model preset configurations"""
    for preset in ["small", "medium", "large"]:
        model = GomokuNet.from_preset(preset)
        assert model.get_model_size() > 0

        # Test forward pass
        batch = torch.randn(4, 5, 15, 15)
        policy, value = model(batch)
        assert policy.shape == (4, 225)
        assert value.shape == (4,)

# tests/integration/test_training.py

def test_training_loop_with_evaluation():
    """Test that evaluation runs during training"""
    # Run 5 epochs with eval_frequency=2
    # Assert evaluation metrics are logged
    ...
```

---

## Phase 8: Documentation Updates

### 8.1 Update README.md

- Add model preset recommendations
- Update training time estimates
- Add evaluation metrics explanation

### 8.2 Update Training Docs

- New: `docs/MODEL_SELECTION.md` - Guide to choosing model size
- New: `docs/EVALUATION.md` - How to interpret evaluation metrics
- Update: `docs/TRAINING_IMPROVEMENTS.md` - Latest best practices

### 8.3 Notebook Updates

```
train_colab.ipynb         # Update with new configs
train_universal.ipynb     # Update with evaluation cells
```

Add evaluation cells:
```python
# NEW CELL: Model Evaluation
from alphagomoku.eval import Evaluator

evaluator = Evaluator(model)
win_rate = evaluator.evaluate_vs_baseline("mcts_400", games=50)
print(f"Win rate vs MCTS-400: {win_rate:.1%}")
```

---

## Phase 9: Implementation Order

### Week 1: Foundation
1. ✅ Create this plan
2. Add evaluation framework (Phase 3)
3. Create model presets config (Phase 4.1)
4. Update tests for new evaluation (Phase 7.1)

### Week 2: Core Refactoring
5. Refactor configuration management (Phase 2.3)
6. Update model architecture with presets (Phase 4.2)
7. Fix training hyperparameters (Phase 5)
8. Run tests, fix breakages

### Week 3: Training Improvements
9. Integrate evaluation into training loop (Phase 3.4)
10. Update Makefile with new configs (Phase 5)
11. Test full training pipeline
12. Update documentation (Phase 8)

### Week 4: Validation
13. Run comparison: old vs new config (Phase 4.3)
14. Update notebooks (Phase 8.3)
15. Final integration tests (Phase 9)
16. Create migration guide

---

## Success Metrics

### Before Refactoring (Current)
- Model: 5M params (18 blocks × 192 ch)
- Training: 3-4 hours/epoch
- Epochs/day: ~6-8
- Total time (200 epochs): 14-16 days
- Evaluation: None ❌
- Win rate tracking: None ❌

### After Refactoring (Target - Small Model)
- Model: 1.2M params (10 blocks × 96 ch)
- Training: 10-15 min/epoch
- Epochs/day: 80-100
- Total time (200 epochs): 2-3 days ✅
- Evaluation: Every 5 epochs ✅
- Win rate vs baseline: Tracked ✅
- Expected strength: 70-80% of large model

### After Refactoring (Target - Medium Model)
- Model: 2.5M params (14 blocks × 128 ch)
- Training: 40-60 min/epoch
- Epochs/day: 24-36
- Total time (200 epochs): 5-8 days ✅
- Evaluation: Every 5 epochs ✅
- Expected strength: 85-90% of large model

---

## Risk Mitigation

1. **Breaking Changes**: Keep old configs as deprecated fallback
2. **Performance Regression**: Benchmark at each phase
3. **Test Failures**: Fix tests before moving to next phase
4. **Model Quality**: Run comparison tests (old vs new)

---

## Next Steps

1. Review this plan with team
2. Create feature branch: `refactor/optimization-phase-1`
3. Start with Phase 1: Evaluation framework
4. Test, test, test!
