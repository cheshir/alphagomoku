# Training Performance & Quality Fixes

## Problem Summary
- **Training speed**: 4-5 epochs/24h (should be 15-20)
- **Model quality**: Still makes random border moves despite passing tactical tests
- **Root causes**:
  - MCTS exploration noise dominates training data
  - Parallel workers are CPU-bound (MPS unused during self-play)
  - Model too small (2.67M params)
  - Too many MCTS simulations (600) for self-play

## Solutions (Prioritized by Impact)

### 🔥 PRIORITY 1: Reduce Self-Play Compute (3-5x speedup)

#### Fix 1A: Drastically Reduce MCTS Simulations for Training
**Impact**: 3-4x faster self-play

**Why**: 600 simulations is overkill for training data generation. AlphaZero used 800 for 19×19 Go with superhuman play. For 15×15 Gomoku during training, 100-200 is sufficient.

**Change in Makefile**:
```makefile
train:
    python scripts/train.py \
        --mcts-simulations 150 \          # WAS: 600 → NOW: 150 (4x reduction)
        --batch-size-mcts 32 \            # WAS: 64 → NOW: 32 (reduce memory)
```

**Rationale**:
- Early training (epochs 0-100): Model is weak, MCTS value estimates unreliable anyway
- We're generating training data, not playing for world championship
- More games with fewer sims > fewer games with many sims
- 150 sims gives good policy targets without excessive compute

---

#### Fix 1B: Reduce Parallel Workers
**Impact**: 2x faster per-game (less CPU contention)

**Why**: 8 workers all competing for CPU with no GPU usage = severe oversubscription

**Change in Makefile**:
```makefile
train:
    python scripts/train.py \
        --parallel-workers 4 \            # WAS: 8 → NOW: 4
```

**Rationale**:
- With 8 workers on CPU, they thrash each other
- 4 workers = better CPU cache locality, less context switching
- Parallel workers can't use MPS due to multiprocessing constraints

---

#### Fix 1C: Increase Games Per Epoch (Better Data Efficiency)
**Impact**: Better model quality without more time

**Change in Makefile**:
```makefile
train:
    python scripts/train.py \
        --selfplay-games 200 \            # WAS: 128 → NOW: 200
```

**Rationale**:
- With 150 sims (vs 600), each game is 4x faster
- Can generate 200 games in same time as 50 games before
- More diverse training data per epoch

---

### 🎯 PRIORITY 2: Improve Training Data Quality (Fix Border Moves)

#### Fix 2A: Reduce MCTS Temperature Noise
**Impact**: Cleaner training data, fewer random moves

**Current issue**: `temperature=1.0` for first 8 moves means random sampling
- This creates noisy training targets
- Model learns to mimic exploration, not optimal play

**Change in `alphagomoku/selfplay/selfplay.py:84`**:
```python
# OLD:
temperature = 1.0 if move_count < temperature_moves else 0.0

# NEW (Phase-based temperature):
if move_count < 3:
    temperature = 0.8  # Some exploration in opening
elif move_count < 6:
    temperature = 0.5  # Less exploration
else:
    temperature = 0.0  # Deterministic (argmax)
```

**Alternative** (add as argument in train.py):
```python
# For epochs > 50, use lower temperature
if epoch < 30:
    temperature_schedule = [1.0] * 8  # Early: explore
elif epoch < 60:
    temperature_schedule = [0.8] * 5  # Mid: less exploration
else:
    temperature_schedule = [0.5] * 3  # Late: minimal exploration
```

---

#### Fix 2B: Increase Tactical Augmentation
**Impact**: More forced-move examples, better tactical play

**Change in `scripts/train.py:321`**:
```python
# OLD:
selfplay_data = augment_with_tactical_data(selfplay_data, board_size=15, augmentation_ratio=0.1)

# NEW:
augmentation_ratio = 0.3 if epoch < 100 else 0.2  # Higher early on
selfplay_data = augment_with_tactical_data(selfplay_data, board_size=15, augmentation_ratio=augmentation_ratio)
```

---

#### Fix 2C: Add Edge Penalty in Pattern Detector
**Impact**: Explicit signal that edges are bad in opening

**Add to `alphagomoku/utils/pattern_detector.py:43`**:
```python
def detect_patterns(board: np.ndarray, player: int) -> np.ndarray:
    h, w = board.shape
    pattern_map = np.zeros((h, w), dtype=np.float32)

    # NEW: Penalize edges in early game
    stones_on_board = np.sum(board != 0)
    if stones_on_board < 10:  # Early game
        # Create edge penalty mask
        edge_penalty = np.ones((h, w), dtype=np.float32)
        edge_penalty[0, :] = 0.3   # Top edge
        edge_penalty[-1, :] = 0.3  # Bottom edge
        edge_penalty[:, 0] = 0.3   # Left edge
        edge_penalty[:, -1] = 0.3  # Right edge
        # Corners even worse
        edge_penalty[0, 0] = 0.1
        edge_penalty[0, -1] = 0.1
        edge_penalty[-1, 0] = 0.1
        edge_penalty[-1, -1] = 0.1
    else:
        edge_penalty = np.ones((h, w), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            if board[r, c] != 0:
                continue

            score = 0.0
            # ... existing pattern detection code ...

            # Apply edge penalty
            pattern_map[r, c] = min(1.0, score * edge_penalty[r, c])

    return pattern_map
```

---

### 🧠 PRIORITY 3: Increase Model Capacity (Better Value Estimates)

#### Fix 3A: Larger Network
**Impact**: Better policy/value estimates, fewer MCTS sims needed

**Why**: 2.67M parameters is tiny. AlphaZero Go used ~20M for 19×19.

**Change in `scripts/train.py:222`**:
```python
# OLD:
model = GomokuNet(board_size=15, num_blocks=12, channels=64)  # 2.67M params

# NEW:
model = GomokuNet(board_size=15, num_blocks=15, channels=96)  # ~8M params
```

**Trade-offs**:
- ✅ Better policy accuracy → less MCTS needed → faster inference
- ✅ Better value estimates → faster MCTS convergence
- ✅ More capacity to learn patterns
- ❌ Training will be ~30% slower (but self-play 3x faster = net win)
- ❌ Need to train from scratch (can't resume from epoch 80)

---

### ⚡ PRIORITY 4: Training Loop Optimizations

#### Fix 4A: Reduce Training Steps Per Epoch
**Impact**: 30-40% faster training phase

**Current**: 1000 steps/epoch with batch_size=512 = 512K positions trained per epoch
**Actual buffer size**: 500K positions

**Change in `alphagomoku/train/trainer.py:186`**:
```python
def train_epoch(
    self,
    data_buffer: DataBuffer,
    batch_size: int = 512,
    steps_per_epoch: int = 1000,  # ← This is the problem
) -> Dict[str, float]:
```

**In `scripts/train.py:339`**:
```python
# NEW: Calculate steps based on buffer size
steps = max(200, min(500, len(data_buffer) // args.batch_size))
metrics = trainer.train_epoch(data_buffer, args.batch_size, steps_per_epoch=steps)
```

**Why**: You're training on 512K positions when buffer only has 500K. Lots of duplicate sampling!

---

#### Fix 4B: Increase Batch Size
**Impact**: Better GPU utilization, 20-30% faster training

**Change in Makefile**:
```makefile
train:
    python scripts/train.py \
        --batch-size 1024 \               # WAS: 512 → NOW: 1024
```

**Check if you have RAM**: 1024 batch × 5 channels × 15×15 × 4 bytes = ~5MB batch. Should be fine.

---

## Summary: Quick Wins Configuration

### Immediate Changes (No Code Edits)

Edit your `Makefile`:
```makefile
train:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
 		--epochs 200 \
 		--selfplay-games 200 \              # 128 → 200 (more data)
 		--mcts-simulations 150 \            # 600 → 150 (4x faster!)
 		--batch-size 1024 \                 # 512 → 1024 (better GPU util)
 		--lr 1e-3 \
 		--min-lr 5e-4 \
 		--warmup-epochs 0 \
 		--lr-schedule cosine \
 		--map-size-gb 12 \
 		--buffer-max-size 500000 \
 		--batch-size-mcts 32 \              # 64 → 32 (less memory per sim)
 		--parallel-workers 4 \              # 8 → 4 (less CPU thrashing)
 		--difficulty medium \
 		--resume auto
```

**Expected improvement**: **3-5x faster** (1.5-2 hours/epoch instead of 6 hours)

---

### Code Changes (30 minutes work)

1. **Temperature schedule** (selfplay.py) - 10 min
2. **Edge penalty** (pattern_detector.py) - 15 min
3. **Training steps** (train.py) - 5 min

**Expected improvement**: Better data quality, fewer border moves

---

### Nuclear Option: Start Fresh with Bigger Model

If you're willing to retrain:
```python
# scripts/train.py:222
model = GomokuNet(board_size=15, num_blocks=15, channels=96)  # ~8M params
```

**Why**:
- Current model at 80 epochs has bad habits from 80 epochs of noisy data
- Bigger model with fixes will learn faster and better
- Will take ~100 epochs to surpass current model, but will be much stronger

---

## Testing Your Fixes

### 1. Speed Test (should be 1.5-2 hours/epoch)
```bash
# Run 1 epoch with new settings
make train
```

### 2. Tactical Test (should stay at 5/5)
```bash
python scripts/test_tactical.py checkpoints/model_epoch_XX.pt
```

### 3. Border Move Test (should improve)
Play a few games and check if early moves avoid edges:
```bash
docker-compose up --build
# Play on localhost:5173
```

---

## Expected Timeline

### With Makefile Changes Only
- **Training speed**: 1.5-2 hours/epoch (was 6 hours)
- **Epochs per day**: 12-16 (was 4)
- **Weeks to strong model**: 2 weeks (200 epochs)

### With Code Changes
- **Training speed**: Same as above
- **Quality**: Better tactical play, fewer border moves
- **Weeks to strong model**: 2 weeks (150-200 epochs)

### With Fresh Start + Bigger Model
- **Training speed**: 2-2.5 hours/epoch
- **Quality**: Significantly better
- **Weeks to strong model**: 3-4 weeks (150-200 epochs)

---

## Why You Had These Issues

1. **MCTS Overuse**: Following AlphaGo/AlphaZero papers literally
   - They had months of compute and huge clusters
   - 600 sims is for *inference*, not *training data generation*

2. **Parallel Worker Design**: CPU-only due to MPS multiprocessing limits
   - This is a known PyTorch limitation
   - Should use fewer workers or sequential processing

3. **Small Model**: Conservative choice for M1 Pro
   - Your hardware can handle 8-10M params easily
   - Training is bottlenecked by self-play, not NN forward pass

4. **Temperature = 1.0**: Standard in papers, but creates noisy data
   - Papers assume unlimited compute to overcome noise
   - With limited compute, cleaner data > more data

---

## Questions?

Run with new settings and let me know:
1. Time per epoch
2. Tactical test results after 5-10 new epochs
3. Any errors or issues
