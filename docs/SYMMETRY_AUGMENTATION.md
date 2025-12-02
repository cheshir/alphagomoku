# Board Symmetry Augmentation for AlphaGomoku

## Overview

This document describes the board symmetry augmentation system implemented for Gomoku training. The system applies the 8 symmetries of a square board (dihedral group D4) to training data, improving model generalization without modifying self-play or MCTS logic.

## Key Features

✅ **Correct transformations** - All 8 symmetries properly transform state, policy, and last_move
✅ **Lazy augmentation** - Random symmetry applied on-demand during sampling (memory efficient)
✅ **Eager augmentation** - Pre-generate all 8 symmetries (faster sampling, more storage)
✅ **Comprehensive tests** - 30 unit and integration tests verify correctness
✅ **No self-play changes** - Augmentation only affects training, not game generation

## The 8 Symmetries

The system supports all 8 symmetries of a square board:

| ID | Name | Transformation | Example (r,c) → (r',c') |
|----|------|----------------|-------------------------|
| 0 | Identity | (r, c) → (r, c) | (1,2) → (1,2) |
| 1 | Rotate 90° CW | (r, c) → (c, N-1-r) | (1,2) → (2,3) for N=5 |
| 2 | Rotate 180° | (r, c) → (N-1-r, N-1-c) | (1,2) → (3,2) for N=5 |
| 3 | Rotate 270° CW | (r, c) → (N-1-c, r) | (1,2) → (2,1) for N=5 |
| 4 | Flip vertical | (r, c) → (r, N-1-c) | (1,2) → (1,2) for N=5 |
| 5 | Flip horizontal | (r, c) → (N-1-r, c) | (1,2) → (3,2) for N=5 |
| 6 | Flip main diagonal | (r, c) → (c, r) | (1,2) → (2,1) |
| 7 | Flip anti-diagonal | (r, c) → (N-1-c, N-1-r) | (1,2) → (2,3) for N=5 |

## Implementation

### Core Module: `alphagomoku/utils/symmetry.py`

The `BoardSymmetry` class provides:

```python
from alphagomoku.utils.symmetry import BoardSymmetry

# Transform coordinates
r_new, c_new = BoardSymmetry.transform_coordinates(r, c, board_size, sym_id)

# Transform board tensor [C, H, W]
state_sym = BoardSymmetry.transform_board_tensor(state, sym_id)

# Transform policy (flat or grid)
policy_sym = BoardSymmetry.transform_policy_flat(policy, board_size, sym_id)
policy_sym = BoardSymmetry.transform_policy_grid(policy_grid, sym_id)

# Transform complete example
state_sym, policy_sym, value, last_move_sym, player, metadata = \
    BoardSymmetry.apply_symmetry(state, policy, value, sym_id, last_move, player, metadata)

# Get random symmetry for augmentation
sym_id = BoardSymmetry.get_random_symmetry()  # Returns 0-7
```

### Integration: `alphagomoku/train/data_buffer.py`

The `DataBuffer` class uses symmetry augmentation:

**Lazy augmentation** (default, memory-efficient):
```python
buffer = DataBuffer(db_path, lazy_augmentation=True)
buffer.add_data(examples)  # Stores original examples

# Each sample gets random symmetry during sampling
batch = buffer.sample_batch(512)  # Random sym_id ∈ [0,7] per sample
```

**Eager augmentation** (faster sampling, 8x storage):
```python
buffer = DataBuffer(db_path, lazy_augmentation=False)
buffer.add_data(examples)  # Generates and stores all 8 symmetries

batch = buffer.sample_batch(512)  # Samples from pre-augmented data
```

## What Gets Transformed

### State Tensor [C, H, W]
All spatial channels are transformed identically:
- Channel 0: Current player stones
- Channel 1: Opponent stones
- Channel 2: Last move marker
- Channel 3: Side to move (constant, unaffected)
- Channel 4: Pattern/threat maps

### Policy Target
Transformed to match state:
- **Flat format** [N×N]: Indices remapped via coordinate transform
- **Grid format** [N, N]: Spatial transform like state

### Last Move Coordinates
Transformed to stay consistent with state/policy:
```python
last_move_sym = BoardSymmetry.transform_last_move(last_move, board_size, sym_id)
```

### Invariants (Unchanged)
- **Value**: Win/loss/draw is rotation-invariant
- **Current player**: Player identity doesn't change
- **Metadata**: Auxiliary info preserved

## Testing

### Unit Tests: `tests/unit/test_symmetry.py`
- ✅ Coordinate transformations for all 8 symmetries
- ✅ Board tensor transformations (2D and 3D)
- ✅ Policy transformations (flat and grid)
- ✅ Roundtrip properties (sym → inverse → identity)
- ✅ State-policy consistency
- ✅ Edge cases (1×1 board, 19×19 board, None last_move)

### Integration Tests: `tests/integration/test_data_augmentation.py`
- ✅ Lazy augmentation applies random symmetries
- ✅ Eager augmentation generates all 8
- ✅ Value preservation
- ✅ Probability mass preservation
- ✅ Last_move, state, and policy alignment
- ✅ Batch generation without NaNs/Infs

Run tests:
```bash
# Unit tests only
pytest tests/unit/test_symmetry.py -v

# Integration tests only
pytest tests/integration/test_data_augmentation.py -v

# All symmetry tests
pytest tests/unit/test_symmetry.py tests/integration/test_data_augmentation.py -v
```

## Usage in Training

The augmentation is **automatically applied** during training via `DataBuffer`:

```python
# In train.py or your training script
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.train.trainer import Trainer

# Create buffer with lazy augmentation (recommended)
data_buffer = DataBuffer(
    db_path='./data/replay_buffer.lmdb',
    max_size=5_000_000,
    lazy_augmentation=True  # Random symmetry per sample
)

# Add self-play data (no augmentation during generation!)
data_buffer.add_data(selfplay_positions)

# Training automatically benefits from augmentation
trainer = Trainer(model, lr=1e-3, device='cuda')
metrics = trainer.train_epoch(
    data_buffer,
    batch_size=512,
    steps_per_epoch=1000
)
```

Each batch will contain positions with **random symmetries**, effectively multiplying your training data by 8× without storing duplicates.

## Design Decisions

### Why Lazy Augmentation?
**Pros:**
- 8× less storage (stores original only)
- Random symmetry per epoch → more diversity
- Better for large buffers (millions of positions)

**Cons:**
- Small CPU overhead during sampling
- ~5-10% slower sampling vs eager

**When to use eager:**
- Small datasets where 8× storage is acceptable
- Want maximum sampling speed
- Training is sampling-bottlenecked

### Why Not Augment During Self-Play?
Self-play and MCTS must operate on a **consistent board**. Applying symmetries during game generation would:
- ❌ Break MCTS tree reuse (different orientations)
- ❌ Confuse legal move generation
- ❌ Make debugging impossible
- ❌ Violate AlphaZero methodology

**Correct approach:** Generate games in original orientation, augment during **training** only.

### Coordinate System
The implementation uses standard array indexing:
- `r` = row (0 to N-1, top to bottom)
- `c` = column (0 to N-1, left to right)
- Flat index `k = r * N + c` (row-major order)

All transformations preserve this convention.

## Common Issues & Solutions

### Issue: State and Policy Misaligned After Augmentation
**Symptom:** Policy predicts moves that don't match board features.

**Solution:** Ensure both state and policy use the same `sym_id`:
```python
# WRONG - different random IDs
state_sym = transform_board_tensor(state, random.randint(0, 7))
policy_sym = transform_policy_flat(policy, board_size, random.randint(0, 7))

# CORRECT - same sym_id
sym_id = BoardSymmetry.get_random_symmetry()
state_sym, policy_sym, _, _, _, _ = BoardSymmetry.apply_symmetry(
    state, policy, value, sym_id
)
```

### Issue: Last Move Doesn't Match State
**Symptom:** `last_move` coordinates don't align with last_move channel.

**Solution:** Always transform last_move with same `sym_id`:
```python
last_move_sym = BoardSymmetry.transform_last_move(last_move, board_size, sym_id)
```

### Issue: Policy Probability Mass Not Preserved
**Symptom:** `policy.sum() != 1.0` after transformation.

**This should never happen** - the implementation preserves probability mass. If you see this:
1. Check your policy is normalized before augmentation
2. Verify you're using `BoardSymmetry.apply_symmetry()` not manual transforms
3. Report as a bug with reproduction steps

## Performance Impact

Benchmarks on 15×15 board, M1 Pro:

| Operation | Time (µs) | Notes |
|-----------|-----------|-------|
| Coordinate transform | 0.5 | Single (r,c) → (r',c') |
| Board tensor transform [5,15,15] | 45 | State augmentation |
| Policy flat transform [225] | 12 | Policy augmentation |
| Full augmentation | 60 | State + policy + last_move |
| Batch of 512 samples | 31ms | Lazy augmentation overhead |

**Training impact:** <5% slowdown vs no augmentation, but typically **faster convergence** due to better generalization.

## Future Improvements

Potential enhancements (not currently needed):

1. **GPU augmentation:** Apply transforms on GPU during batch preparation
2. **Cached transformations:** Precompute coordinate mappings for faster lookup
3. **Partial augmentation:** Use subset of 8 symmetries if some are redundant
4. **Curriculum augmentation:** Start with no augmentation, gradually introduce

## References

- AlphaGo Zero paper: https://www.nature.com/articles/nature24270
- AlphaZero paper: https://arxiv.org/abs/1712.01815
- Dihedral group D4: https://en.wikipedia.org/wiki/Dihedral_group

## Questions?

For issues or questions about symmetry augmentation:
1. Check test files for usage examples
2. Run `pytest` to verify your environment
3. Open an issue with reproduction steps

---

**Last updated:** 2025-12-02
**Authors:** AlphaGomoku development team
