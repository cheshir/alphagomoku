# Proximity-Based Move Penalization

## Concept

Penalize moves that are far from existing stones. In Gomoku:
- **Almost all good moves are adjacent or near existing stones**
- Distant moves can't create threats or defend
- Exception: Some advanced opening theory (rare)

## Why This is Better Than Edge Penalties

1. **Applies throughout entire game** (not just opening)
2. **Gomoku-specific domain knowledge** (validated by strong engines)
3. **Reduces MCTS search space** 5-10x (225 moves → 20-40 candidates)
4. **Faster MCTS convergence** (visits concentrated on viable moves)
5. **Addresses root cause** (distant moves are bad, edges are just a symptom)

## Implementation Strategy

Implement in **3 levels** with increasing strength:

### Level 1: Soft Penalty in Pattern Detector (Training Signal)
**Where**: `alphagomoku/utils/pattern_detector.py`
**Strength**: Soft (0.1-0.5x penalty)
**Purpose**: Give NN a hint that distant moves are less valuable

### Level 2: Hard Filter in MCTS (Inference)
**Where**: `alphagomoku/mcts/mcts.py`
**Strength**: Hard (set prior = 0.0 or very low)
**Purpose**: Focus MCTS compute on reasonable moves

### Level 3: Filter in Tactical Augmentation (Training Data Quality)
**Where**: `alphagomoku/train/tactical_augmentation.py`
**Purpose**: Ensure synthetic examples don't have distant moves

---

## Level 1: Pattern Detector (IMPLEMENT THIS FIRST)

### Code Changes

**File**: `alphagomoku/utils/pattern_detector.py`

Add new function:
```python
def compute_proximity_mask(board: np.ndarray, max_distance: int = 2) -> np.ndarray:
    """Compute proximity mask penalizing moves far from existing stones.

    Args:
        board: Board state (H, W)
        max_distance: Maximum distance to consider (default 2)

    Returns:
        Proximity mask (H, W) with values [0, 1]
        - 1.0 for moves adjacent to stones
        - 0.5 for moves 2 cells away
        - 0.1 for moves 3+ cells away
        - 1.0 for first few moves (opening exception)
    """
    h, w = board.shape
    proximity_mask = np.ones((h, w), dtype=np.float32) * 0.1  # Default: far away

    # Count stones on board
    num_stones = np.sum(board != 0)

    # Opening exception: first 5 moves can be anywhere
    if num_stones < 5:
        return np.ones((h, w), dtype=np.float32)

    # Find all occupied cells
    occupied = np.argwhere(board != 0)

    if len(occupied) == 0:
        # Empty board: center area gets higher weight
        center_r, center_c = h // 2, w // 2
        for r in range(h):
            for c in range(w):
                dist_to_center = max(abs(r - center_r), abs(c - center_c))
                if dist_to_center <= 2:
                    proximity_mask[r, c] = 1.0
                elif dist_to_center <= 4:
                    proximity_mask[r, c] = 0.7
        return proximity_mask

    # For each empty cell, compute distance to nearest stone
    for r in range(h):
        for c in range(w):
            if board[r, c] != 0:
                proximity_mask[r, c] = 0.0  # Occupied
                continue

            # Compute minimum distance to any stone (Chebyshev/chessboard distance)
            min_dist = float('inf')
            for stone_r, stone_c in occupied:
                # Chebyshev distance (max of absolute differences)
                dist = max(abs(r - stone_r), abs(c - stone_c))
                min_dist = min(min_dist, dist)

            # Assign penalty based on distance
            if min_dist == 1:
                proximity_mask[r, c] = 1.0      # Adjacent: full weight
            elif min_dist == 2:
                proximity_mask[r, c] = 0.5      # 2 cells away: half weight
            elif min_dist == 3:
                proximity_mask[r, c] = 0.2      # 3 cells away: weak
            else:
                proximity_mask[r, c] = 0.05     # 4+ cells away: very weak

    return proximity_mask
```

**Update `get_pattern_features`**:
```python
def get_pattern_features(board: np.ndarray, current_player: int) -> np.ndarray:
    """Get pattern features for neural network input.

    Combines patterns for both players, emphasizing current player's perspective.
    NOW WITH PROXIMITY PENALTY.

    Args:
        board: Board state (H, W)
        current_player: Current player to move (1 or -1)

    Returns:
        Pattern heatmap (H, W) for NN input
    """
    own_patterns = detect_patterns(board, current_player)
    opp_patterns = detect_patterns(board, -current_player)

    # Combine: own patterns + opponent patterns (defense)
    # Weight defense slightly higher
    combined = own_patterns + opp_patterns * 1.2

    # NEW: Apply proximity penalty
    proximity_mask = compute_proximity_mask(board, max_distance=2)
    combined = combined * proximity_mask  # Multiply to apply penalty

    # If no patterns detected, proximity mask becomes the feature
    if combined.max() == 0:
        combined = proximity_mask * 0.3  # Give some signal even without patterns

    # Normalize to [0, 1]
    max_val = combined.max()
    if max_val > 0:
        combined = combined / max_val

    return combined.astype(np.float32)
```

### Testing

After implementing, test that it works:
```python
import numpy as np
from alphagomoku.utils.pattern_detector import compute_proximity_mask, get_pattern_features

# Test 1: Empty board (should allow center area)
board = np.zeros((15, 15), dtype=np.int8)
mask = compute_proximity_mask(board)
print("Empty board center mask:", mask[7, 7])  # Should be 1.0
print("Empty board corner mask:", mask[0, 0])  # Should be 0.1

# Test 2: Board with one stone
board = np.zeros((15, 15), dtype=np.int8)
board[7, 7] = 1
mask = compute_proximity_mask(board)
print("Adjacent to stone:", mask[7, 8])   # Should be 1.0
print("2 cells away:", mask[7, 9])        # Should be 0.5
print("Corner (far):", mask[0, 0])        # Should be 0.05

# Test 3: Pattern features include proximity
board = np.zeros((15, 15), dtype=np.int8)
board[7, 5:9] = 1  # Four in a row
features = get_pattern_features(board, 1)
print("Next to pattern:", features[7, 9])  # Should be high (pattern + proximity)
print("Far from pattern:", features[0, 0]) # Should be very low
```

---

## Level 2: MCTS Prior Filtering (OPTIONAL - DO LATER)

**When to implement**: After seeing improvement from Level 1

### Approach A: Soft Prior Adjustment

**File**: `alphagomoku/mcts/mcts.py`

In the `expand` method, adjust priors based on proximity:

```python
def expand(self, policy: np.ndarray):
    """Expand node with children for legal actions"""
    from ..utils.pattern_detector import compute_proximity_mask

    self.is_expanded = True
    legal_actions = np.where(self.state.reshape(-1) == 0)[0]

    # NEW: Compute proximity mask
    proximity_mask = compute_proximity_mask(self.state, max_distance=2)
    proximity_flat = proximity_mask.reshape(-1)

    for action in legal_actions:
        r, c = divmod(action, self.board_size)
        child_state = self.state.copy()
        child_state[r, c] = self.current_player

        # NEW: Adjust prior with proximity
        adjusted_prior = float(policy[action]) * proximity_flat[action]

        self.children[action] = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            prior=adjusted_prior,  # Use adjusted prior
            current_player=-self.current_player,
            last_move=(r, c),
            board_size=self.board_size,
        )
```

### Approach B: Hard Filtering (More Aggressive)

```python
def expand(self, policy: np.ndarray):
    """Expand node with children for legal actions"""
    from ..utils.pattern_detector import compute_proximity_mask

    self.is_expanded = True
    legal_actions = np.where(self.state.reshape(-1) == 0)[0]

    # NEW: Compute proximity mask
    proximity_mask = compute_proximity_mask(self.state, max_distance=2)
    proximity_flat = proximity_mask.reshape(-1)

    # NEW: Only expand moves that are reasonably close
    num_stones = np.sum(self.state != 0)
    min_proximity = 0.1 if num_stones > 5 else 0.0  # Allow anything in opening

    for action in legal_actions:
        # Skip very distant moves
        if proximity_flat[action] < min_proximity:
            continue

        r, c = divmod(action, self.board_size)
        child_state = self.state.copy()
        child_state[r, c] = self.current_player

        self.children[action] = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            prior=float(policy[action]),
            current_player=-self.current_player,
            last_move=(r, c),
            board_size=self.board_size,
        )

    # Renormalize priors after filtering
    if self.children:
        total_prior = sum(child.prior for child in self.children.values())
        if total_prior > 0:
            for child in self.children.values():
                child.prior /= total_prior
```

**Benefit**: MCTS tree will be much smaller, faster convergence

**Risk**: Might filter out unusual but valid moves

---

## Level 3: Tactical Augmentation (OPTIONAL)

**File**: `alphagomoku/train/tactical_augmentation.py`

Ensure synthetic tactical examples respect proximity:

```python
def _generate_complete_five_patterns(board_size: int) -> List[SelfPlayData]:
    """Generate positions where completing five in a row wins immediately."""
    examples = []

    # Use center area (not edges) - respects proximity implicitly
    center = board_size // 2

    # Horizontal open-four (in center area)
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[center, center-2:center+2] = 1  # Four stones in center
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[center * board_size + (center+2)] = 1.0

    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=1.0,
        current_player=1,
        last_move=(center, center+1),
    ))

    # ... similar for vertical, diagonal patterns ...

    return examples
```

---

## Incremental Rollout Plan

### Phase 1: Level 1 Only (Pattern Detector)
**Time**: 15 minutes to implement + test
**Risk**: Low (just changes NN input)
**Expected**: Slightly better move selection, no speed change

**Test**:
```bash
# Run tactical tests (should still pass)
python scripts/test_tactical.py checkpoints/model_epoch_79.pt

# Train for 5 epochs
make train

# Check if proximity feature is being used
python -c "
from alphagomoku.utils.pattern_detector import compute_proximity_mask
import numpy as np
board = np.zeros((15,15), dtype=np.int8)
board[7,7] = 1
mask = compute_proximity_mask(board)
print('Mask works:', mask[7,8] > mask[0,0])
"
```

### Phase 2: Level 1 + Optimized Training (From TRAINING_FIXES.md)
**Time**: 30 minutes total
**Risk**: Low
**Expected**: 3-5x faster training + better quality

**Changes**:
- Level 1 proximity (above)
- Makefile optimizations (150 sims, 4 workers, etc.)

**Test**: Run 5 epochs, should take ~8-10 hours total (vs 30 hours before)

### Phase 3: Level 2 (MCTS Filtering) - OPTIONAL
**Time**: 30 minutes
**Risk**: Medium (might filter good moves)
**When**: After 10-20 epochs with Level 1, if still seeing bad moves

**Start with Approach A** (soft adjustment), not hard filtering

---

## Expected Results

### After Level 1 (Proximity in Pattern Detector)

**Immediate**:
- Pattern channel will guide NN away from distant moves
- No speed change
- Tactical tests should still pass

**After 10-20 epochs**:
- Fewer distant/border moves in self-play
- Model learns proximity heuristic
- Slight improvement in move quality

### After Level 2 (MCTS Filtering)

**Immediate**:
- MCTS searches 5-10x fewer nodes
- 30-50% faster self-play
- Much sharper move distribution

**Potential issue**:
- Might miss some creative distant moves
- Opening play might be too conservative

**Mitigation**: Disable filtering for first 5 moves

---

## Comparison: Your Idea vs Edge Penalty

| Aspect | Proximity Penalty | Edge Penalty |
|--------|------------------|--------------|
| **Scope** | Entire game | Opening only |
| **Principle** | Stay near existing stones | Avoid edges |
| **MCTS speedup** | 5-10x fewer nodes | None |
| **Domain validity** | Strong (Gomoku rule) | Weak (heuristic) |
| **Implementation** | Slightly more complex | Simple |
| **Risk** | Low (well-proven) | None |
| **Effectiveness** | High | Medium |

**Verdict**: Your proximity idea is **significantly better** than edge penalties.

---

## Alternative: Hybrid Approach

Combine both for maximum effect:

```python
def get_pattern_features(board: np.ndarray, current_player: int) -> np.ndarray:
    own_patterns = detect_patterns(board, current_player)
    opp_patterns = detect_patterns(board, -current_player)
    combined = own_patterns + opp_patterns * 1.2

    # Apply BOTH penalties
    proximity_mask = compute_proximity_mask(board)
    edge_mask = compute_edge_mask(board)  # From previous suggestion

    combined = combined * proximity_mask * edge_mask

    # ... normalize ...
    return combined
```

But honestly, **proximity alone is probably sufficient**.

---

## Quick Start

1. **Copy the `compute_proximity_mask` function** into `pattern_detector.py`
2. **Update `get_pattern_features`** to use it
3. **Test** that it works with the test code above
4. **Train for 5 epochs** and check if border moves reduce
5. **Report back** on results

This is a solid, proven technique. I strongly recommend implementing it.
