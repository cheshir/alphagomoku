# Opening Move Strategy

## Overview

The AI now implements intelligent opening move placement to ensure the first response is placed **near the player's opening stone**, creating immediate tactical interaction.

## Problem

Without opening logic, MCTS might place the first AI move far from the player's stone, leading to:
- Boring games with no early interaction
- Two separate groups forming independently
- Poor learning signal for the neural network
- Bad user experience (feels like AI is "running away")

## Solution

When responding to the player's first move (only 1 stone on board), the AI **boosts policy probabilities** for moves near the opponent's stone by 50x, ensuring placement within 1-2 squares.

## Implementation

### Files

- **`alphagomoku/mcts/opening.py`** - Opening move logic and helpers
- **`alphagomoku/mcts/mcts.py`** - Integrated into `_evaluate_node()`

### Key Functions

#### `boost_opening_moves(policy, state, boost_factor=50.0, distance=2)`
Modifies the policy network output to strongly favor moves near opponent's first stone.

**Parameters:**
- `policy`: Neural network policy output (flattened)
- `state`: Current board state
- `boost_factor`: Multiplication factor for nearby moves (default: 50.0)
- `distance`: How far to consider (1=adjacent, 2=two squares, 3=three squares)

**Returns:** Modified policy with nearby moves boosted

#### `get_first_move_near_opponent(state, distance=2)`
Returns list of candidate positions sorted by distance from opponent's stone.

**Returns:** List of action indices, closest first

#### `get_opening_response(state)`
Heuristic for selecting a specific opening response position.

**Strategy:**
- Center opening → respond diagonally
- Edge/corner opening → respond 1-2 squares away

### Integration Point

```python
# In MCTS._evaluate_node()
num_stones = np.sum(node.state != 0)
if num_stones == 1:
    # This is AI's first move response
    policy = boost_opening_moves(
        policy,
        node.state,
        board_size=node.board_size,
        boost_factor=50.0,  # Strong boost
        distance=2,         # Consider 2 squares away
    )
```

## Behavior Examples

### Center Opening
```
Player plays (7,7) - center
AI responds: (6,6), (6,8), (8,6), or (8,8) - diagonal
```

### Corner Opening
```
Player plays (2,2) - near corner
AI responds: Within 1-2 squares
Candidates: (1,1), (1,3), (3,1), (3,3), etc.
```

### Edge Opening
```
Player plays (7,1) - left edge
AI responds: (6,0), (6,2), (8,0), (8,2), etc.
```

## Configuration

### Boost Factor (default: 50.0)
- **10.0**: Gentle nudge, neural network still has influence
- **50.0**: Strong preference for nearby moves (recommended)
- **100.0**: Almost always plays nearby (very aggressive)

### Distance (default: 2)
- **1**: Only adjacent + diagonal (8 positions)
- **2**: Up to 2 squares away (~24 positions) - **recommended**
- **3**: Up to 3 squares away (~48 positions)

## Trade-offs

### Advantages ✅
- **Better user experience**: Immediate interaction
- **Stronger learning**: More tactical positions in training data
- **Professional appearance**: AI doesn't "run away"
- **Easier for beginners**: Clear cause-and-effect

### Considerations ⚠️
- **Overrides neural network**: In first move only
- **Hard-coded rule**: Not learned through self-play
- **May not be optimal**: In some openings, playing far might be stronger

## When Active

The boost **only applies** when:
1. There is exactly **1 stone** on the board (opponent's opening)
2. AI is making its **first move response**
3. Board is otherwise empty

After the first two moves, normal MCTS + neural network takes over completely.

## Disabling Opening Logic

If you want pure learned behavior:

```python
# Option 1: Set boost_factor to 1.0 (no boost)
# In alphagomoku/mcts/mcts.py, line 526:
policy = boost_opening_moves(
    policy,
    node.state,
    board_size=node.board_size,
    boost_factor=1.0,  # No boost - pure neural network
    distance=2,
)

# Option 2: Comment out the entire boost block
# Lines 521-532 in mcts.py
```

## Future Enhancements

1. **Opening book integration**: Use professional opening sequences
2. **Joseki patterns**: Learn common opening patterns from expert games
3. **Adaptive distance**: Adjust based on player skill level
4. **Multiple responses**: Vary opening to avoid predictability
5. **Learned opening policy**: Train separate network for opening moves

## Testing

The opening logic has been tested with:
- Center openings (7,7)
- Corner openings (2,2)
- Edge openings (7,1)
- Policy boosting (verification it works)
- Non-first-move (verification it doesn't activate)

All MCTS tests pass with opening logic enabled.

## Performance Impact

**Negligible** - the boost calculation is:
- Only triggered on move 2 (AI's first move)
- Simple numpy operations (~0.1ms)
- No impact on subsequent moves

## Related Concepts

### Gomoku Opening Theory
- **Direct opening**: Play adjacent to opponent
- **Indirect opening**: Play 1-2 squares away (creates better shape)
- **Distant opening**: Play far away (generally weaker)

Our implementation favors **indirect openings** (distance=2) which are considered strongest in professional play.

### AlphaZero Approach
AlphaZero learned openings through self-play without hard-coded rules. However, in Gomoku:
- First move is critical for game interaction
- Poor openings lead to boring training games
- Hard-coding first move improves training quality

This is a pragmatic compromise: use opening logic for first move only, then let neural network take over.

## References

- Gomoku opening theory: First moves should create potential for tactics
- Professional play: Most games start with stones within 1-2 squares
- User experience: Players expect AI to "respond" to their moves
