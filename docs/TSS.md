# Threat-Space Search (TSS) Comprehensive Guide

## Overview

Threat-Space Search (TSS) is a specialized tactical search module designed to enhance the Gomoku AI's tactical awareness by explicitly detecting and exploiting threat sequences such as open-threes, open-fours, double threats, and forced win/defense lines. TSS operates alongside Monte Carlo Tree Search (MCTS) and the endgame solver to improve move selection in complex tactical scenarios.

### What is TSS?

TSS performs depth-limited search exploring sequences of moves that create, block, or extend threats. It can detect:
- **Open-four**: Four stones with at least one open end (immediate win threat)
- **Broken-four**: Four stones with a gap that can be completed
- **Open-three**: Three stones with open ends (creates double threat potential)
- **Double-three/Double-four**: Multiple simultaneous threats
- **Forced win sequences**: Guaranteed victory within search depth
- **Forced defense moves**: Required responses to avoid immediate loss

### Goals

- Accurately detect critical tactical threats in the current board position
- Identify forced win or forced defense sequences within configurable search depth
- Override or guide MCTS move selection when forced tactical responses are found
- Provide configurable depth and time limits to balance strength and inference latency
- Integrate seamlessly with existing MCTS and endgame solver pipeline
- Support difficulty-level adjustments by tuning depth, time caps, and activation thresholds
- Enable progressive learning where tactical rules are gradually disabled during training

## When to Use TSS

### Training vs Inference

**During Training:**
TSS helps bootstrap the model by providing tactical guidance in early epochs, then gradually reduces assistance as the model learns:
- **Epochs 0-49**: Full TSS assistance (open-four, broken-four defenses enabled)
- **Epochs 50-99**: Reduced TSS (only open-four defenses enabled)
- **Epochs 100+**: Minimal TSS (only game-rule enforcement)

**During Inference:**
TSS strength varies by difficulty level:
- **Easy**: Minimal TSS, mostly learned behavior
- **Medium**: Balanced TSS for tactical support
- **Hard**: Full TSS for maximum strength

### When to Enable TSS

1. **Always check** for immediate threats before MCTS
2. **Use in midgame** when tactical patterns emerge (15-30 moves)
3. **Disable in endgame** when endgame solver takes over (< 10 empty cells)
4. **Skip in opening** unless specific tactical training needed (first 5-10 moves)

### Integration with Other Components

**With MCTS:**
- TSS is invoked before MCTS rollouts during the search phase
- If a forced win or forced defense is detected, TSS overrides MCTS
- Returns the forced move directly without MCTS exploration

**With Endgame Solver:**
- TSS operates primarily in the midgame tactical phase
- When empty cells fall below threshold (~10 cells), endgame solver takes precedence
- TSS focuses on tactical patterns, solver focuses on perfect endgame play

## Configuration

### TSSConfig Options

The `TSSConfig` class controls which tactical rules are enabled:

```python
from alphagomoku.tss import TSSConfig, set_default_config

config = TSSConfig(
    # Game rules (always enabled)
    defend_immediate_five=True,      # Block 5-in-a-row (game rule)

    # Tactical patterns (configurable)
    defend_open_four=True,           # .XXXX. patterns
    defend_broken_four=True,         # XXXX. or X.XXX patterns
    defend_open_three=False,         # .XXX. patterns (usually off)

    # Win search settings
    search_forced_wins=True,         # Look for winning sequences
    max_search_depth=6,              # Maximum search depth in plies
)

# Set as global default
set_default_config(config)
```

### Configuration for Training

Progressive learning schedule that gradually disables TSS assistance:

```python
from alphagomoku.tss import TSSConfig

# Automatic configuration based on epoch
current_epoch = 75
config = TSSConfig.for_training_epoch(current_epoch)
```

**Training Schedule:**

| Epoch Range | Immediate-5 | Open-4 | Broken-4 | Open-3 | Purpose |
|-------------|-------------|--------|----------|--------|---------|
| 0-49        | Always      | Yes    | Yes      | No     | Bootstrap learning |
| 50-99       | Always      | Yes    | No       | No     | Transition phase |
| 100+        | Always      | No     | No       | No     | Natural learning |

**Why This Schedule?**

- **Epoch 0-49 (Bootstrap)**: Model is weak, needs guidance to learn correct tactical responses
- **Epoch 50-99 (Transition)**: Model has seen many examples, can handle semi-open patterns
- **Epoch 100+ (Mastery)**: Model is experienced, handles tactics through learned policy + MCTS

### Configuration for Inference

Difficulty-based configurations for competitive play:

```python
from alphagomoku.tss import TSSConfig

# Easy mode - minimal TSS
easy_config = TSSConfig.for_inference("easy")

# Medium mode - balanced TSS
medium_config = TSSConfig.for_inference("medium")

# Hard mode - full TSS
hard_config = TSSConfig.for_inference("hard")
```

**Difficulty Settings:**

| Difficulty | Immediate-5 | Open-4 | Broken-4 | Depth | Time Cap | Strategy |
|------------|-------------|--------|----------|-------|----------|----------|
| Easy       | Yes         | No     | No       | 2-3   | 30ms     | Natural play |
| Medium     | Yes         | Yes    | No       | 4-5   | 100ms    | Balanced |
| Hard       | Yes         | Yes    | Yes      | 6-7   | 300ms    | Maximum strength |

### Search Parameters

- **Search Depth**: Maximum ply depth to explore (typically 2-7)
  - Depth 2-3: Fast checks (~10-50ms)
  - Depth 4-5: Balanced (~50-200ms)
  - Depth 6+: Deep analysis (~200ms+)

- **Time Cap**: Maximum allowed time per TSS invocation (ms)
  - Easy: 30-50ms
  - Medium: 100-200ms
  - Strong: 300-500ms

- **Activation Threshold**: Minimum game phase to activate TSS
  - Early game (moves < 10): Usually disabled
  - Midgame (moves 10-30): Full activation
  - Endgame (empty < 10): Transition to solver

## Usage Examples

### Basic Usage

```python
from alphagomoku.tss import Position, tss_search
import numpy as np

# Create a position
board = np.zeros((15, 15), dtype=np.int8)
board[7, 7] = 1  # Place a stone
position = Position(board=board, current_player=-1)

# Run TSS analysis
result = tss_search(position, depth=4, time_cap_ms=100)

if result.is_forced_defense:
    print(f"Must defend at {result.forced_move}")
elif result.is_forced_win:
    print(f"Winning move at {result.forced_move}")
else:
    print("No forced sequence found")
```

### Position Class

```python
from alphagomoku.tss import Position

position = Position(
    board=board_array,      # 15x15 numpy array (-1, 0, 1)
    current_player=1,       # 1 or -1
    last_move=(7, 7),      # Optional: (row, col) of last move
    board_size=15          # Board size (default: 15)
)

# Utility methods
legal_moves = position.get_legal_moves()
new_position = position.make_move(7, 8)
is_terminal, winner = position.is_terminal()
```

### Threat Detection

```python
from alphagomoku.tss import ThreatDetector, ThreatType

detector = ThreatDetector()

# Detect all threats for a player
threats = detector.detect_threats(position, player=1)
for row, col, threat_type in threats:
    print(f"Threat at ({row}, {col}): {threat_type.value}")

# Get moves that create or block threats
threat_moves = detector.get_threat_moves(position, player=1)

# Check for forced defense moves
defense_moves = detector.must_defend(position, player=1)
```

### Integration with MCTS

```python
from alphagomoku.tss import tss_search, Position, TSSConfig, set_default_config
import numpy as np

class TSSGuidedMCTS:
    def __init__(self, model, env, difficulty="medium"):
        self.mcts = MCTS(model, env)
        self.env = env

        # Configure TSS for difficulty
        config = TSSConfig.for_inference(difficulty)
        set_default_config(config)

        # TSS parameters
        self.tss_depth = 4 if difficulty == "medium" else (2 if difficulty == "easy" else 6)
        self.tss_time_cap = 100 if difficulty == "medium" else (30 if difficulty == "easy" else 300)

    def search(self, state, temperature=1.0):
        # Convert to TSS position
        position = Position(
            board=state,
            current_player=self.env.current_player,
            last_move=tuple(self.env.last_move) if self.env.last_move[0] >= 0 else None
        )

        # Check for forced moves
        try:
            tss_result = tss_search(position, self.tss_depth, self.tss_time_cap)

            if tss_result.forced_move:
                # Override MCTS with forced move
                action_probs = np.zeros(15 * 15)
                r, c = tss_result.forced_move
                action_probs[r * 15 + c] = 1.0

                reason = tss_result.search_stats.get('reason', 'tss_forced')
                print(f"TSS override: {reason} at ({r}, {c})")

                return action_probs, np.array([1.0])
        except Exception as e:
            print(f"TSS failed: {e}, falling back to MCTS")

        # Use MCTS for normal positions
        return self.mcts.search(state, temperature)
```

### Training Integration

```python
from alphagomoku.tss import TSSConfig, set_default_config

def train_with_progressive_tss(model, num_epochs=200):
    for epoch in range(num_epochs):
        # Update TSS configuration
        tss_config = TSSConfig.for_training_epoch(epoch)
        set_default_config(tss_config)

        # Log configuration changes at key epochs
        if epoch in [0, 50, 100] or epoch % 25 == 0:
            print(f"\nEpoch {epoch} TSS Configuration:")
            print(f"  - defend_open_four: {tss_config.defend_open_four}")
            print(f"  - defend_broken_four: {tss_config.defend_broken_four}")

        # Generate self-play games with current TSS config
        games = parallel_selfplay(model, num_games=100)

        # Train on games
        train_epoch(model, games)

        # Evaluate
        if epoch % 10 == 0:
            evaluate_model(model, epoch)
```

### Custom Configuration

```python
from alphagomoku.tss import TSSConfig, set_default_config

# Custom configuration for specific testing
custom_config = TSSConfig(
    defend_immediate_five=True,   # Always enforce game rules
    defend_open_four=False,       # Test if model learned this
    defend_broken_four=False,     # Test if model learned this
    defend_open_three=False,      # Usually off anyway
    search_forced_wins=True,      # Keep win search
    max_search_depth=4,           # Moderate depth
)

set_default_config(custom_config)

# Now test model performance without TSS assistance
result = tss_search(position, depth=4, time_cap_ms=100)
```

## Performance Impact

### Typical Performance

**Simple positions** (no threats):
- Time: 1-5ms
- Nodes visited: 1-10
- Impact: Minimal overhead

**Tactical positions** (some threats):
- Time: 10-100ms
- Nodes visited: 10-100
- Impact: 5-10% of move time

**Complex positions** (multiple threats):
- Time: 100-500ms
- Nodes visited: 100-1000
- Impact: 20-30% of move time

### Memory Usage

- **Position storage**: ~1KB per position
- **Search tree**: ~10-100KB for typical searches
- **Threat detection**: Minimal overhead (~1-2KB)
- **Total overhead**: < 1MB for most searches

### Speed Tradeoffs

**Benefits:**
- Eliminates tactical blunders
- Finds forced wins quickly
- Reduces MCTS search space in tactical positions

**Costs:**
- Additional latency before MCTS
- Memory for search tree
- CPU for threat detection

**Optimization Tips:**
1. **Time budgeting**: Reserve 10-20% of move time for TSS
2. **Depth scaling**: Increase depth with remaining game time
3. **Early exit**: Return immediately on forced win/defense
4. **Caching**: Cache threat detection results when possible

### Algorithm Details

**Threat Detection:**
1. Analyze board to identify all threat patterns
2. Use pattern maps and heuristics to prioritize candidate moves
3. Focus on relevant moves (open-four, broken-four, etc.)

**Threat-Space Exploration:**
1. Perform depth-limited search exploring threat sequences
2. Alternate between players, simulating forced responses
3. Detect forced win lines (current player guarantees victory)
4. Detect forced defense lines (must respond to avoid loss)

**Pruning and Heuristics:**
1. Use threat heuristics to prune search space
2. Limit branching to moves that extend or block threats
3. Employ iterative deepening to respect time caps
4. Prioritize checking for wins before defenses

**Critical Logic - Prioritize Winning:**
```python
# 1. Check for immediate WIN first
immediate_win = check_immediate_win(position)
if immediate_win:
    return TSSResult(forced_move=immediate_win, is_forced_win=True)

# 2. Check if opponent has immediate win threat
defense_moves = must_defend(position)

# 3. If both have wins, choose our win (we move first!)
if defense_moves and immediate_win:
    return TSSResult(forced_move=immediate_win, is_forced_win=True)

# 4. Only defend if we don't have immediate win
if defense_moves:
    return TSSResult(forced_move=defense_moves[0], is_forced_defense=True)
```

## Best Practices

### When to Use TSS

1. **Always check** for immediate threats before MCTS (< 5ms overhead)
2. **Use in midgame** when tactical patterns emerge (moves 10-30)
3. **Disable in endgame** when endgame solver takes over (< 10 empty)
4. **Skip in opening** unless specific tactical training (moves < 5)

### Integration Tips

1. **Time budgeting**: Reserve 10-20% of move time for TSS
2. **Depth scaling**: Increase depth with remaining game time
3. **Logging**: Track TSS override frequency for analysis
4. **Fallback**: Always have MCTS as backup if TSS fails
5. **Progressive learning**: Disable gradually during training

### Error Handling

```python
try:
    result = tss_search(position, depth=4, time_cap_ms=100)
    if result.forced_move:
        return result.forced_move
except Exception as e:
    print(f"TSS failed: {e}")
    # Fallback to MCTS
    return mcts.search(state)
```

### Monitoring Learning Progress

Check if model learned tactics by testing with TSS disabled:

```python
# Test at epoch 60 (broken-four should be learned)
config = TSSConfig(
    defend_immediate_five=True,
    defend_open_four=True,
    defend_broken_four=False,  # Disabled - test learning
    defend_open_three=False,
)
set_default_config(config)

# Create test position with broken-four threat
position = create_broken_four_position()

# Check MCTS response
action_probs, value = mcts.search(position)
best_move = action_probs.argmax()

if best_move in expected_defense_moves:
    print("Model learned broken-four defense!")
else:
    print("Model hasn't learned yet, extend training")
```

## API Reference

### Entry Point

```python
def tss_search(position: Position, depth: int, time_cap_ms: int) -> TSSResult:
    """
    Perform Threat-Space Search on the given position.

    Args:
        position: Current board state and player to move
        depth: Maximum search depth in plies
        time_cap_ms: Time cap in milliseconds

    Returns:
        TSSResult containing:
            - forced_move: (row, col) or None
            - is_forced_win: True if forced win detected
            - is_forced_defense: True if forced defense detected
            - search_stats: Dict with nodes_visited, time_ms, reason
    """
```

### TSSResult Structure

```python
@dataclass
class TSSResult:
    forced_move: Optional[Tuple[int, int]]  # (row, col) or None
    is_forced_win: bool                     # Winning sequence found
    is_forced_defense: bool                 # Defense required
    search_stats: Dict[str, Any]            # Statistics

    # search_stats contains:
    # - 'nodes_visited': Number of positions explored
    # - 'time_ms': Time spent in milliseconds
    # - 'reason': Why move was chosen (e.g., 'immediate_defense', 'forced_win')
```

### Threat Types

```python
class ThreatType(Enum):
    OPEN_FOUR = "open_four"          # .XXXX. - immediate win threat
    BROKEN_FOUR = "broken_four"      # XXXX. or X.XXX - one move to five
    OPEN_THREE = "open_three"        # .XXX. - creates double threat
    DOUBLE_THREE = "double_three"    # Two three-in-a-rows
    DOUBLE_FOUR = "double_four"      # Two four-in-a-rows
```

## Logging and Monitoring

### Detailed Logging

```python
result = tss_search(position, depth=4, time_cap_ms=100)
stats = result.search_stats

print(f"Nodes visited: {stats['nodes_visited']}")
print(f"Time used: {stats['time_ms']:.1f}ms")
print(f"Reason: {stats['reason']}")

# Reasons include:
# - 'immediate_defense': Found forced defense
# - 'forced_win': Found winning sequence
# - 'immediate_win': Immediate 5-in-a-row
# - 'no_forced_sequence': No tactical override needed
```

### Monitoring Metrics

Track these metrics during training and inference:
- Average latency per TSS call
- Frequency of forced win/defense detections
- Memory usage during TSS
- TSS override rate (vs MCTS)
- Success rate of TSS moves

### Training Logs

The training script automatically logs TSS configuration changes:

```
Epoch 0 TSS Configuration:
   - defend_immediate_five: True
   - defend_open_four: True
   - defend_broken_four: True
   - defend_open_three: False

Epoch 50 TSS Configuration:
   - defend_immediate_five: True
   - defend_open_four: True
   - defend_broken_four: False  <- Changed!
   - defend_open_three: False

Epoch 100 TSS Configuration:
   - defend_immediate_five: True
   - defend_open_four: False  <- Changed!
   - defend_broken_four: False
   - defend_open_three: False
```

## Testing and Validation

### Unit Tests

```bash
# Run TSS unit tests
python -m pytest tests/unit/test_tss.py -v

# Test specific functionality
python -m pytest tests/unit/test_tss.py::test_immediate_win -v
```

### Integration Tests

```bash
# Run TSS integration tests
python -m pytest tests/integration/test_tss_integration.py -v
```

### Manual Testing Scripts

```bash
# Test TSS standalone
python scripts/test_tss.py

# Test TSS with MCTS
python scripts/tss_mcts_example.py
```

### Critical Test Cases

All tests validate correct tactical behavior:

1. **Both have four - choose win**: Bot completes own five instead of blocking
2. **Bot has four, player has three - choose win**: Prioritize win over defense
3. **Player has three, bot no win - defend**: Must block open-three
4. **Open-2 not urgent**: TSS doesn't force defense for weak threats
5. **Open-4 IS urgent**: TSS forces either win or defense

## Troubleshooting

### TSS Too Slow

**Problem**: TSS taking too long per move

**Solutions**:
```python
# Reduce depth
result = tss_search(position, depth=2, time_cap_ms=30)

# Reduce time cap
result = tss_search(position, depth=4, time_cap_ms=50)

# Disable in opening
if move_count < 10:
    skip_tss = True
```

### False Positives

**Problem**: TSS finding "threats" that aren't real

**Solutions**:
```python
# Check threat detection manually
detector = ThreatDetector()
threats = detector.detect_threats(position, player)
for r, c, threat_type in threats:
    print(f"Threat: {threat_type} at ({r}, {c})")
    # Verify on board
```

### Integration Errors

**Problem**: TSS not working with MCTS

**Solutions**:
```python
# Ensure position conversion is correct
position = Position(
    board=env.board,
    current_player=env.current_player,
    last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
)

# Check that TSS is enabled
from alphagomoku.tss import get_default_config
config = get_default_config()
print(f"TSS enabled: defend_open_four={config.defend_open_four}")
```

### Model Not Learning

**Problem**: Model not learning tactics after TSS disabled

**Solutions**:
```python
# Extend training before disabling
# Instead of disabling at epoch 50, try epoch 75
config = TSSConfig.for_training_epoch(min(epoch, 75))

# Increase training games per epoch
games = parallel_selfplay(model, num_games=200)  # Instead of 100

# Add evaluation to track learning
if epoch % 10 == 0:
    test_tactical_knowledge(model, epoch)
```

## Future Enhancements

Planned improvements to the TSS module:

1. **Pattern database**: Add more sophisticated threat patterns
2. **Opening integration**: Connect with opening book for early game
3. **Endgame transition**: Smooth handoff to endgame solver
4. **Learning adaptation**: Adapt threat priorities based on outcomes
5. **Parallel search**: Multi-threaded threat analysis
6. **Adaptive schedule**: Disable rules based on performance, not epoch
7. **Per-pattern metrics**: Track model learning per threat type
8. **Gradual fade**: Probabilistic TSS (100% → 0%) instead of on/off
9. **Double-three detection**: Enhanced complex pattern detection
10. **Performance optimization**: Cache immediate win checks

## Related Files

**Implementation:**
- `alphagomoku/tss/tss_search.py` - Main search logic
- `alphagomoku/tss/threat_detector.py` - Threat detection
- `alphagomoku/tss/tss_config.py` - Configuration system
- `alphagomoku/tss/__init__.py` - Public API

**Testing:**
- `tests/unit/test_tss.py` - Unit tests
- `tests/integration/test_tss_integration.py` - Integration tests
- `scripts/test_tss.py` - Manual testing script

**Integration:**
- `scripts/train.py` - Training with progressive TSS
- `apps/backend/app/inference.py` - Inference with difficulty-based TSS

## References

- AlphaZero paper: Learned purely from self-play without hard-coded rules
- Our approach: Use TSS as training wheels, remove gradually
- Balance: Learn faster early, develop true understanding later

## Summary

TSS provides tactical awareness for Gomoku AI through:
- Automatic threat detection and forced sequence search
- Progressive learning during training (full assistance → natural learning)
- Difficulty-based strength for inference (easy → hard)
- Seamless integration with MCTS and endgame solver
- Configurable depth, time limits, and threat patterns
- Detailed logging and monitoring for analysis

The model learns tactics naturally while receiving just enough guidance to bootstrap effectively!
