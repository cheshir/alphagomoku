# TSS (Threat-Space Search) Usage Guide

## Overview

The Threat-Space Search (TSS) module provides tactical analysis for Gomoku positions, detecting forced win/defense sequences and critical threats. TSS is designed to work alongside MCTS to improve tactical play.

## Quick Start

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

## Core Components

### Position Class

Represents a board state for TSS analysis:

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

### ThreatDetector Class

Detects tactical patterns:

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

### TSS Search Function

Main search interface:

```python
from alphagomoku.tss import tss_search

result = tss_search(
    position=position,      # Position to analyze
    depth=4,               # Maximum search depth
    time_cap_ms=100        # Time limit in milliseconds
)

# Result attributes
result.forced_move         # (row, col) or None
result.is_forced_win      # Boolean
result.is_forced_defense  # Boolean
result.search_stats       # Dict with 'nodes_visited', 'time_ms', 'reason'
```

## Integration with MCTS

TSS can guide MCTS move selection in tactical situations:

```python
class TSSGuidedMCTS:
    def __init__(self, model, env, tss_depth=4, tss_time_cap=50):
        self.mcts = MCTS(model, env)
        self.tss_depth = tss_depth
        self.tss_time_cap = tss_time_cap
    
    def search(self, state, temperature=1.0):
        # Convert to TSS position
        position = Position(board=state, current_player=self.env.current_player)
        
        # Check for forced moves
        tss_result = tss_search(position, self.tss_depth, self.tss_time_cap)
        
        if tss_result.forced_move:
            # Override MCTS with forced move
            action_probs = np.zeros(15 * 15)
            r, c = tss_result.forced_move
            action_probs[r * 15 + c] = 1.0
            return action_probs, np.array([1.0])
        
        # Use MCTS for normal positions
        return self.mcts.search(state, temperature)
```

## Configuration Parameters

### Search Depth
- **Depth 2-3**: Fast tactical checks (~10-50ms)
- **Depth 4-5**: Balanced analysis (~50-200ms)
- **Depth 6+**: Deep analysis (~200ms+)

### Time Limits
- **Easy mode**: 30-50ms time cap
- **Medium mode**: 100-200ms time cap
- **Strong mode**: 300-500ms time cap

### Difficulty Integration
```python
def get_tss_config(difficulty):
    configs = {
        'easy': {'depth': 2, 'time_cap': 30},
        'medium': {'depth': 4, 'time_cap': 100},
        'strong': {'depth': 6, 'time_cap': 300}
    }
    return configs.get(difficulty, configs['medium'])
```

## Threat Types

TSS detects the following threat patterns:

- **OPEN_FOUR**: Four stones with open ends (immediate win threat)
- **BROKEN_FOUR**: Four stones with gap that can be completed
- **OPEN_THREE**: Three stones with open ends (creates double threat)
- **DOUBLE_THREE**: Multiple three-in-a-row threats
- **DOUBLE_FOUR**: Multiple four-in-a-row threats

## Performance Characteristics

### Typical Performance
- **Simple positions**: 1-5ms, 1-10 nodes
- **Tactical positions**: 10-100ms, 10-100 nodes
- **Complex positions**: 100-500ms, 100-1000 nodes

### Memory Usage
- **Position storage**: ~1KB per position
- **Search tree**: ~10-100KB for typical searches
- **Threat detection**: Minimal overhead

## Best Practices

### When to Use TSS
1. **Always check** for immediate threats before MCTS
2. **Use in midgame** when tactical patterns emerge
3. **Disable in endgame** when endgame solver takes over
4. **Skip in opening** unless specific tactical training

### Integration Tips
1. **Time budgeting**: Reserve 10-20% of move time for TSS
2. **Depth scaling**: Increase depth with remaining game time
3. **Logging**: Track TSS override frequency for analysis
4. **Fallback**: Always have MCTS as backup if TSS fails

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

## Testing and Validation

### Unit Tests
```bash
python -m pytest tests/unit/test_tss.py -v
```

### Integration Tests
```bash
python -m pytest tests/integration/test_tss_integration.py -v
```

### Manual Testing
```bash
python scripts/test_tss.py
python scripts/tss_mcts_example.py
```

## Logging and Monitoring

TSS provides detailed logging for analysis:

```python
result = tss_search(position, depth=4, time_cap_ms=100)
stats = result.search_stats

print(f"Nodes visited: {stats['nodes_visited']}")
print(f"Time used: {stats['time_ms']:.1f}ms")
print(f"Reason: {stats['reason']}")

# Reasons include:
# - 'immediate_defense': Found forced defense
# - 'forced_win': Found winning sequence
# - 'no_forced_sequence': No tactical override needed
```

## Future Enhancements

The TSS module is designed for extensibility:

1. **Pattern database**: Add more sophisticated threat patterns
2. **Opening integration**: Connect with opening book
3. **Endgame transition**: Smooth handoff to endgame solver
4. **Learning**: Adapt threat priorities based on game outcomes
5. **Parallel search**: Multi-threaded threat analysis

## Troubleshooting

### Common Issues

**TSS too slow**: Reduce depth or time cap
```python
result = tss_search(position, depth=2, time_cap_ms=30)
```

**False positives**: Check threat detection logic
```python
detector = ThreatDetector()
threats = detector.detect_threats(position, player)
# Manually verify threat patterns
```

**Integration errors**: Ensure position conversion is correct
```python
# Convert from GomokuEnv to TSS Position
position = Position(
    board=env.board,
    current_player=env.current_player,
    last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
)
```

For more examples and advanced usage, see the test files and example scripts in the repository.