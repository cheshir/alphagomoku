# TSS Progressive Learning Configuration

## Overview

The TSS module now supports **progressive disabling** of hard-coded tactical rules, allowing the neural network to learn tactical patterns naturally through self-play and MCTS.

## Philosophy

**Initial Training (Epochs 0-50):**
- TSS provides full tactical assistance
- Helps model learn correct tactical responses quickly
- Acts as a "teacher" showing the right moves

**Mid Training (Epochs 50-100):**
- Disable some TSS rules (broken-four defense)
- Model must learn semi-open patterns on its own
- TSS still helps with most critical patterns

**Late Training (Epochs 100+):**
- Disable most TSS rules (open-four defense too)
- Model handles all tactics through learned policy + MCTS
- Only keep game-rule enforcement (immediate 5-in-a-row)

## Usage

### In Training

```python
from alphagomoku.tss import TSSConfig, set_default_config

# Get config for current epoch
current_epoch = 75
config = TSSConfig.for_training_epoch(current_epoch)

# Set as global default
set_default_config(config)

# Now all TSS searches use this config
from alphagomoku.tss import tss_search, Position
result = tss_search(position, depth=4, time_cap_ms=100)
```

### Progressive Schedule

```python
def get_tss_config_for_epoch(epoch: int) -> TSSConfig:
    """
    Epoch 0-50: Full TSS
      - defend_immediate_five: True (always)
      - defend_open_four: True
      - defend_broken_four: True
      - defend_open_three: False (model learns from start)

    Epoch 50-100: Reduced TSS
      - defend_immediate_five: True (always)
      - defend_open_four: True
      - defend_broken_four: False (model learns)
      - defend_open_three: False

    Epoch 100+: Minimal TSS
      - defend_immediate_five: True (always)
      - defend_open_four: False (model learns)
      - defend_broken_four: False
      - defend_open_three: False
    """
    return TSSConfig.for_training_epoch(epoch)
```

### In Self-Play Script

```python
# In selfplay.py or parallel selfplay

class SelfPlayWorker:
    def __init__(self, model, env, current_epoch):
        self.model = model
        self.env = env

        # Configure TSS based on training progress
        tss_config = TSSConfig.for_training_epoch(current_epoch)
        set_default_config(tss_config)

        self.mcts = MCTS(model, env)

    def generate_game(self):
        # TSS will use the configured settings
        # MCTS will get tactical guidance based on current epoch
        ...
```

### For Inference

```python
from alphagomoku.tss import TSSConfig, set_default_config

# Get config for difficulty
config = TSSConfig.for_inference("hard")  # or "easy", "medium"
set_default_config(config)

# Easy: Minimal TSS (natural learned behavior)
# Medium: Some TSS (balanced)
# Hard: Full TSS (maximum strength)
```

## Configuration Options

### TSSConfig Fields

```python
@dataclass
class TSSConfig:
    # Game rules (always enabled)
    defend_immediate_five: bool = True

    # Tactical patterns (can be disabled)
    defend_open_four: bool = True      # .XXXX. patterns
    defend_broken_four: bool = True    # XXXX. or X.XXX patterns
    defend_open_three: bool = False    # .XXX. patterns

    # Win search
    search_forced_wins: bool = True
    max_search_depth: int = 6
```

### Why This Schedule?

**Open-three (always off):**
- Model should learn this from the beginning
- Not immediately game-losing, so safe to learn
- Teaches model to evaluate long-term threats

**Broken-four (off at epoch 50):**
- Model has seen enough examples by epoch 50
- Semi-open patterns are common in training data
- Policy network should recognize these by now

**Open-four (off at epoch 100):**
- Most critical tactical pattern
- Keep longer to ensure strong tactical foundation
- By epoch 100, model is experienced enough

**Immediate five (always on):**
- This is a game rule, not a learned tactic
- Always block immediate losses
- Ensures valid games during training

## Monitoring Learning

### Check if Model Learned Tactics

```python
# Test position with open-four
board = create_open_four_position()
pos = Position(board=board, current_player=-1)

# Test with TSS disabled
config_no_tss = TSSConfig(
    defend_immediate_five=True,
    defend_open_four=False,
    defend_broken_four=False,
    defend_open_three=False,
)

detector = ThreatDetector(config=config_no_tss)
defense_moves = detector.must_defend(pos, -1)

if not defense_moves:
    print("TSS doesn't force defense (as expected)")

    # Now check if MCTS/policy finds the right move
    policy, value = model(pos)
    best_move = policy.argmax()

    if best_move in expected_defense_positions:
        print("✓ Model learned to defend open-four!")
    else:
        print("✗ Model hasn't learned yet, keep TSS enabled")
```

## Benefits

1. **Faster Early Learning**: TSS helps model learn correct responses quickly
2. **Natural Late Learning**: Model develops true understanding, not just memorization
3. **Stronger Final Model**: Learns patterns through experience, not hard-coding
4. **Flexible Inference**: Can tune TSS per difficulty level

## Implementation Files

- `alphagomoku/tss/tss_config.py` - Configuration dataclass and helpers
- `alphagomoku/tss/threat_detector.py` - Uses config in `must_defend()`
- `alphagomoku/tss/__init__.py` - Exports config classes

## Example: Training Run

```python
# In your training loop

for epoch in range(200):
    # Update TSS configuration
    tss_config = TSSConfig.for_training_epoch(epoch)
    set_default_config(tss_config)

    print(f"Epoch {epoch}:")
    print(f"  - defend_open_four: {tss_config.defend_open_four}")
    print(f"  - defend_broken_four: {tss_config.defend_broken_four}")

    # Generate self-play games
    games = parallel_selfplay(model, num_games=500)

    # Train
    train_epoch(model, games)

    # Evaluate
    if epoch % 10 == 0:
        evaluate_model(model, epoch)
```

## Future Work

- **Adaptive schedule**: Disable rules based on model performance, not epoch
- **Per-pattern metrics**: Track how well model handles each pattern type
- **Gradual fade**: Probabilistically apply TSS rules (100% → 0%) instead of on/off
- **Double-three detection**: Add configurable defense for complex patterns

## References

- AlphaZero paper: Learned purely from self-play without hard-coded rules
- Our approach: Use TSS as training wheels, remove gradually
- Balance: Learn faster early, but develop true understanding later
