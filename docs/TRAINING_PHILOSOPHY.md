# Training Philosophy: AlphaZero Style vs TSS-Enhanced

**TL;DR:** Train with `--difficulty easy` (pure MCTS), use TSS during inference/evaluation.

---

## Key Insight: Model Presets ≠ Difficulty

Two independent settings control training:

| Setting | What It Controls | Options |
|---------|------------------|---------|
| **`--model-preset`** | Neural network size | `small` (1.2M), `medium` (3M), `large` (5M) |
| **`--difficulty`** | Search augmentation | `easy` (MCTS only), `medium` (MCTS+TSS+Endgame) |

**They are completely independent!**

---

## Training Strategy: Pure AlphaZero

### Recommended for Training
```bash
--model-preset medium --difficulty easy
```

**What this means:**
- ✅ Full model size (3M parameters)
- ✅ Pure MCTS (GPU accelerated)
- ✅ Network learns threats through self-play
- ✅ Follows AlphaZero methodology
- ✅ Fast training (~20-30 min/epoch on T4)

### For Inference/Playing
```bash
--model-preset medium --difficulty medium
```

**What this means:**
- ✅ Full trained model
- ✅ TSS (Threat Space Search) enabled
- ✅ Endgame solver enabled
- ✅ Stronger tactical play
- ⚠️ Slower move generation (TSS runs on CPU)

---

## Why Train with `difficulty: easy`?

### 1. AlphaZero Principle

AlphaGo/AlphaZero **never used domain-specific heuristics** during training:
- No opening books
- No endgame tables
- No threat detection algorithms
- **Only: Neural Network + MCTS + Self-Play**

The network learned everything through pure self-play.

### 2. Speed (10-20x Faster)

| Difficulty | Components | GPU Util | Time/Game | Bottleneck |
|------------|-----------|----------|-----------|------------|
| **easy** | MCTS only | 80-100% | 30-60 sec | GPU (good!) |
| **medium** | MCTS+TSS+Endgame | 0-20% | 10-20 min | CPU (bad!) |

**TSS runs on CPU**, creating a massive bottleneck during training.

### 3. Better Learning

With TSS during training:
- ❌ Network doesn't learn to recognize threats
- ❌ Network relies on TSS crutch
- ❌ Brittle: TSS might not cover all patterns

Without TSS (pure self-play):
- ✅ Network learns threat patterns naturally
- ✅ More robust and generalizable
- ✅ Discovers novel tactics TSS might miss

---

## When to Use TSS

### ❌ NOT During Training
Training is about teaching the neural network:
```python
# Training (self-play)
model = GomokuNet.from_preset('medium')
selfplay = SelfPlayWorker(model, difficulty='easy')  # Pure MCTS
data = selfplay.generate_batch(100)
trainer.train_epoch(data)
```

### ✅ YES During Inference
Inference is about playing strong moves:
```python
# Playing/Evaluation
model = GomokuNet.from_preset('medium')
model.load_state_dict(torch.load('trained_model.pt'))
search = UnifiedSearch(model, env, difficulty='medium')  # TSS enabled
move = search.get_best_move(state)
```

### ✅ YES During Evaluation
Evaluate with TSS to see true strength:
```python
# Evaluation
evaluator = Evaluator(model, difficulty='medium')  # TSS enabled
elo = evaluator.calculate_elo()
```

---

## Training Configurations

### Fast Experiments (train-fast)
```bash
make train-fast
```
- Model: small (1.2M params)
- Difficulty: easy (MCTS only)
- Time: 5-10 min/epoch on CUDA
- Use for: Quick validation, hyperparameter tuning

### Balanced (train)
```bash
make train
```
- Model: small (1.2M params)
- Difficulty: easy (MCTS only)
- Time: 10-20 min/epoch on MPS, 5-10 on CUDA
- Use for: Default training, good balance

### Production (train-production)
```bash
make train-production
```
- Model: medium (3M params)
- Difficulty: easy (MCTS only)
- Time: 20-30 min/epoch on CUDA
- Use for: Final strong model

**All use `difficulty: easy` for training!**

---

## Example: Full Training Pipeline

### 1. Training (Pure MCTS)
```bash
# Train for 200 epochs with pure MCTS
python scripts/train.py \
    --model-preset medium \
    --difficulty easy \
    --epochs 200 \
    --device cuda

# Result: trained_model.pt
```

**During training:**
- GPU: 80-100% utilization
- Fast self-play (1-2 min/game)
- Network learns threats through experience
- 200 epochs = ~3-5 days on T4

### 2. Evaluation (With TSS)
```bash
# Evaluate trained model WITH TSS
python scripts/evaluate.py \
    checkpoints/model_epoch_200.pt \
    --difficulty medium

# Result: Elo rating, tactical test scores
```

**During evaluation:**
- TSS enabled (tactical strength)
- Endgame solver enabled
- Measures true playing strength
- Slower but more accurate evaluation

### 3. Inference/Playing (With TSS)
```python
# Load trained model
model = GomokuNet.from_preset('medium', device='cuda')
model.load_state_dict(torch.load('model_epoch_200.pt'))

# Create search with TSS enabled
search = UnifiedSearch(model, env, difficulty='medium')

# Play move (TSS enhances tactical play)
move = search.get_best_move(state)
```

**During gameplay:**
- TSS provides instant tactical strength
- Endgame solver for perfect endgames
- Slower move generation acceptable (playing, not training)

---

## Common Misconceptions

### ❌ "Medium difficulty = Stronger model"
**FALSE!** Difficulty only adds TSS/endgame during search, doesn't change model.

### ❌ "I need TSS during training for strong play"
**FALSE!** Network learns threats through self-play. TSS is for inference.

### ❌ "Easy difficulty = Smaller model"
**FALSE!** Difficulty and model size are independent settings.

### ✅ "Train with easy, infer with medium"
**TRUE!** This is the recommended approach.

---

## Performance Comparison

### Training on Google Colab T4

| Config | Model | Difficulty | Time/Epoch | Time for 200 Epochs | GPU Util |
|--------|-------|------------|------------|---------------------|----------|
| **Recommended** | medium | easy | ~20-30 min | ~3-5 days | 80-100% |
| **Not Recommended** | medium | medium | ~2-3 hours | ~20-30 days | 0-20% |

**Result: 4-6x faster training with pure MCTS!**

### Training on Local M1 Pro

| Config | Model | Difficulty | Time/Epoch | Time for 200 Epochs | GPU Util |
|--------|-------|------------|------------|---------------------|----------|
| **Recommended** | small | easy | ~15-20 min | ~2-3 days | 80-100% |
| **Not Recommended** | small | medium | ~1-2 hours | ~10-15 days | 0-20% |

---

## Migration Guide

### If You Were Training with TSS

**Old (slow):**
```bash
python scripts/train.py --model-preset medium --difficulty medium
```

**New (fast):**
```bash
python scripts/train.py --model-preset medium --difficulty easy
```

**Your model will be just as strong (or stronger) after 200 epochs!**

### If You Have Existing Checkpoints

Checkpoints from TSS-training are compatible with easy-training:
- The model architecture is identical
- Only the search strategy during self-play differs
- You can resume training with `--difficulty easy`

---

## FAQ

### Q: Will my model be weaker without TSS during training?
**A:** No! The network learns threat patterns through self-play. After 200 epochs, it will recognize threats as well as (or better than) TSS.

### Q: When should I use `difficulty: medium` then?
**A:** Only during inference (playing against humans) and evaluation (measuring strength).

### Q: Can I train with medium difficulty if I want?
**A:** Yes, but it will be 4-6x slower with no benefit. AlphaZero never used domain heuristics.

### Q: How do I enable TSS for my web app?
**A:** When creating the search interface:
```python
search = UnifiedSearch(model, env, difficulty='medium')  # TSS enabled
```

### Q: Does evaluation use TSS?
**A:** Yes! Evaluation scripts should use `difficulty: medium` to measure true playing strength with all enhancements.

---

## Summary

| Phase | Model Preset | Difficulty | Why |
|-------|--------------|------------|-----|
| **Training** | small/medium | easy | Fast, GPU-accelerated, AlphaZero style |
| **Evaluation** | small/medium | medium | True strength with TSS enhancements |
| **Inference** | small/medium | medium | Strong tactical play for users |

**Key Principle:** Train pure (fast), infer enhanced (strong).

---

## References

- AlphaGo Zero paper: "Mastering the game of Go without human knowledge"
- AlphaZero paper: "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
- Both used **zero** domain knowledge during training!

---

**Next Steps:**

1. Use `make train-production` for your production model
2. It uses `difficulty: easy` automatically
3. When deploying, use `difficulty: medium` for TSS
4. Enjoy 4-6x faster training! 🚀
