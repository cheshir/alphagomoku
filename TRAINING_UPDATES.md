# Training Configuration Updates

**Date:** 2025-11-30
**Changes:** Updated all training presets to use AlphaZero methodology

---

## What Changed

All training configurations now use `--difficulty easy` by default:
- ✅ `make train-fast` → uses `difficulty: easy`
- ✅ `make train` → uses `difficulty: easy`
- ✅ `make train-production` → uses `difficulty: easy`

## Why This Change

### Problem Discovered
Training with `--difficulty medium` (which enables TSS) caused:
- ❌ 4-6x slower training (20+ min/game vs 1-2 min/game)
- ❌ 0-20% GPU utilization (TSS runs on CPU)
- ❌ Against AlphaZero methodology (no domain heuristics)

### Solution
Train with `--difficulty easy` (pure MCTS):
- ✅ 4-6x faster training
- ✅ 80-100% GPU utilization
- ✅ Follows AlphaZero methodology
- ✅ Network learns threats through self-play

## Key Concepts

### Model Preset vs Difficulty

Two **independent** settings:

| Setting | Controls | Values |
|---------|----------|--------|
| `--model-preset` | Neural network size | `small`, `medium`, `large` |
| `--difficulty` | Search augmentation | `easy`, `medium`, `strong` |

**They are NOT related!**

### Training vs Inference

| Phase | Use Difficulty | Why |
|-------|----------------|-----|
| **Training** | `easy` | Fast, GPU-accelerated, pure learning |
| **Inference** | `medium` | Strong tactical play with TSS |
| **Evaluation** | `medium` | Measure true strength with enhancements |

## Updated Commands

### For Training (Fast, AlphaZero Style)
```bash
# Fastest
make train-fast

# Balanced (recommended)
make train

# Production (strongest model)
make train-production
```

All use `--difficulty easy` automatically!

### For Inference/Playing (TSS-Enhanced)
```python
# In your web app or game:
model = GomokuNet.from_preset('medium', device='cuda')
search = UnifiedSearch(model, env, difficulty='medium')  # TSS enabled
move = search.get_best_move(state)
```

## Migration Guide

### If You Have Existing Checkpoints

Checkpoints are compatible:
- Model architecture unchanged
- Only self-play strategy differs
- Can resume with `--difficulty easy`

### If You Were Training with TSS

**Old (slow):**
```bash
python scripts/train.py --model-preset medium --difficulty medium
```

**New (fast, recommended):**
```bash
python scripts/train.py --model-preset medium --difficulty easy
```

**OR:**
```bash
make train-production  # Uses optimal settings automatically
```

## Performance Comparison

### Google Colab T4

| Config | Difficulty | Time/Epoch | Total (200 epochs) | GPU Util |
|--------|------------|------------|-------------------|----------|
| **Old** | medium | 2-3 hours | 20-30 days | 0-20% ❌ |
| **New** | easy | 20-30 min | 3-5 days | 80-100% ✅ |

**Result: 4-6x faster!**

### M1 Pro 16GB

| Config | Difficulty | Time/Epoch | Total (200 epochs) | GPU Util |
|--------|------------|------------|-------------------|----------|
| **Old** | medium | 1-2 hours | 10-15 days | 0-20% ❌ |
| **New** | easy | 15-20 min | 2-3 days | 80-100% ✅ |

## Documentation

See detailed explanation:
- **[docs/TRAINING_PHILOSOPHY.md](docs/TRAINING_PHILOSOPHY.md)** - Full rationale and examples
- **[docs/CUDA_MULTIPROCESSING_FIX.md](docs/CUDA_MULTIPROCESSING_FIX.md)** - CUDA setup fixes
- **[docs/CLOUD_VM_RECOMMENDATIONS.md](docs/CLOUD_VM_RECOMMENDATIONS.md)** - Cloud training guide

## Quick Reference

### Training Commands
```bash
make train-fast        # 5-10 min/epoch, small model
make train             # 10-20 min/epoch, small model (recommended)
make train-production  # 20-30 min/epoch on CUDA, medium model
```

### Check Configuration
```bash
make show-config  # See all presets
make help         # See all commands
```

### Verify Setup
```bash
python scripts/check_device.py  # Check GPU availability
```

## FAQ

**Q: Will my model be weaker without TSS during training?**
A: No! The network learns threats through self-play. After 200 epochs, it recognizes threats as well as (or better than) TSS.

**Q: When do I use TSS then?**
A: During inference (web app, playing) and evaluation. Not during training.

**Q: Can I still train with `difficulty: medium`?**
A: Yes, but it's 4-6x slower with no benefit. Not recommended.

**Q: Do I need to retrain from scratch?**
A: No, existing checkpoints work fine. You can resume with `--difficulty easy`.

---

**Bottom Line:** Use `make train` or `make train-production` for optimal training!
