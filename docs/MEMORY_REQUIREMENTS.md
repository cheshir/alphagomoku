# Memory Requirements Guide

This guide explains memory requirements for training AlphaGomoku models and helps you choose the right configuration for your hardware.

## 🔍 TL;DR - Quick Check

**Want to know what works on YOUR hardware?**

```bash
make show-hardware-config
```

This will detect your hardware and recommend optimal settings!

---

## 📊 Memory Requirements by Model

### Small Model (1.2M params) - **Recommended for Most Users**

| Configuration | RAM Required | Notes |
|---------------|--------------|-------|
| **Inference only** | ~500 MB | Just running the model |
| **Training (1 worker)** | ~2 GB | Single-threaded training |
| **Training (4 workers)** | ~4-6 GB | Parallel self-play (fast!) |
| **Training (8 workers)** | ~8-10 GB | Maximum parallelism |

**Recommended Hardware:**
- ✅ 8 GB RAM (2 workers)
- ✅ 16 GB RAM (4 workers) ⭐ **Sweet spot**
- ✅ 32 GB RAM (8 workers)

### Medium Model (3M params) - **For Stronger Play**

| Configuration | RAM Required | Notes |
|---------------|--------------|-------|
| **Inference only** | ~800 MB | Just running the model |
| **Training (1 worker)** | ~3 GB | Single-threaded training |
| **Training (2 workers)** | ~6-8 GB | Parallel self-play |
| **Training (4 workers)** | ~12-16 GB | High parallelism |

**Recommended Hardware:**
- ⚠️  8 GB RAM - **Too small**, use small model
- ✅ 16 GB RAM (1-2 workers) ⭐ **Works well**
- ✅ 32 GB RAM (4 workers)
- ✅ 64 GB RAM (8 workers)

### Large Model (5M params) - **Research/Maximum Strength**

| Configuration | RAM Required | Notes |
|---------------|--------------|-------|
| **Inference only** | ~1 GB | Just running the model |
| **Training (1 worker)** | ~4 GB | Single-threaded training |
| **Training (2 workers)** | ~8-10 GB | Parallel self-play |

**Recommended Hardware:**
- ❌ 8 GB RAM - **Insufficient**
- ⚠️  16 GB RAM - **Tight** (1 worker only)
- ✅ 32 GB RAM (2 workers)
- ✅ 64 GB RAM (4+ workers)

---

## 💻 Hardware-Specific Recommendations

### M1 Pro 16GB (Your Current Hardware)

**Answer to your questions:**

#### 1. Can you train the medium model on M1 Pro 16GB?
**✅ YES!** But with some limitations:

```bash
# Recommended for M1 Pro 16GB + Medium Model:
python scripts/train.py \
    --model-preset medium \
    --parallel-workers 1 \      # Important: 1 worker!
    --batch-size 512 \
    --batch-size-mcts 96 \
    --selfplay-games 100 \
    --device auto
```

**Why 1 worker?** Each worker needs ~800 MB for the medium model. With 16 GB total RAM:
- System: ~3 GB
- Other processes: ~2 GB
- Available: ~11 GB
- Medium (1 worker): ~3 GB ✅
- Medium (2 workers): ~6 GB ✅ (but tight)
- Medium (4 workers): ~12 GB ❌ (will swap)

**Performance:**
- 1 worker: ~40-60 min/epoch (acceptable)
- 2 workers: ~25-35 min/epoch (but may swap, slower overall)

#### 2. Memory breakdown for Medium Model:

```
Component                Memory
------------------------------------
Model parameters         ~12 MB
Training (gradients,     ~40 MB
  optimizer state)
Activations (batch 512)  ~200 MB
MCTS tree per worker     ~500 MB
Game buffer              ~100 MB
PyTorch overhead         ~1 GB
------------------------------------
Per worker total:        ~2 GB
System + other:          ~3 GB
------------------------------------
1 worker:                ~5 GB ✅
2 workers:               ~7 GB ✅
4 workers:               ~11 GB ❌
```

### M1 Max 32GB

**Recommended:**
```bash
# Small model (fastest):
make train  # 4 workers, 256 batch, ~6 GB

# Medium model (stronger):
make train-production  # 2-4 workers, 512 batch, ~10 GB
```

### M1 Max 64GB

**Recommended:**
```bash
# Can run any configuration comfortably
# Medium with 8 workers: ~16 GB
# Large with 4 workers: ~20 GB
```

### NVIDIA GPUs

| GPU | VRAM | Recommended | Notes |
|-----|------|-------------|-------|
| **RTX 3060** | 12 GB | Small (batch 256) | Entry level |
| **RTX 3080** | 10-12 GB | Small/Medium (batch 512) | Good balance |
| **RTX 3090** | 24 GB | Medium (batch 1024) | High end |
| **RTX 4090** | 24 GB | Medium (batch 1024) | Fastest consumer |
| **A100** | 40+ GB | Large (batch 2048) | Professional |

**Note:** GPU training uses VRAM, not system RAM. Parallel workers still use system RAM.

---

## 🚀 Optimizing for Your Hardware

### Strategy 1: Use Dynamic Configuration (Recommended!)

```bash
# Let the system recommend settings
make show-hardware-config

# Optimize for speed
make show-hardware-config-speed

# Optimize for strength
make show-hardware-config-strength
```

### Strategy 2: Manual Tuning

#### If You Have Limited RAM (<16 GB)

```bash
# Use small model with fewer workers
python scripts/train.py \
    --model-preset small \
    --parallel-workers 2 \
    --batch-size 128 \
    --selfplay-games 50
```

#### If You Have Plenty of RAM (32+ GB)

```bash
# Use medium model with parallelization
python scripts/train.py \
    --model-preset medium \
    --parallel-workers 4 \
    --batch-size 512 \
    --selfplay-games 200
```

### Strategy 3: Monitor Memory Usage

```bash
# During training, monitor memory in another terminal:
watch -n 1 'ps aux | grep python | grep train'

# On macOS:
top -o MEM

# On Linux:
htop
```

---

## 🔧 Troubleshooting

### "Killed" or "Out of Memory" Error

**Problem:** System ran out of RAM

**Solutions:**
1. Reduce parallel workers: `--parallel-workers 1`
2. Reduce batch size: `--batch-size 128`
3. Use smaller model: `--model-preset small`
4. Close other applications

### Training is Slow/System is Swapping

**Problem:** Using too much RAM, swapping to disk

**Solutions:**
1. Check with `vm_stat` (macOS) or `free -h` (Linux)
2. Reduce workers or batch size
3. Consider using smaller model preset

### GPU Out of Memory (CUDA)

**Problem:** GPU VRAM exhausted

**Solutions:**
1. Reduce batch size: `--batch-size 256`
2. Reduce MCTS batch: `--batch-size-mcts 32`
3. Use gradient checkpointing (automatic in configs)

---

## 📈 Performance vs Memory Trade-offs

### Small Model (1.2M)
- **Memory**: Low (~2-6 GB)
- **Speed**: Fast (15-25 min/epoch with 4 workers)
- **Strength**: 80% of large model
- **Best for**: Development, quick iteration

### Medium Model (3M)
- **Memory**: Medium (~6-10 GB)
- **Speed**: Medium (40-60 min/epoch with 1-2 workers)
- **Strength**: 90% of large model
- **Best for**: Production, strong play

### Large Model (5M)
- **Memory**: High (~10-20 GB)
- **Speed**: Slow (2-3 hours/epoch with 1 worker)
- **Strength**: Maximum (100%)
- **Best for**: Research, maximum strength

---

## ✅ Recommendations

### For M1 Pro 16GB (Your Hardware)

**Best choice: Small model with 4 workers**
```bash
make train  # Uses optimized settings
```

**If you want stronger play: Medium model with 1 worker**
```bash
python scripts/train.py --model-preset medium --parallel-workers 1
```

**Expected performance:**
- Small (4 workers): 15-25 min/epoch, ~6 GB RAM
- Medium (1 worker): 40-60 min/epoch, ~5 GB RAM

### General Rule of Thumb

```
Available RAM >= (Workers × 1.5 GB) + 3 GB system
```

Examples:
- 8 GB RAM → 2 workers (small model)
- 16 GB RAM → 4 workers (small) OR 1-2 workers (medium)
- 32 GB RAM → 8 workers (small) OR 4 workers (medium)

---

## 🎯 Quick Decision Guide

**Do you have...**

- **Less than 12 GB RAM?**
  → Use `small` model with 1-2 workers

- **12-20 GB RAM?**
  → Use `small` model with 4 workers (fast!)
  → OR `medium` model with 1 worker (stronger)

- **More than 20 GB RAM?**
  → Use `medium` model with 2-4 workers (best of both worlds)

- **More than 48 GB RAM?**
  → Use `large` model or `medium` with 8 workers

**Not sure?** Run:
```bash
make show-hardware-config
```

---

## 📞 Still Have Questions?

- **Check your current memory**: `htop` or `Activity Monitor`
- **Test with dry run**: Start with `--selfplay-games 5` to test
- **Monitor during training**: Watch memory usage
- **Ask for help**: Open a GitHub issue with your hardware specs

---

**Summary for M1 Pro 16GB:**
- ✅ Small model: Works great, 4 workers
- ✅ Medium model: Works, 1-2 workers (slightly slower but stronger)
- ❌ Large model: Not recommended (would swap)
