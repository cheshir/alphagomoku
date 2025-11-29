# Device-Specific Training Guide

## Quick Start

### Auto-detect best device:
```bash
make train  # Automatically selects: CUDA > MPS > CPU
```

### Force specific device:
```bash
make train-cuda  # NVIDIA GPU (A100, RTX, etc.)
make train-mps   # Apple Silicon (M1/M2/M3)
make train-cpu   # CPU only
```

### Manual device selection:
```bash
python scripts/train.py --device cuda --batch-size 2048 ...
python scripts/train.py --device mps --batch-size 512 ...
python scripts/train.py --device cpu --batch-size 256 ...
```

## Device-Specific Configurations

### CUDA (NVIDIA GPUs)
**Optimized for:** A100, A6000, RTX 4090, etc.

```bash
make train-cuda
```

**Settings:**
- Batch size: **2048** (4x faster than MPS)
- MCTS batch: **128** (larger for better throughput)
- Parallel workers: **4**
- Expected speed: **~2-3 hours/epoch** on A100

**Memory:** Uses ~25 GB on A100 (40 GB available)

### MPS (Apple Silicon)
**Optimized for:** M1 Pro/Max, M2/M3 series

```bash
make train-mps
```

**Settings:**
- Batch size: **512** (limited by 18 GB unified memory)
- MCTS batch: **64**
- Parallel workers: **4**
- Expected speed: **~15-30 hours/epoch** (if no swapping)

**WARNING:** Ensure system RAM >= 32 GB to avoid swap!
- 16 GB RAM → massive swapping → 39+ hours/epoch
- 32 GB RAM → no swapping → 15-20 hours/epoch

### CPU
**For:** Systems without GPU or testing

```bash
make train-cpu
```

**Settings:**
- Batch size: **256** (smaller for memory efficiency)
- MCTS batch: **32**
- Parallel workers: **8** (more workers compensate for no GPU)
- Expected speed: **~50-100 hours/epoch**

**Not recommended** for full 200-epoch training (would take months)

## Performance Comparison

| Device | Batch Size | Speed/Epoch | Total (200 epochs) |
|--------|-----------|-------------|-------------------|
| A100 GPU | 2048 | ~2.5h | ~21 days |
| M1 Pro (32GB) | 512 | ~15h | ~125 days |
| M1 Pro (16GB) | 512 | ~39h | ~325 days |
| CPU | 256 | ~75h | ~625 days |

## Memory Optimization Tips

### If running out of memory on MPS:

1. **Reduce LMDB map size:**
   ```bash
   --map-size-gb 2  # Instead of 12
   ```

2. **Reduce buffer size:**
   ```bash
   --buffer-max-size 100000  # Instead of 500000
   ```

3. **Enable gradient checkpointing:**
   Edit scripts/train.py:
   ```python
   model = GomokuNet(..., use_checkpoint=True)
   ```
   Then increase batch size back to 1024.

### If running out of memory on CUDA:

CUDA has 40 GB, should not happen with current settings.
If it does, reduce batch size:
```bash
--batch-size 1024  # Instead of 2048
```

## Device Detection Logic

The `--device auto` flag detects in this order:
1. **CUDA** - If available → use CUDA
2. **MPS** - If CUDA unavailable but MPS available → use MPS
3. **CPU** - Fallback if no GPU

## Debugging

Enable detailed memory logging:
```bash
python scripts/train.py --device auto --debug-memory ...
```

Shows memory at each phase:
- Epoch start
- Before/after selfplay
- Before/after training

Helps identify memory leaks or pressure issues.
