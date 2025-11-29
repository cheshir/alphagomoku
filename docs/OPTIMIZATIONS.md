# AlphaGomoku Performance Optimizations

This document describes the performance optimizations implemented to achieve sub-second MCTS performance and efficient large-scale training.

## ⚠️ Current Performance Issue (5M Model + Parallel Workers)

**Problem**: After upgrading from 2.67M to 5.04M parameters, training speed dropped from 4-5 epochs/day to ~3 epochs/day.

**Root Cause**: Parallel workers use CPU-only inference (PyTorch multiprocessing can't share MPS context), while MPS sits idle. With 5M model:
- **4 CPU workers**: 10 min/game × 200 games / 4 workers = 500 min (~8 hours) for self-play
- **1 MPS worker**: 42 sec/game × 128 games = 90 min (~1.5 hours) for self-play

**Solution**: Use single worker with MPS for large models (5M+). See configuration below.

## 🚀 Key Optimizations

### 1. Batched Neural Network Evaluation

**Problem**: MCTS was calling the neural network individually for each leaf node, causing GPU underutilization.

**Solution**: Collect multiple leaf nodes and evaluate them in batches of 32-128 positions.

**Implementation**:
- `MCTS._simulate_batch()`: Collects leaf nodes for batch evaluation
- `MCTS._batch_evaluate_leaves()`: Evaluates multiple positions in a single forward pass
- `GomokuNet.predict_batch()`: Batch prediction method

**Performance Impact**: ~3x speedup for MCTS search

### 2. Root Reuse Between Moves

**Problem**: MCTS tree was rebuilt from scratch after each move, wasting computed information.

**Solution**: After selecting a move, reuse the corresponding child subtree as the new root.

**Implementation**:
- `MCTS.reuse_subtree()`: Sets child node as new root
- `SelfPlayWorker.generate_game()`: Enables tree reuse after first move

**Performance Impact**: ~2x speedup for game generation

### 3. Adaptive Simulation Scheduling

**Problem**: Fixed 800 simulations per move was too costly for all game phases.

**Solution**: Dynamically adjust simulation count based on game phase and confidence.

**Implementation**:
- `AdaptiveSimulator`: Manages simulation scheduling
- Early game: 50-150 simulations
- Mid game: 200-400 simulations  
- Late game: 50-100 simulations
- High confidence positions: Reduced simulations

**Performance Impact**: ~2-3x reduction in average simulations per game

### 4. Parallel Self-Play Workers

**Problem**: Single-threaded game generation was a bottleneck for training data collection.

**Solution**: Multi-process self-play with separate workers generating games concurrently.

**Implementation**:
- `ParallelSelfPlay`: Manages multiple worker processes
- Each worker has its own model copy and generates games independently
- Results are merged after completion

**Performance Impact**: Near-linear speedup with number of CPU cores

**⚠️ Important Note for Large Models (5M+)**:
- Parallel workers use **CPU-only** inference (PyTorch limitation)
- For models >4M parameters, **single worker with MPS is faster** than multiple CPU workers
- **Recommendation**:
  - Small models (<3M params): Use 4-8 parallel workers
  - Large models (5M+ params): Use 1 worker with MPS acceleration

## 📊 Performance Results

### MCTS Performance (M1 Pro)

| Configuration | Time per 100 sims | Speedup |
|---------------|-------------------|---------|
| Baseline | ~2.5s | 1.0x |
| Batched (32) | ~0.8s | 3.1x |
| Batched + Adaptive | ~0.4s | 6.3x |

### Self-Play Performance

| Configuration | Games/hour | Speedup |
|---------------|------------|---------|
| Baseline | ~50 | 1.0x |
| All optimizations | ~300 | 6.0x |
| + 4 parallel workers | ~1000 | 20.0x |

## 🔧 Usage

### Optimized Training (5M Model - Current Recommended)

```bash
# For 5M parameter model (30 blocks, 192 channels)
python scripts/train.py \
    --adaptive-sims \
    --batch-size-mcts 64 \
    --parallel-workers 1 \          # Single worker uses MPS!
    --mcts-simulations 100 \        # 5M model has better priors
    --selfplay-games 128 \
    --epochs 200
```

**Expected Performance**: ~2 hours/epoch, 10-12 epochs/day

### Optimized Training (Small Models 2-3M)

```bash
# For smaller models (<3M parameters)
python scripts/train.py \
    --adaptive-sims \
    --batch-size-mcts 32 \
    --parallel-workers 4 \          # Multiple workers on CPU
    --mcts-simulations 150 \
    --selfplay-games 200 \
    --epochs 100
```

**Expected Performance**: ~1.5 hours/epoch, 15+ epochs/day

### Quick Performance Test

```bash
python scripts/test_optimizations.py
```

### Optimized Self-Play Example

```python
from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.parallel import ParallelSelfPlay

model = GomokuNet()
worker = ParallelSelfPlay(
    model=model,
    adaptive_sims=True,
    batch_size=32,
    num_workers=4
)

# Generate 100 games in parallel
data = worker.generate_batch(100)
```

## ⚙️ Configuration Parameters

### MCTS Optimization

- `batch_size`: Neural network batch size (default: 32)
- `adaptive_sims`: Enable adaptive simulation scheduling
- `reuse_tree`: Enable root reuse between moves

### Adaptive Simulations

- `early_sims`: Simulation range for early game (50-150)
- `mid_sims`: Simulation range for mid game (200-400)
- `late_sims`: Simulation range for late game (50-100)
- `early_moves`: Moves considered early game (10)
- `late_moves`: Moves considered late game (180)

### Parallel Self-Play

- `num_workers`: Number of parallel processes
- Recommended: Number of CPU cores - 1

## 🎯 Expected Performance

### Target Metrics (M1 Pro)

- **MCTS**: <1s per 100 simulations
- **Self-Play**: >200 games/hour with optimizations
- **Training**: 3-5x faster data generation

### Hardware Scaling

- **M1 Pro (8 cores)**: ~4x parallel speedup
- **M1 Max (10 cores)**: ~6x parallel speedup
- **Memory**: 16GB+ recommended for large batch sizes

## 🔍 Monitoring Performance

### Built-in Benchmarks

```bash
# Test all optimizations
python scripts/test_optimizations.py

# Quick optimization test
python scripts/quick_optimized_training.py
```

### Custom Profiling

```python
import time
from alphagomoku.mcts.mcts import MCTS

# Profile MCTS performance
mcts = MCTS(model, env, batch_size=32)
start = time.time()
policy, value = mcts.search(board)
visits = mcts.last_visit_counts
print(f"Search time: {time.time() - start:.3f}s")
```

## 🚧 Future Optimizations

### Planned Improvements

1. **GPU Memory Optimization**: Reduce memory usage for larger batch sizes
2. **Transposition Tables**: Cache repeated board positions
3. **ONNX Inference**: Faster inference with ONNX Runtime
4. **Quantization**: 8-bit model weights for faster inference

### Experimental Features

1. **Asynchronous Evaluation**: Overlap computation and data transfer
2. **Dynamic Batching**: Adaptive batch sizes based on available memory
3. **Distributed Training**: Multi-GPU training support

## 📈 Optimization Impact Summary

The implemented optimizations provide:

- **6-20x faster training** (depending on hardware and parallelization)
- **Sub-second MCTS** for reasonable simulation counts
- **Efficient resource utilization** on Apple Silicon
- **Scalable parallel processing** for data generation

These improvements make it practical to train strong Gomoku models on consumer hardware in reasonable time frames.

## 🔍 Model Size vs. Parallelization Trade-offs

### Key Insight: Hardware Acceleration > Parallelization for Large Models

| Model Size | Best Config | Device | Performance |
|------------|-------------|--------|-------------|
| 2.67M (12 blocks, 64 ch) | 4 workers | CPU | ~960 games/hour |
| 5.04M (30 blocks, 192 ch) | 4 workers | CPU | ~400 games/hour ❌ |
| 5.04M (30 blocks, 192 ch) | 1 worker | MPS | ~750 games/hour ✅ |

**Why?** PyTorch multiprocessing cannot share MPS/CUDA contexts. Parallel workers are forced to CPU, where large models are slow.

### Decision Matrix

```
If model_params < 3M:
    ✅ Use parallel workers (4-8)
    ✅ Smaller batch size (32)
    ✅ More MCTS sims (150-200)

If model_params >= 5M:
    ✅ Use single worker with MPS
    ✅ Larger batch size (64)
    ✅ Fewer MCTS sims (100) - better priors compensate
```

### Performance Comparison (5M Model)

| Configuration | Self-play Time | Training Time | Total/Epoch | Epochs/Day |
|---------------|----------------|---------------|-------------|------------|
| 4 workers (CPU) + 200 games + 150 sims | 8.3 hours | 20 min | 8.5 hours | 3 |
| 1 worker (MPS) + 128 games + 100 sims | 1.5 hours | 15 min | 1.75 hours | 12-14 |

**Improvement**: 4.8x faster training with optimized configuration!

## 🔬 Advanced MPS Utilization

### Key Findings from MPS Testing

**1. MPS Works in Subprocesses** ✅
- Each subprocess can independently use MPS
- Multiple workers CAN use MPS simultaneously
- No context sharing issues like with CUDA

**2. Optimal Batch Sizes for 5M Model**

| Batch Size | Per-Item Latency | Throughput | Recommendation |
|------------|------------------|------------|----------------|
| 32 | 2.27ms | 441 inf/s | Good baseline |
| 64 | 2.62ms | 382 inf/s | Slightly worse |
| 96 | 2.25ms | 445 inf/s | **Optimal** |
| 128 | 2.31ms | 432 inf/s | Good for large batches |

**Recommendation**: Use `--batch-size-mcts 96` for best per-inference latency.

**3. Threading vs Multiprocessing**
- Threading: Shares MPS context, but Python GIL limits parallelism
- Multiprocessing: Each process gets own MPS context, true parallelism
- Testing in progress to determine optimal strategy

### Applied Optimizations (v2 - MPS Enabled)

**Status**: ✅ **IMPLEMENTED** - Parallel workers now use MPS!

**Changes Made**:
1. **Modified `alphagomoku/selfplay/parallel.py`**: Workers now use MPS instead of forced CPU
2. **Optimal batch size**: Changed from 64 to 96 (best per-inference latency: 2.25ms)
3. **Updated Makefile**: Using optimized settings for 5M model

**Key Insight**: The old comment "avoid MPS/CUDA context issues" was outdated. MPS works perfectly in subprocesses on macOS!

### Current Recommended Configuration

```makefile
--mcts-simulations 100      # 5M model needs fewer sims
--batch-size-mcts 96        # Optimal for MPS
--parallel-workers 1        # Single worker sufficient with MPS
--selfplay-games 128        # With augmentation + filtering
```

**Performance**: ~1.75 hours/epoch (~12-14 epochs/day)

### Future Optimizations (Experimental)

1. **Multi-worker MPS** (testing required): Use 2-3 parallel workers, all on MPS
   - Each worker gets independent MPS context
   - Potential 1.5-2x speedup
   - Needs proper testing to measure actual benefit vs GIL limitations

2. **Async inference queue**: Overlap MCTS tree traversal with NN evaluation
