# Training Pipeline: Where GPU is Actually Used

## CRITICAL CLARIFICATION

The "99% CPU" measurement was ONLY for **self-play move generation**, NOT the complete training pipeline!

## Complete Training Pipeline

### Phase 1: Self-Play (Game Generation)
**Purpose**: Generate training data (states, policies, outcomes)

```
For each game:
  For each move:
    1. MCTS tree traversal (CPU) ← 99% of time
    2. Neural network inference (GPU) ← 1% of time
    Total: ~3.7s per move on MPS, ~0.5-1s on CUDA
```

**GPU Usage**: Low (only batched inference)
**CPU Usage**: High (tree traversal)

### Phase 2: Neural Network Training
**Purpose**: Train the neural network on collected data

```
For each batch (512-1024 positions):
  1. Load batch from buffer (CPU/RAM)
  2. Forward pass (GPU) ← ~40% of training time
  3. Calculate loss (GPU) ← ~10% of training time
  4. Backward pass (GPU) ← ~40% of training time
  5. Update weights (GPU) ← ~10% of training time
```

**GPU Usage**: HIGH (95-99% GPU utilization)
**CPU Usage**: Low (just data loading)

## Complete Epoch Breakdown

### Time Distribution (200 games per epoch)

| Phase | MPS (M1 Max) | CUDA (T4) | GPU Util | What's Happening |
|-------|--------------|-----------|----------|------------------|
| **Self-play** | 3h 40m | 30-40 min | 5-20% | MCTS tree search (CPU-bound) |
| **Training** | 10-20 min | 3-5 min | 95-99% | Neural net training (GPU-bound) |
| **Total** | ~4 hours | ~35-45 min | ~25% avg | Combined workload |

### Why CUDA is Still Much Faster Overall

Even though self-play is CPU-bound, CUDA systems win because:

1. **Faster CPU**: Xeon/EPYC > Apple Silicon for single-threaded MCTS
2. **No .cpu() overhead**: 600ms saved per move (10-20x speedup for inference portion)
3. **Better multi-core**: Can run 4+ parallel workers efficiently
4. **Faster training**: 3-5 min vs 10-20 min per epoch

### Real Training Time Comparison

**Training 1000 epochs (200 games each)**:

| Platform | Self-Play | NN Training | Total | Cost |
|----------|-----------|-------------|-------|------|
| MPS (M1 Max) | 3,670h | 167h | **3,837h (160 days)** | Free (your laptop) |
| CUDA (T4, 1 worker) | 583h | 50h | **633h (26 days)** | ~$220 |
| CUDA (T4, 4 workers) | 146h | 50h | **196h (8 days)** | ~$70 |
| CUDA (RTX 4090, 4 workers) | 100h | 25h | **125h (5 days)** | ~$100 or owned |

## AlphaGo/AlphaZero Architecture

### How DeepMind Did It

**AlphaGo Zero** (2017):
- 4 TPUs (Tensor Processing Units) for self-play
- 64 TPUs for neural network training
- ~40 days of training

**Key Architecture**:
```
Self-Play Workers (4 TPUs):
  - Each TPU runs MCTS + inference
  - Generate 4.9M games over 40 days
  - Batched inference reduces CPU bottleneck

Training Workers (64 TPUs):
  - Massive parallel training
  - 700K training steps
  - Batch size: 2048
```

### What They Did Differently

1. **Custom MCTS in C++**
   - Not Python! ~10-100x faster tree traversal
   - Still CPU-bound, but much faster CPU code

2. **Batched Inference at Root**
   - Instead of batching during tree traversal
   - Evaluate all legal moves at root in one batch
   - Reduces number of inference calls

3. **Asynchronous Execution**
   - Tree traversal on CPU
   - Inference on GPU
   - Both run in parallel (pipeline)

4. **Distributed Architecture**
   - Many machines running self-play
   - Separate cluster for training
   - Queue-based communication

## Can We Move More to GPU?

### What Can Be GPU-Accelerated

✅ **Neural network inference** - Already on GPU
✅ **Neural network training** - Already on GPU
⚠️ **Legal move generation** - Could use GPU, but overhead high
❌ **Tree traversal** - Inherently sequential, not parallelizable

### Why MCTS Can't Fully Use GPU

**MCTS Algorithm**:
```python
def select_node(root):
    node = root
    while not node.is_leaf():
        node = best_child(node)  # Sequential decision
    return node
```

This is **inherently sequential**:
- Each decision depends on previous decision
- Can't parallelize within a single tree traversal
- GPU is designed for massive parallelism

**GPU is good for**:
- Matrix multiplication (1000s of parallel operations)
- Convolutions (parallel across spatial dimensions)

**GPU is bad for**:
- Sequential decisions (if-then-else chains)
- Tree traversal (pointer chasing)
- Dynamic memory allocation

### What We CAN Optimize

#### 1. **Batch at Root (Virtual Loss)**
Instead of batching during traversal, batch at root:

```python
# Current: Traverse tree, batch leaf evaluations
for sim in range(num_sims):
    leaf = select_leaf()
    if batch_full:
        evaluate_batch()  # GPU

# Optimized: All root actions in one batch
root_batch = [state + action for action in legal_actions]
evaluate_batch(root_batch)  # GPU - one big batch
for sim in range(num_sims):
    leaf = select_leaf()  # Use cached root values
```

**Benefit**: Fewer GPU calls, larger batches
**Tradeoff**: Less accurate (uses root values for whole tree)

#### 2. **Rewrite MCTS in C++/Cython**
```cpp
// C++ is 10-100x faster than Python for tree traversal
Node* select_leaf(Node* root) {
    while (!root->is_leaf()) {
        root = best_child(root);  // Native loop, no Python overhead
    }
    return root;
}
```

**Benefit**: 10-100x faster tree traversal
**Tradeoff**: Complex to implement and maintain

#### 3. **Use Numba JIT**
```python
@numba.jit(nopython=True)
def ucb_score(node):
    return node.value + c_puct * node.prior * sqrt(parent.visits) / (1 + node.visits)
```

**Benefit**: 5-10x faster hot loops
**Tradeoff**: Limited Python features in JIT functions

#### 4. **GPU-Accelerated Legal Move Generation**
```python
# Current: NumPy on CPU
legal_mask = get_legal_moves(state)  # CPU

# Optimized: PyTorch on GPU
legal_mask = get_legal_moves_gpu(state_tensor)  # GPU
```

**Benefit**: Could save 10-50ms per move
**Tradeoff**: GPU kernel launch overhead might cancel gains

## Practical Recommendations

### For Current Codebase

✅ **Keep as-is for now**:
- Self-play is fast enough on CUDA (0.5-1s/move)
- Training is already GPU-accelerated
- Optimization is complex and error-prone

### For Future Optimization (in order of impact)

1. **Use more parallel workers** (easiest, 4x speedup)
   ```bash
   --parallel-workers 4  # Run 4 games in parallel
   ```

2. **Numba JIT for hot paths** (medium effort, 5-10x for tree ops)
   - Compile UCB calculation
   - Compile node selection
   - Still Python-compatible

3. **Cython for MCTS core** (hard, 10-50x for tree ops)
   - Rewrite Node class in Cython
   - Compile to C extension
   - Maintains Python interface

4. **Root batching** (medium effort, 2-3x inference speedup)
   - Change MCTS algorithm
   - Trade accuracy for speed
   - Good for inference, risky for training

## Bottom Line

### Your Question 1: Why CUDA vs MPS?

**Because training has TWO phases**:
- Self-play: 90% of time, benefits from faster CPU + multi-worker
- NN training: 10% of time, 100% GPU-bound

**Overall speedup**: 5-8x faster on CUDA even though self-play is CPU-bound

### Your Question 2: AlphaZero and GPU Optimization

**AlphaZero used**:
- Custom C++ MCTS (100x faster tree traversal than our Python)
- TPUs (specialized for neural networks)
- Distributed architecture (100s of workers)
- 40 days of training even with all that!

**For us**:
- Current architecture is reasonable for research project
- CUDA will give 5-8x speedup (good enough)
- Further optimization requires C++/Cython (major effort)
- Parallel workers give easy 4x speedup

## Verification: Check GPU Usage During Training

```bash
# On CUDA machine, run in separate terminal:
watch -n 1 nvidia-smi

# You'll see:
# During self-play: 5-20% GPU (MCTS is CPU-bound)
# During training: 95-99% GPU (NN training is GPU-bound)
```

The GPU IS being used heavily, just not during the self-play phase! 🎯
