# Gradient Checkpointing Explained

## What is it?

**Gradient checkpointing** is a memory optimization technique that trades compute time for memory.

## How PyTorch Normally Works

### Forward Pass (Computing Output):
```
Input (batch=1024)
  ↓
Block1: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ALL 6 intermediate activations in GPU memory]
Block2: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ALL 6 intermediate activations in GPU memory]
...
Block30: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ALL 6 intermediate activations in GPU memory]
Output
```

**Total saved:** 30 blocks × 6 layers = **180 activation maps** in memory!

### Backward Pass (Computing Gradients):
```
Uses saved activations to compute gradients quickly
No need to recompute anything
```

**Memory:** High (29.66 GB with batch=1024)  
**Speed:** Fast (single forward pass)

---

## With Gradient Checkpointing

### Forward Pass:
```
Input (batch=1024)
  ↓
Block1: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ONLY final block output, discards intermediates]
Block2: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ONLY final block output, discards intermediates]
...
Block30: Conv → BN → ReLU → Conv → BN → SE → Add → ReLU
  ↓ [SAVES ONLY final block output, discards intermediates]
Output
```

**Total saved:** 30 blocks × 1 output = **30 activation maps** in memory

### Backward Pass:
```
For Block30:
  1. Re-run forward pass for Block30 → Get intermediate activations
  2. Compute gradients using those activations
  3. Discard activations

For Block29:
  1. Re-run forward pass for Block29 → Get intermediate activations
  2. Compute gradients using those activations
  3. Discard activations

... (repeat for all 30 blocks)
```

**Memory:** Low (4.94 GB with batch=1024)  
**Speed:** Slower (~30% overhead from re-computing)

---

## The Math

### Your Model (30 blocks, 192 channels, batch=1024):

| Mode | Activation Maps Stored | Memory | Speed |
|------|----------------------|---------|-------|
| Normal | 180 (6 per block) | 29.66 GB | 100% |
| Checkpointing | 30 (1 per block) | 4.94 GB | ~70% |

**Memory reduction:** 83% (24.72 GB saved!)

---

## Code Implementation

```python
# Enable checkpointing
model = GomokuNet(board_size=15, num_blocks=30, channels=192, 
                  use_checkpoint=True)  # ← Add this flag

# What happens internally:
def forward(self, x):
    x = self.input_conv(x)
    
    if self.use_checkpoint and self.training:
        # Checkpointed mode
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            # Only saves block output, not intermediates
            x = checkpoint(block, x, use_reentrant=False)
    else:
        # Normal mode (saves all activations)
        for block in self.blocks:
            x = block(x)
    
    # Rest of forward pass...
```

**Key Details:**
- `use_reentrant=False` - Uses safer, newer checkpointing API
- Only active during training (`self.training`)
- Inference unaffected (no backward pass needed)

---

## When Should You Use It?

### ✅ USE Checkpointing When:

1. **GPU memory is limited**
   - MPS with 18 GB → Can't fit batch=1024 without it
   - Want to train but hitting OOM errors

2. **Want larger batch sizes**
   - Better gradient estimates
   - More stable training
   - Better convergence

3. **Training time not critical**
   - 30% slower is acceptable
   - Still faster than swapping to disk!

### ❌ DON'T Use Checkpointing When:

1. **Plenty of GPU memory**
   - A100 with 40 GB → No need, memory not bottleneck
   - Already using small batch size (256)

2. **Speed is critical**
   - Need fastest possible training
   - Willing to reduce batch size instead

3. **Model is small**
   - 12 blocks instead of 30
   - Already fits in memory

---

## Practical Recommendations

### For Your Situation:

**M1 Pro (16 GB RAM, 18 GB MPS):**
```bash
# Currently: batch=512, no checkpointing → 39h/epoch (swapping!)
# Option A: Enable checkpointing
model = GomokuNet(..., use_checkpoint=True)
--batch-size 1024  # 2x larger batch
# Result: ~25h/epoch (slower per batch, but fewer batches)
```

**A100 (40 GB GPU):**
```bash
# Don't use checkpointing!
model = GomokuNet(..., use_checkpoint=False)  # Default
--batch-size 2048  # Large batch, no memory issues
# Result: ~2.5h/epoch
```

---

## Real-World Impact

| Configuration | Batch | Memory | Time/Epoch | Notes |
|--------------|-------|---------|-----------|-------|
| M1 (no checkpoint) | 512 | 14.9 GB | 39h | Swapping! |
| M1 (checkpoint) | 1024 | 11.0 GB | ~25h | No swap, fewer batches |
| A100 (no checkpoint) | 2048 | 25 GB | 2.5h | Optimal |
| A100 (checkpoint) | 4096 | 18 GB | 3.5h | Not worth it |

**Conclusion:** Checkpointing is a **lifesaver for limited memory**, but **unnecessary with ample GPU memory**.

---

## How to Enable

### Option 1: Modify train.py
```python
# Line ~275 in scripts/train.py
model = GomokuNet(board_size=15, num_blocks=30, channels=192,
                  use_checkpoint=True)  # ← Add this
```

### Option 2: Add command-line flag (future improvement)
```bash
python scripts/train.py --use-checkpoint --batch-size 1024
```

### Option 3: Use in Makefile
```makefile
train-mps-checkpoint:
    # Edit train.py first to enable checkpointing
    --batch-size 1024
```

---

## Technical Details

**What PyTorch does:**

1. **Forward pass:** Only saves minimal checkpoints
2. **Backward pass:** For each checkpoint segment:
   - Temporarily re-enable gradient tracking
   - Re-run forward pass for that segment
   - Compute gradients
   - Discard intermediate activations
   - Continue to next segment

**Why it works:**
- Re-computing is cheap (modern GPUs are fast)
- Memory bandwidth is expensive (loading/storing 30 GB)
- Net benefit: 83% less memory for 30% time cost

**Alternatives:**
- Gradient accumulation (multiple small batches)
- Mixed precision training (FP16 uses half memory)
- Smaller model (fewer blocks/channels)
