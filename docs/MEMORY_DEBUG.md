# Memory Debugging Guide

## Memory Logging Added

### 1. Always-On Logs (Every Epoch)
These logs are shown by default in `make train`:

```
💾 MPS before cleanup: allocated=X.XX GB, driver=X.XX GB
✨ MPS after cleanup: allocated=X.XX GB, driver=X.XX GB
📉 Freed: X.XX GB
```

**What to look for:**
- `allocated` should be < 1 GB after selfplay
- `freed` should show significant memory returned
- If `allocated` stays > 2 GB, there's a leak

### 2. Debug Mode Logs
Run with `make train-debug-memory` for detailed tracking:

```
🔍 Epoch N Memory Tracking:
   [Epoch start] MPS allocated: X.XX GB, driver: X.XX GB
   [Epoch start] Process RSS: X.XX GB
   [Before selfplay] MPS allocated: X.XX GB, driver: X.XX GB
   [After selfplay] MPS allocated: X.XX GB, driver: X.XX GB
   [Before training] MPS allocated: X.XX GB, driver: X.XX GB
   [After training] MPS allocated: X.XX GB, driver: X.XX GB
```

**What to look for:**
- Gradual increase across epochs = leak
- Sudden spike = specific phase issue
- Process RSS >> MPS allocated = CPU memory issue

## Expected Memory Profile (Healthy)

```
Epoch 0:
   [Epoch start] MPS: 0.08 GB
   [Before selfplay] MPS: 0.08 GB
   [After selfplay] MPS: 0.60 GB    # Workers used MPS
   💾 Before cleanup: 0.60 GB
   ✨ After cleanup: 0.08 GB         # ✓ Memory freed!
   [Before training] MPS: 0.08 GB
   [After training] MPS: 0.45 GB     # Training peak
   
Epoch 1:
   [Epoch start] MPS: 0.45 GB        # Some training memory lingers
   ... (similar pattern)
```

## Troubleshooting

### Issue: MPS allocated > 10 GB before cleanup
**Cause:** MCTS memory leak (tensors not moved to CPU)
**Fix:** Already implemented in mcts.py:472-479

### Issue: MPS memory not freed after cleanup
**Cause:** Worker processes not terminating
**Fix:** Already implemented in parallel.py:212-222

### Issue: Process RSS grows but MPS stable
**Cause:** CPU memory leak (numpy arrays, MCTS tree nodes)
**Solution:** Add periodic tree cleanup in selfplay

### Issue: Gradual increase across epochs
**Cause:** Model checkpoint accumulation or buffer growth
**Check:** `ls -lh checkpoints/` and buffer size logs

## Quick Test

Run just 1 epoch with memory debugging:
```bash
python scripts/train.py --epochs 1 --selfplay-games 10 --debug-memory --resume auto
```

Should complete with MPS < 1 GB at end.
