# Training AlphaGomoku on Google Colab

**Note**: After refactoring, we now use **model presets** for easy configuration!

## Quick Start

1. **Upload `train_colab.ipynb` or `train_universal.ipynb` to Google Colab**
   - Go to https://colab.research.google.com
   - File → Upload notebook → Select notebook

2. **Enable GPU** ⚠️ IMPORTANT
   - Runtime → Change runtime type → Hardware accelerator → GPU
   - Select T4, V100, or A100 (A100 recommended with Colab Pro)
   - Click **Save** to apply changes

3. **Verify GPU is working** (highly recommended)
   ```bash
   python scripts/check_device.py
   ```
   You should see: `✅ CUDA is available!`

4. **Run all cells**
   - Runtime → Run all
   - Follow prompts to mount Google Drive

5. **Monitor training**
   - Checkpoints saved to Google Drive every epoch
   - Evaluation runs automatically every 5-10 epochs
   - View Elo ratings in `checkpoints/elo_history.json`

## Model Presets

Choose the right model for your hardware:

| Preset | Parameters | Description | Recommended For |
|--------|-----------|-------------|-----------------|
| **small** | 1.2M | Fast iteration, 80% strength | T4, RTX 3060, Development |
| **medium** | 3M | Balanced, 90% strength | V100, RTX 3080+, Production |
| **large** | 5M | Maximum strength, slow | A100, Research |

## Hardware Auto-Configuration

The training script automatically detects your hardware and optimizes settings:

### GPU Detection

| GPU | Memory | Model Preset | Batch Size | Expected Time/Epoch |
|-----|--------|--------------|-----------|-------------------|
| **A100** | 40+ GB | medium | 1024 | ~20-30 min |
| **V100** | 24-40 GB | small | 512 | ~15-25 min |
| **T4** | 16 GB | small | 256 | ~25-35 min |
| **RTX 3060** | <16 GB | small | 128 | ~30-40 min |

### CPU Fallback

| Config | Model | Batch Size | Use Case |
|--------|-------|------------|----------|
| **CPU** | small | 128 | Testing only (very slow) |

## What Happens Automatically

1. **Device Detection**
   ```
   🚀 Auto-detected device: CUDA (GPU: Tesla A100-SXM4-40GB)
   ⚙️ Hardware Configuration: Tesla A100 (40GB) - High-end GPU
   ```

2. **Batch Size Configuration**
   ```
   Auto-configured batch size: 2048
   ```

3. **Model Configuration**
   ```
   🧠 Model Configuration:
      Parameters: 5,000,000
      Blocks: 30, Channels: 192
      Gradient checkpointing: ✗ Disabled
   ```

4. **Memory Optimization** (if needed)
   ```
   ⚙️ Hardware Configuration: Apple Silicon with 16GB RAM - Reduced model to avoid swapping
   Auto-configured batch size: 256
   🧠 Model Configuration:
      Parameters: 3,000,000
      Blocks: 20, Channels: 128
      Gradient checkpointing: ✓ Enabled
   ```

## Training on Different Platforms

### Google Colab (Recommended for Cloud Training)

**Free Tier:**
- GPU: T4 (16 GB)
- Session: 12 hours
- Cost: Free
- Training time: ~6h/epoch

**Colab Pro ($10/month):**
- GPU: V100 or A100
- Session: 24 hours
- Cost: $10/month
- Training time: 2.5-4h/epoch

**Colab Pro+ ($50/month):**
- GPU: A100 (40 GB)
- Session: Longer (priority)
- Cost: $50/month
- Training time: ~2.5h/epoch

### Kaggle Notebooks

**Free Tier:**
- GPU: P100 (16 GB) or T4
- Session: 12 hours/week (30 hours/week GPU quota)
- Cost: Free
- Similar to Colab free tier

### AWS/GCP/Azure

**Manual Setup Required:**
1. Upload code to VM
2. Install dependencies
3. Run training script
4. Auto-configuration still works!

**Recommended Instances:**
- AWS: p4d.24xlarge (8× A100)
- GCP: a2-highgpu-1g (1× A100)
- Azure: NC24ads_A100_v4 (1× A100)

## Resuming Training

Training automatically resumes from the latest checkpoint:

```python
!python scripts/train.py --resume auto ...
```

If Colab disconnects:
1. Reconnect to runtime
2. Re-run all cells
3. Training resumes automatically from last epoch

## Monitoring Progress

### Real-time Metrics

The notebook includes a monitoring cell that shows:
- Training loss
- Policy accuracy
- Value MAE
- Learning rate

### Saved Files

All files saved to Google Drive:
```
/content/drive/MyDrive/alphagomoku/
├── checkpoints/
│   ├── model_epoch_0.pt
│   ├── model_epoch_1.pt
│   ├── ...
│   ├── model_final.pt
│   └── training_metrics.csv
└── data/
    └── replay_buffer/
```

## Cost Estimation

### Google Colab Pro (Recommended)

**Setup:**
- $10/month subscription
- A100 GPU when available
- 24-hour sessions

**Training:**
- 200 epochs × 2.5h = 500 hours
- 500h ÷ 24h/session = ~21 sessions
- ~21 days of training (with reconnecting)

**Total Cost:** $10-20 (1-2 months)

### AWS EC2 (p4d.24xlarge)

**Setup:**
- On-demand: $32.77/hour
- Spot: ~$9.87/hour (70% discount)

**Training:**
- 200 epochs × 2.5h = 500 hours
- Cost: 500h × $9.87 = **$4,935** (spot)

**Total Cost:** Much more expensive than Colab!

### Recommendation

**For most users:** Google Colab Pro ($10/month) is the best value.

## Tips for Success

1. **Use Colab Pro**: A100 is 2-3× faster than T4
2. **Monitor sessions**: Colab disconnects after 12-24 hours
3. **Save checkpoints**: Every epoch saved to Google Drive
4. **Download important models**: Save key checkpoints locally
5. **Use spot instances carefully**: Can be interrupted on AWS/GCP

## Troubleshooting

### "MPS not available, using CPU" on Google Colab

**Problem:** Training says it's using CPU instead of GPU

**Solution:**
1. **Check GPU is enabled:**
   - Runtime → Change runtime type → Hardware accelerator → GPU
   - Click **Save** (very important!)
   - Restart runtime

2. **Verify CUDA is available:**
   ```bash
   python scripts/check_device.py
   ```
   You should see `✅ CUDA is available!`

3. **If CUDA still not detected:**
   ```bash
   # Reinstall PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify installation:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should print True
   print(torch.cuda.get_device_name(0))  # Should print GPU name
   ```

### Out of Memory (even with auto-config)

Reduce batch size manually:
```python
!python scripts/train.py --batch-size 256 ...  # or even 128
```

### Slow training on T4

Expected! T4 is 2-3× slower than A100. Consider:
- Use small model: `--model-preset small`
- Reduce epochs: `--epochs 100`
- Reduce games: `--selfplay-games 100`
- Upgrade to Colab Pro for A100 access

### Colab disconnects frequently

- Use Colab Pro for longer sessions (24+ hours)
- Keep browser tab open
- Use https://colab.research.google.com/drive (Drive integration)
- Training auto-resumes from checkpoints with `--resume auto`

### Can't mount Google Drive

- Check permissions
- Try different browser
- Clear cookies and retry
- Ensure Google Drive has enough space (need ~1-5GB)

## Advanced: Multi-GPU Training

For AWS/GCP with multiple GPUs, the script currently uses single GPU.
Multi-GPU support can be added with:
- `torch.nn.DataParallel`
- `torch.distributed`

(Not implemented yet, but can be added if needed)
