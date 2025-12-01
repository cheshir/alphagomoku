# Distributed Training Quick Start

Complete setup guide for distributed AlphaGomoku training using Mac (self-play) + Colab (training).

## Architecture

```
Mac M1 Pro (Self-Play)  →  Redis Queue  →  Colab T4 (Training)
   6 CPU Workers            REDIS_DOMAIN      GPU Training
   120-180 games/hour                             600-1000 games/hour
```

## Quick Setup (5 Minutes)

### 1. Setup Redis (remote or locally)

### 2. Set Redis URL

**On Mac:**
```bash
# Add to ~/.zshrc or ~/.bashrc
export REDIS_URL='redis://:your_password@REDIS_DOMAIN:6379/0'
source ~/.zshrc
```

**On Colab:**
```python
import os
os.environ['REDIS_URL'] = 'redis://:your_password@REDIS_DOMAIN:6379/0'
```

### 3. Install Dependencies

```bash
# On both Mac and Colab
pip install redis>=5.0.0
```

### 4. Start Self-Play (Mac)

```bash
# Start 6 CPU workers
make distributed-selfplay-cpu-workers

# Or start 1 MPS worker
make distributed-selfplay-mps-worker
```

### 5. Start Training (Colab)

```bash
# In Colab notebook
!git clone https://github.com/your-repo/alphagomoku
%cd alphagomoku
!pip install -r requirements.txt
!pip install -e .

# Set Redis URL
import os
os.environ['REDIS_URL'] = 'redis://:password@REDIS_DOMAIN:6379/0'

# Start training
!make distributed-training-gpu
```

### 6. Monitor Progress

```bash
# On Mac (separate terminal)
make distributed-monitor
```

## Expected Performance

### Mac M1 Pro (Self-Play)
- **6 CPU workers:** 120-180 games/hour
- **CPU usage:** 60-80%
- **Memory:** 4-6 GB
- **Cost:** $0

### Colab T4 (Training)
- **Processing rate:** 600-1000 games/hour
- **GPU utilization:** 80-95%
- **VRAM:** 6-8 GB
- **Cost:** $0 (free tier) or ~$0.10/hour (Pro)

### Overall
- **Queue stays balanced** (training faster than generation)
- **Cost:** $0/month (with Colab free tier)
- **Games/day:** 2,880-4,320
- **Positions/day:** ~600,000-900,000

## Commands Reference

```bash
# Self-play workers
make distributed-selfplay-cpu-workers  # 6 CPU workers
make distributed-selfplay-mps-worker   # 1 MPS worker

# Training worker
make distributed-training-gpu          # GPU training

# Monitoring
make distributed-monitor               # Queue status
make distributed-help                  # Full help

# Manual commands (if Makefile doesn't work)
python scripts/distributed_selfplay_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset small \
    --mcts-simulations 50 \
    --device cpu \
    --worker-id mac-worker-1

python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset medium \
    --batch-size 1024 \
    --device cuda

python scripts/monitor_queue.py \
    --redis-url "$REDIS_URL"
```

## Troubleshooting

### Queue Growing Too Fast
```bash
# Check training worker is running
# Increase batch size: --batch-size 2048
# Use better GPU (RTX 4090 instead of T4)
```

### No Games in Queue
```bash
# Check self-play workers are running
# Check Redis connection: redis-cli -h REDIS_DOMAIN ping
# Verify REDIS_URL environment variable
```

### Workers Not Updating Model
```bash
# Check model publish frequency: --publish-frequency 5
# Check model update frequency: --model-update-frequency 10
# Monitor Redis Commander UI for model queue
```

## Advanced Usage

### Multiple Macs
```bash
# Mac 1
make distributed-selfplay-cpu-workers

# Mac 2
python scripts/distributed_selfplay_worker.py \
    --redis-url "$REDIS_URL" \
    --worker-id mac2-worker-1 \
    ...
```

### Better GPU
```bash
# RTX 4090 (3x faster than T4)
python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset large \
    --batch-size 2048 \
    --device cuda
```

## Resource Calculations

### Mac M1 Pro Capacity

| Workers | Games/Hour | CPU % | Memory | Recommended |
|---------|------------|-------|--------|-------------|
| 6 CPU   | 120-180    | 60-80 | 4-6 GB | ✓ Yes       |
| 8 CPU   | 160-240    | 80-95 | 6-8 GB | Possible    |
| 1 MPS   | 30-40      | 20-30 | 2-3 GB | Alternative |

### GPU Training Capacity

| GPU       | Games/Hour | Batch Size | VRAM   | Cost/Hour |
|-----------|------------|------------|--------|-----------|
| T4        | 600-1000   | 1024       | 6-8 GB | $0.10     |
| RTX 4090  | 1800-2400  | 2048       | 12 GB  | $0.70     |
| A100 40GB | 2400-3000  | 4096       | 20 GB  | $1.50     |

## Cost Breakdown

### Free Setup (Hobbyist)
- Self-play: $0 (local Mac)
- Redis: $0 (local Docker or remote VM free tier)
- Training: $0 (remote VM free tier, 12 hours/day)
- **Total: $0/month**

### Pro Setup (24/7)
- Self-play: $0 (local Mac)
- Redis: $7/month (Remote VM)
- Training: $72/month (Colab Pro, 24/7)
- **Total: $79/month**

## File Structure

```
alphagomoku/
├── .env.example                # Environment variables template
├── alphagomoku/
│   └── queue/
│       ├── __init__.py
│       └── redis_queue.py            # Redis queue wrapper
├── scripts/
│   ├── distributed_selfplay_worker.py    # Self-play worker
│   ├── distributed_training_worker.py    # Training worker
│   └── monitor_queue.py                  # Queue monitor
├── Makefile                          # Distributed training commands
└── docs/
    └── TRAINING.md                   # Full documentation
```

## Next Steps

1. **Deploy and test:** Follow Quick Setup above
2. **Monitor for 1 hour:** Check queue stays balanced
3. **Let it run:** Training will continue indefinitely
4. **Download checkpoints:** Every N iterations from training worker
5. **Evaluate models:** Test against baseline periodically

## Full Documentation

For complete details, see [docs/TRAINING.md#distributed-training](docs/TRAINING.md#distributed-training)

## Support

- Redis Commander UI: https://REDIS_DOMAIN
- Monitor script: `make distributed-monitor`
- Help command: `make distributed-help`
- Logs: Check worker output for errors
