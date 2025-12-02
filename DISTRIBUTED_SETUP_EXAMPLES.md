# Distributed Training Setup Examples

Real-world examples and commands for AlphaGomoku distributed training.

## Example 1: Basic Setup (Mac + Colab)

### Scenario
- Mac M1 Pro for self-play
- Google Colab T4 for training
- Redis queue

### Step-by-Step

**1. Deploy Redis**
```bash
# On your server
git clone https://github.com/your-repo/alphagomoku
cd alphagomoku

# Copy environment file
cp .env.example .env

# Edit .env (set passwords)
nano .env
```

**2. Configure Mac**
```bash
# Install dependencies
pip install redis>=5.0.0

# Set environment variable
echo 'export REDIS_URL="redis://:YOUR_PASSWORD@REDIS_DOMAIN:6379/0"' >> ~/.zshrc
source ~/.zshrc

# Verify connection
python -c "import redis; r = redis.from_url('$REDIS_URL'); print('Connected!' if r.ping() else 'Failed')"
```

**3. Start Self-Play on Mac**
```bash
# Terminal 1: Start workers
make distributed-selfplay-cpu-workers

# Expected output:
# [mac-cpu-worker-1] Starting...
# [mac-cpu-worker-2] Starting...
# ...
# [mac-cpu-worker-1] Generating 10 games (batch 1)...
```

**4. Monitor Queue on Mac**
```bash
# Terminal 2: Monitor
make distributed-monitor

# Expected output:
# Queue Status: 15 batches
# Active Workers: 6 self-play, 0 training
# Games Pushed: 150 (180/hour)
```

**5. Start Training on Colab**
```python
# Colab cell 1: Setup
!git clone https://github.com/your-repo/alphagomoku
%cd alphagomoku
!pip install -r requirements.txt
!pip install -e .

# Colab cell 2: Start training
import os
os.environ['REDIS_URL'] = 'redis://:YOUR_PASSWORD@REDIS_DOMAIN:6379/0'

!python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset medium \
    --batch-size 1024 \
    --device cuda

# Expected output:
# [training-worker] Connected to Redis
# [training-worker] Pulling 50 game batches...
# [training-worker] Training on 10,000 positions...
# [training-worker] Iteration 1 complete: loss=0.4523
```

## Example 2: Multiple Macs

### Scenario
- 2 Mac M1 Pro machines
- Total: 12 CPU workers
- Google Colab T4 for training

### Setup

**Mac 1:**
```bash
# Start 6 workers
export REDIS_URL="redis://:PASSWORD@REDIS_DOMAIN:6379/0"
make distributed-selfplay-cpu-workers
```

**Mac 2:**
```bash
# Start 6 workers with different IDs
export REDIS_URL="redis://:PASSWORD@REDIS_DOMAIN:6379/0"

for i in 7 8 9 10 11 12; do
    python scripts/distributed_selfplay_worker.py \
        --redis-url "$REDIS_URL" \
        --model-preset small \
        --mcts-simulations 50 \
        --device cpu \
        --worker-id "mac2-cpu-worker-$i" \
        --games-per-batch 10 &
done
wait
```

**Expected Throughput:**
- 12 workers × 25 games/hour = 300 games/hour
- Need better GPU (RTX 4090) for training

## Example 3: Mac MPS + Colab T4

### Scenario
- Mac with Apple Silicon (MPS)
- Use MPS for faster self-play
- Colab T4 for training

### Setup

**Mac:**
```bash
export REDIS_URL="redis://:PASSWORD@REDIS_DOMAIN:6379/0"

# Start MPS worker (faster than CPU)
make distributed-selfplay-mps-worker

# Expected: 30-40 games/hour
```

**Colab:**
```python
# Same as Example 1
```

## Example 4: Local Redis (No Cloud)

### Scenario
- Run Redis locally on Mac
- No external dependencies
- Mac for self-play
- Colab for training (connects to Mac's Redis)

### Setup

**Mac: Start Redis**
```bash
# Install Redis
brew install redis

# Start Redis with password
redis-server --requirepass YOUR_PASSWORD --port 6379 &

# Verify
redis-cli -a YOUR_PASSWORD ping  # Should return PONG

# Get your IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Expose port (if needed for Colab)
# Use ngrok or cloudflared tunnel
ngrok tcp 6379
```

**Mac: Start Self-Play**
```bash
export REDIS_URL="redis://:YOUR_PASSWORD@localhost:6379/0"
make distributed-selfplay-cpu-workers
```

**Colab: Connect to Mac's Redis**
```python
# Use ngrok URL
import os
os.environ['REDIS_URL'] = 'redis://:YOUR_PASSWORD@0.tcp.ngrok.io:12345/0'

!python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset medium \
    --batch-size 1024 \
    --device cuda
```

## Example 5: High-Performance Setup

### Scenario
- Mac M1 Max (10-core CPU)
- RTX 4090 for training (local or cloud)
- Maximum throughput

### Setup

**Mac: 8 CPU Workers**
```bash
export REDIS_URL="redis://:PASSWORD@REDIS_DOMAIN:6379/0"

# Start 8 workers (for 10-core CPU)
for i in 1 2 3 4 5 6 7 8; do
    python scripts/distributed_selfplay_worker.py \
        --redis-url "$REDIS_URL" \
        --model-preset small \
        --mcts-simulations 50 \
        --device cpu \
        --worker-id "mac-cpu-worker-$i" \
        --games-per-batch 10 &
done
wait

# Expected: 200-240 games/hour
```

**RTX 4090: Training**
```bash
export REDIS_URL="redis://:PASSWORD@REDIS_DOMAIN:6379/0"

python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset large \
    --batch-size 2048 \
    --device cuda \
    --publish-frequency 5

# Expected: 1800-2400 games/hour processing
```

## Example 6: Monitoring and Debugging

### Monitor Queue
```bash
# Terminal 1: Queue monitor
make distributed-monitor

# Terminal 2: Redis Commander (web UI)
open https://REDIS_DOMAIN

# Terminal 3: Worker logs
tail -f worker.log
```

### Check Queue Stats Programmatically
```python
from alphagomoku.queue import RedisQueue
import os

queue = RedisQueue(os.environ['REDIS_URL'])
stats = queue.get_stats()
print(f"Queue size: {stats['queue_size']}")
print(f"Positions pushed: {stats['games_pushed']}")  # Note: counts positions, not games
print(f"Positions pulled: {stats['games_pulled']}")  # Note: counts positions, not games

workers = queue.get_active_workers()
print(f"Self-play workers: {workers['selfplay']}")
print(f"Training workers: {workers['training']}")
```

### Clear Queue (if needed)
```python
from alphagomoku.queue import RedisQueue
import os

queue = RedisQueue(os.environ['REDIS_URL'])

# WARNING: This deletes all data!
queue.clear_queue()  # Clear all queues
queue.reset_stats()  # Reset statistics
```

## Example 7: Resume Training

### Scenario
- Training worker crashed
- Resume from checkpoint
- Don't lose progress

### Setup

**Training Worker:**
```bash
# Check checkpoints
ls -lh checkpoints_distributed/

# Checkpoint files:
# model_iteration_5.pt
# model_iteration_10.pt
# ...

# Training worker auto-loads latest model from Redis queue
# No special resume needed - just restart

python scripts/distributed_training_worker.py \
    --redis-url "$REDIS_URL" \
    --model-preset medium \
    --batch-size 1024 \
    --device cuda

# Will continue from where it left off
```

## Resource Calculations

### Mac M1 Pro (6 workers)
```
CPU Usage: 60-80%
Memory: 4-6 GB
Network: ~10 MB/hour upload (game data)
Cost: $0

Games/hour: 120-180
Games/day: 2,880-4,320
```

### Colab T4 (training)
```
GPU Utilization: 80-95%
VRAM: 6-8 GB
Network: ~10 MB/hour download (game data)
Cost: $0 (free tier) or $0.10/hour (Pro)

Processing: 600-1000 games/hour
Can handle 6-8 self-play workers
```

### Network Requirements
```
Game data size: ~1-2 KB per game
Model size: ~15 MB per model

Bandwidth:
- Upload (self-play): ~200 KB/hour (negligible)
- Download (training): ~200 KB/hour (negligible)
- Model updates: ~15 MB every 5 batches (negligible)

Total: < 1 GB/month
```

## Troubleshooting Examples

### Issue: Queue growing too fast

**Diagnosis:**
```bash
make distributed-monitor

# Output shows:
# Queue size: 150 batches (growing)
# Self-play workers: 6
# Training workers: 0  ← Problem!
```

**Solution:**
```bash
# Training worker is not running
# Start it on Colab
```

### Issue: No games in queue

**Diagnosis:**
```bash
make distributed-monitor

# Output shows:
# Queue size: 0 batches
# Self-play workers: 0  ← Problem!
# Training workers: 1
```

**Solution:**
```bash
# Self-play workers crashed or not started
# Check logs
tail -f ~/alphagomoku-worker.log

# Restart workers
make distributed-selfplay-cpu-workers
```

### Issue: Workers not updating model

**Diagnosis:**
```python
from alphagomoku.queue import RedisQueue
queue = RedisQueue(os.environ['REDIS_URL'])
stats = queue.get_stats()
print(f"Models pushed: {stats['models_pushed']}")  # Should increase
print(f"Models pulled: {stats['models_pulled']}")  # Should increase
```

**Solution:**
```bash
# Check training worker is publishing
# --publish-frequency 5 (default)

# Check self-play workers are fetching
# --model-update-frequency 10 (default)
```

## Best Practices

1. **Monitor first hour:** Watch queue size to ensure balance
2. **Start small:** Begin with 2-3 workers, scale up
3. **Log everything:** Save worker logs for debugging
4. **Download checkpoints:** Regularly download from training worker
5. **Use free tier wisely:** Colab free tier is sufficient for hobbyists
6. **Local Redis first:** Test with local Redis before deploying to cloud

## Cost Optimization

### Free Setup (Recommended for Hobbyists)
```
Mac M1 Pro: $0 (you own it)
Redis: $0 (local Docker)
Colab: $0 (free tier, 12 hours/day)
Total: $0/month
```

### Cloud Setup (Continuous Training)
```
Mac M1 Pro: $0 (you own it)
Redis: $7/month (Remote VM)
Colab Pro: $10/month (100 compute units)
Total: $17/month

Expected results:
- ~50,000 games/week
- ~10M positions/week
- Elo ~1700-1800 after 2-3 weeks
```

### High-Performance Setup
```
Mac M1 Max: $0 (you own it)
Redis: $7/month (Remote VM)
RTX 4090 cloud: $500/month (24/7 at $0.70/hour)
Total: $507/month

Expected results:
- ~300,000 games/week
- ~60M positions/week
- Elo ~1900+ after 2-3 weeks
```

## Next Steps

1. Start with Example 1 (Basic Setup)
2. Monitor for 1 hour
3. Let it run for 1 week
4. Evaluate model strength
5. Scale up if needed

## Support

- Full docs: [docs/TRAINING.md](docs/TRAINING.md)
- Quick start: [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md)
- Help: `make distributed-help`
- Monitor: `make distributed-monitor`
