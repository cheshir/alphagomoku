# Model Auto-Update Mechanism

## Overview

The distributed selfplay manager automatically detects and downloads new models from Redis using an efficient timestamp-based polling system.

## How It Works

### 1. Timestamp-Based Polling (Efficient)

The model updater thread checks for new models every N batches (default: 10):

```python
# STEP 1: Check timestamp (lightweight, ~30 bytes)
current_timestamp = queue.get_model_timestamp()

# STEP 2: Compare timestamps
if current_timestamp == last_timestamp:
    continue  # Model hasn't changed, skip download

# STEP 3: Download model only if timestamp changed
model_data = queue.pull_model(timeout=0)
```

**Efficiency**:
- Checking timestamp: ~30 bytes transfer
- Downloading model: ~50 MB transfer
- **Saves ~1.6 million times bandwidth** when model unchanged!

### 2. Version Management

The system uses Redis UTC timestamp as the canonical version:

```
Training Worker publishes:
  iteration: 15
  timestamp: "2025-12-02T22:30:45.123456"

Selfplay Workers see:
  Model: 2025-12-02 22:30:45
```

**Benefits**:
- ✅ Consistent across all workers (Redis time, not local time)
- ✅ Easy to compare (timestamp string comparison)
- ✅ Human-readable in dashboard
- ✅ Timezone-independent (always UTC)

### 3. Update Flow

```
Training Worker                Redis                 Selfplay Manager
      |                         |                           |
      | 1. Train model         |                           |
      |----------------------->|                           |
      | push_model(state,      |                           |
      |   timestamp=UTC_NOW)   |                           |
      |                        |                           |
      |                        | 2. Poll every 10 batches |
      |                        |<--------------------------|
      |                        | get_model_timestamp()    |
      |                        |-------------------------->|
      |                        | "2025-12-02T22:30:45"    |
      |                        |                           |
      |                        | 3. Compare timestamps     |
      |                        |    (new != old?)          |
      |                        |                           |
      |                        | 4. Download if changed   |
      |                        |<--------------------------|
      |                        | pull_model()              |
      |                        |-------------------------->|
      |                        | {state, metadata}         |
      |                        |                           |
      |                        |    5. Save checkpoint     |
      |                        |    6. Update version      |
      |                        |    7. Workers reload      |
```

### 4. Worker Process Update

When a new model is detected, the manager:

1. Downloads model from Redis
2. Saves checkpoint to disk: `model_v{iteration}.pt`
3. Updates shared `current_model_version` value
4. Updates shared `current_model_timestamp` value
5. Workers check version on each game
6. If version changed, worker reloads from checkpoint

```python
# In worker process
current_version = current_version_value.value

if current_version > worker_model_version:
    checkpoint_path = os.path.join(checkpoint_dir, f'model_v{current_version}.pt')
    checkpoint = torch.load(checkpoint_path, ...)
    worker.model.load_state_dict(checkpoint['model_state'])
    worker_model_version = current_version
```

## Configuration

### Update Frequency

Control how often workers check for new models:

```bash
python scripts/distributed_selfplay_manager.py \
    --model-update-frequency 10  # Check every 10 batches (default)
```

**Recommendations**:
- **10 batches** (default): Good balance
  - ~10 minutes between checks (100 games @ 10 games/batch)
  - Workers stay reasonably up-to-date
  - Minimal overhead

- **5 batches** (more frequent):
  - ~5 minutes between checks
  - Workers stay very up-to-date
  - Slightly more overhead

- **20 batches** (less frequent):
  - ~20 minutes between checks
  - Less overhead
  - Workers may lag behind slightly

### Model Update Trigger

Workers check for model updates:
1. **After every N batches** (configurable via `--model-update-frequency`)
2. **On startup** (always loads latest from Redis)

## Verification

### Check Auto-Update is Working

1. **Start selfplay manager**:
```bash
make distributed-selfplay-workers
```

2. **Watch dashboard** - should show:
```
⏱  Runtime: 00:15:23  |  Model: 2025-12-02 22:30:45
```

3. **Publish new model from training worker**

4. **Wait ~10 minutes** (10 batches @ default frequency)

5. **Dashboard should update**:
```
⏱  Runtime: 00:25:45  |  Model: 2025-12-02 22:40:12
                                    ^^^^^^^^^^^^
                                    Updated!
```

6. **Check logs**:
```
[manager] ✓ Updated to model v16 (timestamp: 2025-12-02T22:40:12.345678)
```

### Troubleshooting

**Model not updating automatically:**

1. **Check update frequency**:
```bash
# Default is 10 batches (~10 minutes)
# Try more frequent:
--model-update-frequency 5
```

2. **Check Redis has new model**:
```bash
python scripts/monitor_queue.py --redis-url $REDIS_URL --once
# Should show: "Latest model: v16 (timestamp: 2025-12-02T22:40:12)"
```

3. **Check manager logs**:
```bash
# Should see periodic checks:
[manager] Checking for model updates...
[manager] Model timestamp: 2025-12-02T22:40:12
```

4. **Force immediate check**:
```bash
# Restart manager (will check on startup)
# Or reduce frequency to 1 batch temporarily
```

## Architecture Details

### Shared Memory

The manager uses multiprocessing shared values for coordination:

```python
self.current_model_version = self.manager.Value('i', 0)      # Integer: iteration
self.current_model_timestamp = self.manager.Value('c', b'')  # Bytes: timestamp string
self.version_lock = self.manager.Lock()                      # Thread-safe updates
```

**Why this works**:
- Workers run as separate processes (not threads)
- Each process has its own Python interpreter
- Shared values allow coordination without IPC overhead
- Lock ensures atomic updates

### Checkpoint Files

Models are saved as versioned checkpoints:

```
checkpoints/
├── model_v0.pt      # Random initialization
├── model_v15.pt     # Training iteration 15
├── model_v16.pt     # Training iteration 16
└── ...
```

Each checkpoint contains:
```python
{
    'model_state': state_dict,        # Model weights
    'iteration': 16,                  # Training iteration
    'timestamp': '2025-12-02T22:40:12.345678',  # Redis UTC timestamp
    'metadata': {                     # Training info
        'total_positions': 50000,
        'metrics': {...}
    }
}
```

### Redis Queue Integration

The system uses `RedisQueue` class methods:

```python
# Training worker publishes (in distributed_training_worker.py)
queue.push_model(
    model_state=model.state_dict(),
    metadata={'iteration': 16, 'total_positions': 50000}
)
# Automatically adds timestamp: datetime.utcnow().isoformat()

# Selfplay manager polls (in distributed_selfplay_manager.py)
timestamp = queue.get_model_timestamp()  # Lightweight check
if timestamp != last_timestamp:
    model_data = queue.pull_model()      # Download if changed
```

## Performance Impact

### Network Usage

**Without timestamp polling** (naive approach):
- Check every 10 batches: 10 minutes × 6 times/hour = 60 checks/hour
- Model size: 50 MB
- Bandwidth: 60 × 50 MB = **3 GB/hour**

**With timestamp polling** (current implementation):
- Timestamp checks: 60 × 30 bytes = 1.8 KB/hour
- Model downloads: ~1/hour × 50 MB = 50 MB/hour
- Bandwidth: **~50 MB/hour** (60x reduction!)

### Latency

- Timestamp check: ~10 ms
- Model download: ~500-1000 ms (50 MB @ 50-100 MB/s)
- Worker reload: ~100-200 ms (load checkpoint, update model)
- **Total**: ~1 second (negligible compared to 10-minute intervals)

### CPU/Memory

- Minimal CPU overhead (timestamp string comparison)
- Memory: 14 MB per worker for model copy
- No memory pressure from frequent checks

## Best Practices

1. **Keep default frequency** (10 batches): Good balance for most use cases

2. **Monitor dashboard**: Verify model timestamp updates periodically

3. **Check Redis regularly**: Ensure training worker is publishing models
   ```bash
   python scripts/monitor_queue.py --redis-url $REDIS_URL --once
   ```

4. **Use consistent timezone**: All timestamps are UTC (automatic)

5. **Don't restart workers unnecessarily**: They auto-update automatically

6. **Verify after training**: After training publishes, wait ~10 minutes and check dashboard

## Example Session

```bash
# Terminal 1: Start selfplay workers
$ make distributed-selfplay-workers
[manager] ✓ Loaded model from Redis (v10, 2025-12-02T22:00:00)
[manager] Starting 8 worker processes...
[dashboard] Model: 2025-12-02 22:00:00

# Terminal 2: Start training worker (publishes new model)
$ make distributed-training-gpu
[training] Training iteration 11 complete
[training] Publishing updated model to queue...
[training] ✓ Model published (iteration 11, timestamp: 2025-12-02T22:10:15)

# Terminal 1: After ~10 minutes
[manager] Checking for model updates...
[manager] New timestamp detected: 2025-12-02T22:10:15
[manager] Downloading model...
[manager] ✓ Updated to model v11 (timestamp: 2025-12-02T22:10:15)
[worker-1] Updated to v11
[worker-2] Updated to v11
...
[dashboard] Model: 2025-12-02 22:10:15  # Updated!

# Terminal 2: Training continues
[training] Training iteration 12 complete
[training] ✓ Model published (iteration 12, timestamp: 2025-12-02T22:20:30)

# Terminal 1: After another ~10 minutes
[manager] ✓ Updated to model v12 (timestamp: 2025-12-02T22:20:30)
[dashboard] Model: 2025-12-02 22:20:30  # Updated again!
```

The system works seamlessly in the background, keeping all workers synchronized with the latest trained model!
