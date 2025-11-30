#!/usr/bin/env python3
"""Quick test: Is model on CUDA during self-play?"""

import torch
import time
from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayWorker

print("=" * 70)
print("CUDA Self-Play Test")
print("=" * 70)

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available!")
    exit(1)

print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")

# Create model on CUDA
model = GomokuNet.from_preset('small', device='cuda')
print(f"✅ Model created: {model.get_model_size():,} params")
print(f"✅ Model device: {next(model.parameters()).device}")

# Initial GPU memory
initial_mem = torch.cuda.memory_allocated() / 1024**3
print(f"✅ Initial GPU memory: {initial_mem:.3f} GB")

# Create worker with difficulty='easy' (pure MCTS, no TSS)
print("\n🎮 Creating SelfPlayWorker with difficulty='easy'...")
worker = SelfPlayWorker(
    model=model,
    mcts_simulations=100,
    batch_size=64,
    difficulty='easy',  # Pure MCTS
    epoch=0
)

# Check model device after worker creation
print(f"✅ Worker model device: {next(worker.search.mcts.model.parameters()).device}")

# Generate one game
print("\n🎲 Generating 1 self-play game...")
print("   Run 'watch -n 0.5 nvidia-smi' to see GPU usage!")
print("   You should see:")
print("   - GPU memory increase to ~1-2 GB")
print("   - GPU utilization spike to 50-100%")
print()

start = time.time()
data = worker.generate_game()
elapsed = time.time() - start

# Check GPU memory after game
final_mem = torch.cuda.memory_allocated() / 1024**3
print(f"\n✅ Game complete: {len(data)} positions in {elapsed:.1f}s")
print(f"✅ Final GPU memory: {final_mem:.3f} GB")
print(f"✅ Memory used during game: {final_mem - initial_mem:.3f} GB")

# Test batched inference directly
print("\n🧪 Testing direct batched inference on GPU...")
test_states = torch.randn(64, 5, 15, 15).cuda()  # Batch of 64 states
print(f"✅ Created test batch on: {test_states.device}")

start = time.time()
with torch.no_grad():
    policy, value = model(test_states)
elapsed = time.time() - start

print(f"✅ Batch inference: {elapsed*1000:.1f}ms for 64 positions")
print(f"✅ Output device: {policy.device}")
print(f"✅ GPU memory: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")

print("\n" + "=" * 70)
print("If GPU memory increased and inference was fast (< 100ms),")
print("then GPU IS working. The 0% utilization might be because:")
print("1. MCTS spends time on CPU tree traversal between batches")
print("2. Batch size is too small")
print("3. Need to increase parallel workers")
print("=" * 70)
