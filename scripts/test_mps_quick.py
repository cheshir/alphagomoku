#!/usr/bin/env python3
"""Quick test to verify MPS is used in parallel workers"""

import torch
import time
from alphagomoku.model.network import GomokuNet
from alphagomoku.train.trainer import Trainer
from alphagomoku.selfplay.parallel import ParallelSelfPlay

print("=" * 70)
print("Quick MPS Verification Test")
print("=" * 70)

# Create model and trainer (moves to MPS)
print("\n1. Creating model...")
model = GomokuNet(board_size=15, num_blocks=30, channels=192)
trainer = Trainer(model, lr=1e-3)
print(f"   Main process device: {next(model.parameters()).device}")

# Create parallel worker with 2 workers
print("\n2. Creating ParallelSelfPlay with 2 workers...")
print("   Look for '[Worker subprocess]' messages below:\n")
print("-" * 70)

parallel_worker = ParallelSelfPlay(
    model=model,
    board_size=15,
    mcts_simulations=50,  # Reduced for faster test
    adaptive_sims=False,
    batch_size=96,
    num_workers=2,
    difficulty='medium'
)

print("-" * 70)
print("\n3. Generating 4 games (2 per worker)...")
print("-" * 70)

start = time.time()
data = parallel_worker.generate_batch(4, debug=False)
elapsed = time.time() - start

print("-" * 70)
print(f"\n4. ✅ Test completed successfully!")
print(f"   Time: {elapsed:.1f}s")
print(f"   Positions: {len(data)}")
print(f"   Speed: {4 / (elapsed / 3600):.0f} games/hour")
print(f"\n   If you saw '[Worker subprocess] Model loaded on device: mps:0' above,")
print(f"   then MPS is working correctly in parallel workers!")
