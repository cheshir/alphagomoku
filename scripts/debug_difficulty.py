#!/usr/bin/env python3
"""Debug script to verify difficulty='easy' disables TSS."""

import torch
import numpy as np
from alphagomoku.model.network import GomokuNet
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.search.unified_search import UnifiedSearch
from alphagomoku.selfplay.selfplay import SelfPlayWorker

print("=" * 70)
print("Difficulty Debug Test")
print("=" * 70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Device: {device}")

# Create model
model = GomokuNet.from_preset('small', device=device)
print(f"✅ Model: {model.get_model_size():,} params on {next(model.parameters()).device}")

# Create environment
env = GomokuEnv(board_size=15)

print("\n" + "=" * 70)
print("Test 1: UnifiedSearch with difficulty='easy'")
print("=" * 70)

search_easy = UnifiedSearch(model, env, difficulty='easy')
print(f"✅ Created UnifiedSearch(difficulty='easy')")
print(f"   TSS searcher: {search_easy.tss_searcher}")
print(f"   TSS config enabled: {search_easy.tss_config.get('enabled', False)}")
print(f"   Endgame solver: {search_easy.endgame_solver}")
print(f"   Endgame config enabled: {search_easy.endgame_config.get('enabled', False)}")

if search_easy.tss_searcher is None:
    print("   ✅ TSS correctly disabled for difficulty='easy'")
else:
    print("   ❌ ERROR: TSS should be None but isn't!")

print("\n" + "=" * 70)
print("Test 2: UnifiedSearch with difficulty='medium'")
print("=" * 70)

search_medium = UnifiedSearch(model, env, difficulty='medium')
print(f"✅ Created UnifiedSearch(difficulty='medium')")
print(f"   TSS searcher: {search_medium.tss_searcher}")
print(f"   TSS config enabled: {search_medium.tss_config.get('enabled', False)}")
print(f"   Endgame solver: {search_medium.endgame_solver}")
print(f"   Endgame config enabled: {search_medium.endgame_config.get('enabled', False)}")

if search_medium.tss_searcher is not None:
    print("   ✅ TSS correctly enabled for difficulty='medium'")
else:
    print("   ❌ ERROR: TSS should be enabled but isn't!")

print("\n" + "=" * 70)
print("Test 3: SelfPlayWorker with difficulty='easy'")
print("=" * 70)

worker_easy = SelfPlayWorker(
    model=model,
    mcts_simulations=100,
    batch_size=64,
    difficulty='easy',
    epoch=0
)

print(f"✅ Created SelfPlayWorker(difficulty='easy', epoch=0)")
print(f"   Search TSS searcher: {worker_easy.search.tss_searcher}")
print(f"   Search TSS config enabled: {worker_easy.search.tss_config.get('enabled', False)}")

if worker_easy.search.tss_searcher is None:
    print("   ✅ TSS correctly disabled in SelfPlayWorker")
else:
    print("   ❌ ERROR: TSS should be disabled!")

print("\n" + "=" * 70)
print("Test 4: Actual search with difficulty='easy'")
print("=" * 70)

# Do a test search
state = env.reset()
print(f"✅ Created initial game state")

# Monkey-patch to track which search method is used
search_method_used = []

original_mcts_search = worker_easy.search.mcts.search
def tracked_mcts_search(*args, **kwargs):
    search_method_used.append('mcts')
    return original_mcts_search(*args, **kwargs)
worker_easy.search.mcts.search = tracked_mcts_search

if worker_easy.search.tss_searcher is not None:
    original_tss_search = worker_easy.search.tss_searcher.search
    def tracked_tss_search(*args, **kwargs):
        search_method_used.append('tss')
        return original_tss_search(*args, **kwargs)
    worker_easy.search.tss_searcher.search = tracked_tss_search

# Perform search
result = worker_easy.search.search(state, temperature=1.0)
print(f"✅ Search complete: method={result.search_method}")
print(f"   Methods called during search: {search_method_used}")

if 'tss' in search_method_used:
    print("   ❌ ERROR: TSS was called even though difficulty='easy'!")
elif 'mcts' in search_method_used:
    print("   ✅ Only MCTS was used (correct for difficulty='easy')")
else:
    print("   ⚠️  Neither TSS nor MCTS was called (unexpected)")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

if worker_easy.search.tss_searcher is None and 'tss' not in search_method_used:
    print("✅ ALL TESTS PASSED: difficulty='easy' correctly disables TSS")
    print("\nIf training is still slow, the problem is NOT TSS.")
    print("Possible causes:")
    print("1. MCTS batch size too small (GPU underutilized)")
    print("2. Single worker (no parallelism)")
    print("3. Model not actually on GPU in workers")
else:
    print("❌ TESTS FAILED: TSS is being used when it shouldn't be!")
    print("This explains the slow training.")

print("=" * 70)
