#!/usr/bin/env python3
"""Example of TSS integration with MCTS."""

import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.tss import Position, tss_search


class TSSGuidedMCTS:
    """MCTS with TSS tactical guidance."""
    
    def __init__(self, model, env, mcts_simulations=400, tss_depth=4, tss_time_cap=100):
        self.mcts = MCTS(model, env, num_simulations=mcts_simulations)
        self.tss_depth = tss_depth
        self.tss_time_cap = tss_time_cap
        self.env = env
    
    def search(self, state: np.ndarray, temperature: float = 1.0) -> tuple:
        """Search with TSS tactical override."""
        # Convert to TSS position
        position = Position(
            board=state,
            current_player=self.env.current_player,
            last_move=tuple(self.env.last_move) if self.env.last_move[0] >= 0 else None
        )
        
        # Run TSS first
        tss_result = tss_search(position, self.tss_depth, self.tss_time_cap)
        
        # If TSS finds forced move, use it
        if tss_result.forced_move is not None:
            action_probs = np.zeros(self.env.board_size * self.env.board_size)
            r, c = tss_result.forced_move
            action = r * self.env.board_size + c
            action_probs[action] = 1.0
            
            print(f"TSS Override: {'Forced Win' if tss_result.is_forced_win else 'Forced Defense'}")
            print(f"TSS Move: ({r}, {c}), Stats: {tss_result.search_stats}")
            
            return action_probs, np.array([1.0])  # Single visit for TSS move
        
        # Otherwise use MCTS
        print("Using MCTS (no forced moves found)")
        return self.mcts.search(state, temperature)


def demonstrate_tss_mcts():
    """Demonstrate TSS-guided MCTS in action."""
    print("TSS-Guided MCTS Demonstration")
    print("=" * 40)
    
    # Setup components
    env = GomokuEnv(board_size=15)
    model = GomokuNet(board_size=15, num_blocks=4, channels=32)  # Small model for demo
    tss_mcts = TSSGuidedMCTS(model, env, mcts_simulations=200, tss_depth=4, tss_time_cap=50)
    
    # Test scenario 1: Forced defense
    print("\\nScenario 1: Forced Defense")
    print("-" * 25)
    
    env.reset()
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create opponent's open four threat
    for i in range(4):
        board[7, 5 + i] = -1
    
    env.board = board
    env.current_player = 1
    env.last_move = np.array([7, 8])
    
    print("Board state (opponent has open four):")
    env.render()
    
    action_probs, visits = tss_mcts.search(board)
    best_action = np.argmax(action_probs)
    best_r, best_c = divmod(best_action, 15)
    print(f"Recommended move: ({best_r}, {best_c})\\n")
    
    # Test scenario 2: Normal position (MCTS)
    print("Scenario 2: Normal Position")
    print("-" * 25)
    
    env.reset()
    env.step(7 * 15 + 7)  # Center move
    
    print("Board state (normal opening):")
    env.render()
    
    action_probs, visits = tss_mcts.search(env.board)
    best_action = np.argmax(action_probs)
    best_r, best_c = divmod(best_action, 15)
    print(f"Recommended move: ({best_r}, {best_c})\\n")
    
    # Test scenario 3: Tactical position
    print("Scenario 3: Tactical Position")
    print("-" * 25)
    
    env.reset()
    board = np.zeros((15, 15), dtype=np.int8)
    
    # Create tactical setup
    moves = [
        (7, 7, 1), (7, 8, -1), (8, 7, 1), (8, 8, -1),
        (6, 7, 1), (9, 7, -1), (5, 7, 1)  # Potential winning line
    ]
    
    for r, c, player in moves:
        board[r, c] = player
    
    env.board = board
    env.current_player = 1
    env.last_move = np.array([5, 7])
    
    print("Board state (tactical position):")
    env.render()
    
    action_probs, visits = tss_mcts.search(board)
    best_action = np.argmax(action_probs)
    best_r, best_c = divmod(best_action, 15)
    print(f"Recommended move: ({best_r}, {best_c})\\n")


def benchmark_tss_vs_mcts():
    """Benchmark TSS vs pure MCTS performance."""
    print("TSS vs MCTS Performance Comparison")
    print("=" * 40)
    
    env = GomokuEnv(board_size=15)
    model = GomokuNet(board_size=15, num_blocks=4, channels=32)
    
    # Pure MCTS
    mcts = MCTS(model, env, num_simulations=200)
    
    # TSS-guided MCTS
    tss_mcts = TSSGuidedMCTS(model, env, mcts_simulations=200, tss_depth=3, tss_time_cap=30)
    
    # Test on tactical position
    board = np.zeros((15, 15), dtype=np.int8)
    for i in range(4):
        board[7, 5 + i] = -1  # Opponent threat
    
    env.board = board
    env.current_player = 1
    env.last_move = np.array([7, 8])
    
    import time
    
    # Time pure MCTS
    start = time.time()
    mcts_probs, _ = mcts.search(board)
    mcts_time = time.time() - start
    mcts_move = np.argmax(mcts_probs)
    
    # Time TSS-guided MCTS
    start = time.time()
    tss_probs, _ = tss_mcts.search(board)
    tss_time = time.time() - start
    tss_move = np.argmax(tss_probs)
    
    print(f"Pure MCTS: {mcts_time:.3f}s, move {divmod(mcts_move, 15)}")
    print(f"TSS-MCTS:  {tss_time:.3f}s, move {divmod(tss_move, 15)}")
    print(f"Speedup: {mcts_time/tss_time:.1f}x faster with TSS")


if __name__ == "__main__":
    try:
        demonstrate_tss_mcts()
        benchmark_tss_vs_mcts()
        print("✅ TSS-MCTS integration demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)