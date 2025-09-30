"""AI inference engine using MCTS, TSS, and endgame solver."""

import time
from typing import Tuple, List, Optional
import numpy as np
import torch

from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.tss.tss_search import TSSSearcher
from alphagomoku.tss.position import Position as TSSPosition
from alphagomoku.tss import TSSConfig, set_default_config
from alphagomoku.endgame.endgame_solver import EndgameSolver
from alphagomoku.endgame.position import EndgamePosition

from .config import settings, DifficultyConfig
from .models import DebugInfo, TopMove


class InferenceEngine:
    """Handles AI move generation with unified search stack."""

    def __init__(self, model_path: str):
        """Initialize the inference engine."""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = GomokuNet(board_size=settings.BOARD_SIZE)
        # Set weights_only=False since this is a trusted checkpoint from training
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")

        # Create environment
        self.env = GomokuEnv(board_size=settings.BOARD_SIZE)

        # Initialize TSS and endgame solver
        self.tss_searcher = TSSSearcher(board_size=settings.BOARD_SIZE)
        self.endgame_solver = EndgameSolver()

        # Initialize TSS configs for each difficulty
        # For inference, we use full TSS to maximize strength
        self.tss_configs = {
            "easy": TSSConfig.for_inference("easy"),
            "medium": TSSConfig.for_inference("medium"),
            "hard": TSSConfig.for_inference("hard"),
        }
        print("TSS configs initialized for all difficulty levels")

    def get_ai_move(
        self, board: np.ndarray, current_player: int, difficulty: str
    ) -> Tuple[Tuple[int, int], DebugInfo]:
        """
        Get AI move using unified search stack (Endgame → TSS → MCTS).

        Args:
            board: Current board state (15x15 numpy array)
            current_player: Current player (1 or -1)
            difficulty: Difficulty level

        Returns:
            Tuple of (move, debug_info)
        """
        start_time = time.time()
        search_method = "mcts"  # Track which method was used
        tss_result = None
        endgame_result = None

        # Get difficulty config
        config = settings.DIFFICULTIES[difficulty]

        # Set TSS config for this difficulty level
        tss_config = self.tss_configs.get(difficulty, TSSConfig.for_inference("medium"))
        set_default_config(tss_config)

        # Set environment state
        self.env.board = board.copy()
        self.env.current_player = current_player

        # Count empty cells
        empty_cells = int(np.sum(board == 0))

        # 1. Try endgame solver first (if enabled and threshold met)
        if config.use_endgame_solver and empty_cells <= config.endgame_threshold:
            try:
                endgame_pos = EndgamePosition(
                    board=board.copy(),
                    current_player=current_player,
                    board_size=settings.BOARD_SIZE
                )
                endgame_result = self.endgame_solver.search(
                    endgame_pos,
                    max_depth=20,
                    time_limit=1.0
                )

                if endgame_result.best_move and (endgame_result.is_win or endgame_result.is_loss):
                    # Endgame solver found a forced sequence
                    row, col = endgame_result.best_move
                    thinking_time_ms = (time.time() - start_time) * 1000

                    debug_info = self._create_debug_info_from_endgame(
                        endgame_result=endgame_result,
                        board=board,
                        config=config,
                        thinking_time_ms=thinking_time_ms,
                        search_method="endgame_solver"
                    )

                    return (int(row), int(col)), debug_info
            except Exception as e:
                print(f"Endgame solver error: {e}")

        # 2. Try TSS (if enabled and no endgame solution)
        if config.use_tss:
            try:
                tss_pos = TSSPosition(
                    board=board.copy(),
                    current_player=current_player,
                    board_size=settings.BOARD_SIZE
                )
                tss_result = self.tss_searcher.search(
                    tss_pos,
                    depth=config.tss_depth,
                    time_cap_ms=config.tss_time_cap_ms
                )

                if tss_result.forced_move and (tss_result.is_forced_win or tss_result.is_forced_defense):
                    # TSS found a forced tactical sequence
                    row, col = tss_result.forced_move
                    thinking_time_ms = (time.time() - start_time) * 1000

                    debug_info = self._create_debug_info_from_tss(
                        tss_result=tss_result,
                        board=board,
                        config=config,
                        thinking_time_ms=thinking_time_ms,
                        search_method="tss_forced"
                    )

                    return (int(row), int(col)), debug_info
            except Exception as e:
                print(f"TSS error: {e}")

        # 3. Fall back to MCTS
        mcts = MCTS(
            model=self.model,
            env=self.env,
            num_simulations=config.simulations,
            cpuct=config.cpuct,
            batch_size=settings.BATCH_SIZE,
        )

        # Run MCTS search
        action_probs, value = mcts.search(board, temperature=config.temperature)

        # Get the best move
        legal_mask = self.env._get_action_mask()
        masked_probs = action_probs * legal_mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

        action = np.argmax(masked_probs)
        row, col = divmod(action, settings.BOARD_SIZE)

        # Calculate thinking time
        thinking_time_ms = (time.time() - start_time) * 1000

        # Get debug info
        debug_info = self._extract_debug_info(
            action_probs=masked_probs,
            value=value,
            mcts=mcts,
            config=config,
            thinking_time_ms=thinking_time_ms,
            search_method="mcts",
            tss_result=tss_result,
            endgame_result=endgame_result,
        )

        return (int(row), int(col)), debug_info

    def _create_debug_info_from_endgame(
        self,
        endgame_result,
        board: np.ndarray,
        config: DifficultyConfig,
        thinking_time_ms: float,
        search_method: str
    ) -> DebugInfo:
        """Create debug info from endgame solver result."""
        # Create a simple policy distribution showing the endgame move
        policy_2d = np.zeros((settings.BOARD_SIZE, settings.BOARD_SIZE))
        if endgame_result.best_move:
            row, col = endgame_result.best_move
            policy_2d[row, col] = 1.0

        return DebugInfo(
            policy_distribution=policy_2d.tolist(),
            value_estimate=float(endgame_result.evaluation),
            simulations=0,  # Endgame solver doesn't use simulations
            thinking_time_ms=thinking_time_ms,
            top_moves=[
                TopMove(
                    row=endgame_result.best_move[0],
                    col=endgame_result.best_move[1],
                    probability=1.0,
                    visits=endgame_result.search_stats.get('nodes_searched', 0)
                )
            ] if endgame_result.best_move else [],
            search_depth=abs(endgame_result.depth_to_mate) if endgame_result.depth_to_mate else 0,
            nodes_explored=endgame_result.search_stats.get('nodes_searched', 0),
        )

    def _create_debug_info_from_tss(
        self,
        tss_result,
        board: np.ndarray,
        config: DifficultyConfig,
        thinking_time_ms: float,
        search_method: str
    ) -> DebugInfo:
        """Create debug info from TSS result."""
        # Create a simple policy distribution showing the TSS move
        policy_2d = np.zeros((settings.BOARD_SIZE, settings.BOARD_SIZE))
        if tss_result.forced_move:
            row, col = tss_result.forced_move
            policy_2d[row, col] = 1.0

        # Estimate value based on TSS result
        if tss_result.is_forced_win:
            value_estimate = 0.9
        elif tss_result.is_forced_defense:
            value_estimate = -0.3
        else:
            value_estimate = 0.0

        return DebugInfo(
            policy_distribution=policy_2d.tolist(),
            value_estimate=value_estimate,
            simulations=0,  # TSS doesn't use simulations
            thinking_time_ms=thinking_time_ms,
            top_moves=[
                TopMove(
                    row=tss_result.forced_move[0],
                    col=tss_result.forced_move[1],
                    probability=1.0,
                    visits=tss_result.search_stats.get('nodes_visited', 0)
                )
            ] if tss_result.forced_move else [],
            search_depth=config.tss_depth,
            nodes_explored=tss_result.search_stats.get('nodes_visited', 0),
        )

    def _extract_debug_info(
        self,
        action_probs: np.ndarray,
        value: np.ndarray,
        mcts: MCTS,
        config: DifficultyConfig,
        thinking_time_ms: float,
        search_method: str = "mcts",
        tss_result=None,
        endgame_result=None,
    ) -> DebugInfo:
        """Extract debug information from MCTS search."""
        # Reshape policy to 2D
        policy_2d = action_probs.reshape(settings.BOARD_SIZE, settings.BOARD_SIZE)

        # Get top moves
        top_indices = np.argsort(action_probs.flatten())[::-1][:5]
        top_moves = []

        for idx in top_indices:
            if action_probs.flatten()[idx] > 0:
                row, col = divmod(int(idx), settings.BOARD_SIZE)
                # Get visit count from root node's children if available
                visit_count = 0
                if mcts.root and idx in mcts.root.children:
                    visit_count = mcts.root.children[idx].visit_count

                top_moves.append(
                    TopMove(
                        row=row,
                        col=col,
                        probability=float(action_probs.flatten()[idx]),
                        visits=visit_count,
                    )
                )

        # Calculate nodes explored
        nodes_explored = self._count_nodes(mcts.root) if mcts.root else 0
        total_visits = mcts.root.visit_count if mcts.root else 0

        # Estimate search depth
        search_depth = self._estimate_depth(mcts.root) if mcts.root else 0

        return DebugInfo(
            policy_distribution=policy_2d.tolist(),
            value_estimate=float(value) if isinstance(value, (int, float)) else float(value[0]) if len(value.shape) > 0 else float(value),
            simulations=config.simulations,
            thinking_time_ms=thinking_time_ms,
            top_moves=top_moves,
            search_depth=search_depth,
            nodes_explored=nodes_explored,
        )

    def _count_nodes(self, node) -> int:
        """Recursively count all nodes in the tree."""
        if node is None:
            return 0
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _estimate_depth(self, node, current_depth=0) -> int:
        """Estimate average depth of the tree."""
        if node is None or not node.children:
            return current_depth

        max_depth = current_depth
        for child in node.children.values():
            child_depth = self._estimate_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)

        return max_depth