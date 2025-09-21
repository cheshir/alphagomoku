import math
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..env.gomoku_env import GomokuEnv
from .config import MCTSConfig


class MCTSNode:
    """MCTS tree node"""

    def __init__(
        self,
        state: np.ndarray,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
        current_player: Optional[int] = None,
        last_move: Optional[Tuple[int, int]] = None,
        board_size: Optional[int] = None,
    ):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.prior = prior

        stones = int(np.sum(self.state != 0))
        self.current_player = (
            current_player
            if current_player is not None
            else (1 if stones % 2 == 0 else -1)
        )
        self.last_move = last_move if last_move is not None else (-1, -1)
        self.board_size = board_size

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.is_expanded = False
        self.in_flight = False  # virtual loss marker for batching
        self._lock = threading.Lock()  # Protect node state updates

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(
        self,
        cpuct: float,
        parent_visits: int,
        virtual_loss_penalty: float = -1e9,
    ) -> float:
        """Upper Confidence Bound for Trees with prior"""
        with self._lock:
            q = self.value()
            u = (
                cpuct
                * self.prior
                * math.sqrt(max(1, parent_visits))
                / (1 + self.visit_count)
            )
            if self.in_flight:
                return virtual_loss_penalty
            return q + u

    def select_child(
        self, cpuct: float, virtual_loss_penalty: float = -1e9
    ) -> "MCTSNode":
        """Select child with highest UCT score"""
        return max(
            self.children.values(),
            key=lambda child: child.uct_score(cpuct, self.visit_count, virtual_loss_penalty),
        )

    def expand(self, policy: np.ndarray):
        """Expand node with children for legal actions"""
        self.is_expanded = True
        legal_actions = np.where(self.state.reshape(-1) == 0)[0]
        for action in legal_actions:
            r, c = divmod(action, self.board_size)
            child_state = self.state.copy()
            child_state[r, c] = self.current_player
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                action=action,
                prior=float(policy[action]),
                current_player=-self.current_player,
                last_move=(r, c),
                board_size=self.board_size,
            )

    def backup(self, value: float):
        """Accumulate value to this node only (no recursion)."""
        with self._lock:
            self.visit_count += 1
            self.value_sum += value

    def set_in_flight(self, value: bool):
        """Thread-safe setter for in_flight flag."""
        with self._lock:
            self.in_flight = value

    def is_in_flight(self) -> bool:
        """Thread-safe getter for in_flight flag."""
        with self._lock:
            return self.in_flight


class MCTS:
    """Monte Carlo Tree Search with neural network guidance"""

    def __init__(
        self,
        model,
        env: GomokuEnv,
        config: Optional[MCTSConfig] = None,
        *,
        num_simulations: Optional[int] = None,
        cpuct: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        """Create an MCTS instance.

        Older call sites often passed ``num_simulations``/``cpuct``/``batch_size``
        directly to the constructor or mutated attributes on the instance. To
        remain source compatible we allow those keyword arguments and expose
        properties that transparently forward to :class:`MCTSConfig`.
        """

        self.model = model
        self.env = env

        base_config = config or MCTSConfig()
        if num_simulations is not None:
            if num_simulations < 0:
                raise ValueError("num_simulations must be non-negative")
            base_config.num_simulations = int(num_simulations)
        if cpuct is not None:
            base_config.cpuct = float(cpuct)
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            base_config.batch_size = int(batch_size)

        self.config = base_config
        self.root: Optional[MCTSNode] = None
        self._last_temperature: Optional[float] = None

        # Batched evaluation
        self.eval_queue = deque()
        self.eval_results = {}
        self.last_visit_counts: np.ndarray = np.array([], dtype=float)

    @property
    def num_simulations(self) -> int:
        """Number of simulations to run during search."""
        return self.config.num_simulations

    @num_simulations.setter
    def num_simulations(self, value: int) -> None:
        self.config.num_simulations = int(value)

    @property
    def cpuct(self) -> float:
        """Exploration constant used during tree policy selection."""
        return self.config.cpuct

    @cpuct.setter
    def cpuct(self, value: float) -> None:
        self.config.cpuct = float(value)

    @property
    def batch_size(self) -> int:
        """Batch size for neural network evaluations."""
        return self.config.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.config.batch_size = int(value)

    def search(
        self,
        state: np.ndarray,
        temperature: Optional[float] = None,
        reuse_tree: Optional[bool] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run MCTS and return action probabilities and visit counts"""
        # Use defaults from config if not specified
        temperature = (
            temperature
            if temperature is not None
            else self.config.default_temperature
        )
        reuse_tree = (
            reuse_tree if reuse_tree is not None else self.config.enable_tree_reuse
        )

        if (
            reuse_tree
            and self._last_temperature is not None
            and not math.isclose(temperature, self._last_temperature, rel_tol=0.05, abs_tol=0.05)
        ):
            reuse_tree = False

        if self.config.num_simulations < 0:
            raise ValueError("num_simulations must be non-negative")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        state = np.asarray(state)
        expected_shape = (self.env.board_size, self.env.board_size)
        if state.shape != expected_shape:
            raise ValueError(
                f"state must have shape {expected_shape}, got {state.shape}"
            )
        if not np.issubdtype(state.dtype, np.integer):
            raise ValueError("state must contain integer values")
        if not np.isin(state, (-1, 0, 1)).all():
            raise ValueError("state contains invalid stone values")
        state = state.astype(np.int8, copy=False)

        winner = self._detect_winner_full(state, self.env.board_size)
        board_full = not (state == 0).any()

        if winner != 0 or board_full:
            self.root = MCTSNode(state, board_size=self.env.board_size)
            action_probs = np.zeros(self.env.board_size * self.env.board_size)
            value = self._terminal_value(state, winner)
            self.last_visit_counts = np.array([], dtype=float)
            return action_probs, value

        # Create or reuse root node only when state matches
        if (
            not reuse_tree
            or self.root is None
            or self.root.state.shape != state.shape
            or not np.array_equal(self.root.state, state)
        ):
            self.root = MCTSNode(state, board_size=self.env.board_size)

        # Use batching when enabled and model is on accelerated device
        device = next(self.model.parameters()).device
        use_batched = self.config.batch_size > 1
        if use_batched:
            self._simulate_batched()
        else:
            # Standard simulation
            for _ in range(self.config.num_simulations):
                self._simulate_single()

        # Calculate action probabilities from visit counts
        # Pre-allocate arrays for better performance
        children_items = list(self.root.children.items())
        if len(children_items) == 0:
            actions = np.array([], dtype=int)
            visits = np.array([], dtype=float)
            probs = np.array([], dtype=float)
        else:
            actions = np.array([action for action, _ in children_items], dtype=int)
            visits = np.array([child.visit_count for _, child in children_items], dtype=float)

            if temperature <= self.config.deterministic_threshold:
                # Deterministic selection
                probs = np.zeros(len(visits), dtype=float)
                if len(visits) > 0:
                    probs[np.argmax(visits)] = self.config.win_value
            else:
                # Temperature-based sampling with safeguards for zero totals
                exponent = self.config.win_value / max(temperature, 1e-8)
                visits_temp = np.power(visits, exponent, dtype=float)
                total = float(visits_temp.sum())
                if len(visits_temp) == 0:
                    probs = np.array([], dtype=float)
                elif total <= 0.0 or not np.isfinite(total):
                    # Fall back to uniform distribution over explored moves
                    probs = np.full(len(visits_temp), 1.0 / len(visits_temp), dtype=float)
                else:
                    probs = visits_temp / total

        # Convert to full action space
        action_probs = np.zeros(self.env.board_size * self.env.board_size)
        if actions.size > 0:
            action_probs[actions] = probs

        self.last_visit_counts = visits.copy()
        root_value = self.root.value() if self.root is not None else 0.0
        self._last_temperature = float(temperature)
        return action_probs, float(root_value)

    def _simulate_single(self):
        """Single MCTS simulation (no batching)."""
        node = self.root
        path = [node]

        # Selection
        while node.is_expanded and not node.is_leaf():
            node = node.select_child(self.config.cpuct, self.config.virtual_loss_penalty)
            path.append(node)

        # Terminal check
        terminal, winner = self._is_terminal(
            node.state, node.last_move, self.env.board_size
        )
        if terminal:
            if winner == 0:
                value = self.config.draw_value
            else:
                value = (
                    self.config.win_value
                    if winner == node.current_player
                    else self.config.loss_value
                )
        else:
            # Evaluate and expand if necessary
            policy, value = self._evaluate_node(node)
            if not node.is_expanded:
                node.expand(policy)

        # Backup along path
        v = value
        for n in reversed(path):
            n.backup(v)
            v = -v

    def _simulate_batched(self):
        """Run simulations in batches, sharing NN evaluations with optimized device transfers."""
        sims_done = 0
        device = next(self.model.parameters()).device

        while sims_done < self.config.num_simulations:
            leaves, terminal_count = self._collect_leaves_for_batch(sims_done)

            if terminal_count:
                sims_done += terminal_count

            if not leaves:
                # No expandable leaves available; avoid spinning if terminals exhausted budget
                if terminal_count == 0:
                    break
                continue

            sims_done += self._process_batch_leaves_optimized(leaves, device)

    def _collect_leaves_for_batch(
        self, sims_done: int
    ) -> Tuple[List[Tuple[MCTSNode, List[MCTSNode]]], int]:
        """Collect leaf nodes for batched evaluation.

        Returns a pair ``(leaves, terminal_count)`` where ``leaves`` holds the
        expandable nodes to evaluate and ``terminal_count`` tracks how many
        simulations finished immediately at terminal states during collection.
        """

        leaves: List[Tuple[MCTSNode, List[MCTSNode]]] = []
        terminal_count = 0

        while (
            len(leaves) < self.config.batch_size
            and sims_done + len(leaves) + terminal_count < self.config.num_simulations
        ):
            leaf_result = self._select_and_process_leaf()
            if leaf_result is None:
                break
            if leaf_result == "terminal_processed":
                # Count terminal simulations to advance progress even without NN evals
                terminal_count += 1
                continue
            leaves.append(leaf_result)

        return leaves, terminal_count

    def _select_and_process_leaf(
        self,
    ) -> Optional[Tuple[MCTSNode, List[MCTSNode]]]:
        """Select a leaf node and handle terminal cases.
        Returns None for continue, 'terminal_processed' for handled terminals, or (node, path) for batch.
        """
        node = self.root
        path = [node]

        # Selection with virtual loss marking
        while node.is_expanded and not node.is_leaf():
            node = node.select_child(self.config.cpuct, self.config.virtual_loss_penalty)
            path.append(node)

        # Mark in-flight to reduce duplicate selection within this batch
        node.set_in_flight(True)

        terminal, winner = self._is_terminal(
            node.state, node.last_move, self.env.board_size
        )
        if terminal:
            self._handle_terminal_node(node, path, winner)
            return "terminal_processed"

        return (node, path)

    def _handle_terminal_node(
        self, node: MCTSNode, path: List[MCTSNode], winner: int
    ):
        """Handle terminal node backup and cleanup."""
        if winner == 0:
            value = self.config.draw_value
        else:
            value = (
                self.config.win_value
                if winner == node.current_player
                else self.config.loss_value
            )

        v = value
        for n in reversed(path):
            n.backup(v)
            v = -v
        node.set_in_flight(False)

    def _process_batch_leaves_optimized(
        self, leaves: List[Tuple[MCTSNode, List[MCTSNode]]], device: torch.device
    ) -> int:
        """Process a batch of leaves with optimized device transfers."""
        # Batch convert states to tensors and transfer to device once
        batch_states_np = []
        legal_masks_np = []

        for node, _ in leaves:
            # Convert to numpy first (faster than individual tensor conversions)
            state_np = self._state_to_numpy_from_node(node)
            batch_states_np.append(state_np)
            legal_masks_np.append((node.state.reshape(-1) == 0).astype(np.float32))

        # Single device transfer for entire batch
        batch_states = torch.from_numpy(np.stack(batch_states_np)).float().to(device)
        legal_masks = torch.from_numpy(np.stack(legal_masks_np)).to(device)

        policies_t, values_t = self.model.predict_batch(batch_states)
        if not torch.isfinite(policies_t).all() or not torch.isfinite(values_t).all():
            raise ValueError("Model produced non-finite outputs during batched evaluation")

        # Apply legal masks and normalize per leaf, then expand and backup
        for i, (node, path) in enumerate(leaves):
            policy = policies_t[i]
            legal_mask = legal_masks[i]
            policy = policy * legal_mask
            policy = policy / (policy.sum() + self.config.policy_epsilon)

            value = float(values_t[i].item())

            if not node.is_expanded:
                node.expand(policy.detach().cpu().numpy())

            # Backup along path
            v = value
            for n in reversed(path):
                n.backup(v)
                v = -v

            # Use thread-safe method to clear in_flight flag
            node.set_in_flight(False)

        return len(leaves)

    def _process_single_leaf_result(
        self,
        node: MCTSNode,
        path: List[MCTSNode],
        policy: torch.Tensor,
        value: torch.Tensor,
    ):
        """Process a single leaf's neural network evaluation result."""
        # Apply legal mask and normalize
        legal_mask = torch.from_numpy((node.state.reshape(-1) == 0)).to(
            policy.device, dtype=policy.dtype
        )
        policy = policy * legal_mask
        policy = policy / (policy.sum() + self.config.policy_epsilon)

        value_float = float(value.item())

        if not node.is_expanded:
            node.expand(policy.detach().cpu().numpy())

        # Backup along path
        v = value_float
        for n in reversed(path):
            n.backup(v)
            v = -v

        node.set_in_flight(False)
    def _evaluate_node(self, node: MCTSNode) -> Tuple[np.ndarray, float]:
        """Evaluate a node with the neural network and mask illegal moves."""
        device = next(self.model.parameters()).device

        # Convert to numpy first, then create single tensor on device
        state_np = self._state_to_numpy_from_node(node)
        state_tensor = torch.from_numpy(state_np).float().to(device)

        policy_t, value = self.model.predict(state_tensor)

        # Create legal mask on device directly
        legal_mask_np = (node.state.reshape(-1) == 0).astype(np.float32)
        legal_mask = torch.from_numpy(legal_mask_np).to(device)

        if not torch.isfinite(policy_t).all() or not torch.isfinite(value).all():
            raise ValueError("Model produced non-finite outputs during evaluation")

        policy_t = policy_t * legal_mask
        policy_t = policy_t / (policy_t.sum() + self.config.policy_epsilon)

        policy = policy_t.detach().cpu().numpy()
        return policy, value

    def reuse_subtree(self, action: int):
        """Reuse subtree after making a move"""
        if self.root and action in self.root.children:
            # Set child as new root
            new_root = self.root.children[action]
            new_root.parent = None
            self.root = new_root
        else:
            # No subtree to reuse
            self.root = None

    def _state_to_numpy_from_node(self, node: MCTSNode) -> np.ndarray:
        """Convert node state to numpy array for faster batch processing."""
        board_size = node.board_size

        own_stones = (node.state == node.current_player).astype(np.float32)
        opp_stones = (node.state == -node.current_player).astype(np.float32)

        last_move = np.zeros((board_size, board_size), dtype=np.float32)
        lr, lc = node.last_move
        if lr is not None and lr >= 0:
            last_move[lr, lc] = 1.0

        side_to_move = np.ones((board_size, board_size), dtype=np.float32)
        pattern_maps = np.zeros((board_size, board_size), dtype=np.float32)

        return np.stack(
            [own_stones, opp_stones, last_move, side_to_move, pattern_maps]
        )

    def _state_to_tensor_from_node(self, node: MCTSNode) -> torch.Tensor:
        """Convert node state to NN input tensor."""
        return torch.from_numpy(self._state_to_numpy_from_node(node)).float()

    def _is_terminal(
        self, state: np.ndarray, last_move: Tuple[int, int], board_size: int
    ) -> Tuple[bool, int]:
        """Check terminal condition using last move.
        Returns (terminal, winner), where winner in {-1, 0, 1}.
        """
        stones = int(np.sum(state != 0))
        # Draw
        if stones == board_size * board_size:
            return True, 0

        r, c = last_move
        if r is None or r < 0:
            return False, 0

        player = state[r, c]
        if player == 0:
            return False, 0

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while (
                0 <= rr < board_size
                and 0 <= cc < board_size
                and state[rr, cc] == player
            ):
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while (
                0 <= rr < board_size
                and 0 <= cc < board_size
                and state[rr, cc] == player
            ):
                count += 1
                rr -= dr
                cc -= dc
            if count >= self.config.winning_sequence_length:
                return True, int(player)
        return False, 0

    def _detect_winner_full(self, state: np.ndarray, board_size: int) -> int:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(board_size):
            for c in range(board_size):
                player = state[r, c]
                if player == 0:
                    continue

                for dr, dc in directions:
                    prev_r, prev_c = r - dr, c - dc
                    if (
                        0 <= prev_r < board_size
                        and 0 <= prev_c < board_size
                        and state[prev_r, prev_c] == player
                    ):
                        continue

                    count = 1
                    rr, cc = r + dr, c + dc
                    while (
                        0 <= rr < board_size
                        and 0 <= cc < board_size
                        and state[rr, cc] == player
                    ):
                        count += 1
                        if count >= self.config.winning_sequence_length:
                            return int(player)
                        rr += dr
                        cc += dc

        return 0

    def _terminal_value(self, state: np.ndarray, winner: int) -> float:
        if winner == 0:
            return self.config.draw_value

        stones = int(np.sum(state != 0))
        current_player = 1 if stones % 2 == 0 else -1
        return (
            self.config.win_value
            if winner == current_player
            else self.config.loss_value
        )
