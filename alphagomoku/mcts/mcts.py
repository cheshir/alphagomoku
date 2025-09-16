import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import deque
from ..env.gomoku_env import GomokuEnv


class MCTSNode:
    """MCTS tree node"""

    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None,
                 action: Optional[int] = None, prior: float = 0.0,
                 current_player: Optional[int] = None,
                 last_move: Optional[Tuple[int, int]] = None,
                 board_size: Optional[int] = None):
        self.state = state.copy()
        self.parent = parent
        self.action = action
        self.prior = prior

        stones = int(np.sum(self.state != 0))
        self.current_player = current_player if current_player is not None else (1 if stones % 2 == 0 else -1)
        self.last_move = last_move if last_move is not None else (-1, -1)
        self.board_size = board_size

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}
        self.is_expanded = False
        self.in_flight = False  # virtual loss marker for batching
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(self, cpuct: float, parent_visits: int) -> float:
        """Upper Confidence Bound for Trees with prior"""
        q = self.value()
        u = cpuct * self.prior * math.sqrt(max(1, parent_visits)) / (1 + self.visit_count)
        if self.in_flight:
            return -1e9
        return q + u

    def select_child(self, cpuct: float) -> 'MCTSNode':
        """Select child with highest UCT score"""
        return max(self.children.values(),
                  key=lambda child: child.uct_score(cpuct, self.visit_count))

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
                board_size=self.board_size
            )



    def backup(self, value: float):
        """Accumulate value to this node only (no recursion)."""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """Monte Carlo Tree Search with neural network guidance"""

    def __init__(self, model, env: GomokuEnv, cpuct: float = 1.8,
                 num_simulations: int = 800, batch_size: int = 32):
        self.model = model
        self.env = env
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        # Disable batching for now due to performance issues
        self.batch_size = batch_size
        self.root: Optional[MCTSNode] = None

        # Batched evaluation
        self.eval_queue = deque()
        self.eval_results = {}

    def search(self, state: np.ndarray, temperature: float = 1.0, reuse_tree: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Run MCTS and return action probabilities and visit counts"""
        # Create or reuse root node
        if not reuse_tree or self.root is None:
            self.root = MCTSNode(state, board_size=self.env.board_size)

        # Use batching when enabled and model is on accelerated device
        device = next(self.model.parameters()).device
        use_batched = (self.batch_size > 1) and (device.type in ("mps", "cuda"))
        if use_batched:
            self._simulate_batched()
        else:
            # Standard simulation
            for _ in range(self.num_simulations):
                self._simulate_single()

        # Calculate action probabilities from visit counts
        # Pre-allocate arrays for better performance
        children_items = list(self.root.children.items())
        if len(children_items) == 0:
            actions = np.array([], dtype=int)
            visits = np.array([], dtype=float)
        else:
            actions = np.array([action for action, _ in children_items])
            visits = np.array([child.visit_count for _, child in children_items])

        if temperature == 0:
            # Deterministic selection
            probs = np.zeros(len(visits))
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature-based sampling
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()

        # Convert to full action space
        action_probs = np.zeros(self.env.board_size * self.env.board_size)
        if actions.size > 0:
            action_probs[actions] = probs

        return action_probs, visits

    def _simulate_single(self):
        """Single MCTS simulation (no batching)."""
        node = self.root
        path = [node]

        # Selection
        while node.is_expanded and not node.is_leaf():
            node = node.select_child(self.cpuct)
            path.append(node)

        # Terminal check
        terminal, winner = self._is_terminal(node.state, node.last_move, self.env.board_size)
        if terminal:
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == node.current_player else -1.0
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

        while sims_done < self.num_simulations:
            leaves: List[Tuple[MCTSNode, List[MCTSNode]]] = []

            # Collect up to batch_size leaves
            while len(leaves) < self.batch_size and (sims_done + len(leaves) < self.num_simulations):
                node = self.root
                path = [node]

                # Selection with virtual loss marking
                while node.is_expanded and not node.is_leaf():
                    node = node.select_child(self.cpuct)
                    path.append(node)

                # Mark in-flight to reduce duplicate selection within this batch
                node.in_flight = True

                terminal, winner = self._is_terminal(node.state, node.last_move, self.env.board_size)
                if terminal:
                    # Immediate backup; no NN eval needed
                    if winner == 0:
                        value = 0.0
                    else:
                        value = 1.0 if winner == node.current_player else -1.0
                    v = value
                    for n in reversed(path):
                        n.backup(v)
                        v = -v
                    node.in_flight = False
                    sims_done += 1
                    continue

                leaves.append((node, path))

            if not leaves:
                continue

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

            # Apply legal masks and normalize per leaf, then expand and backup
            for i, (node, path) in enumerate(leaves):
                policy = policies_t[i]
                legal_mask = legal_masks[i]
                policy = policy * legal_mask
                policy = policy / (policy.sum() + 1e-8)

                value = float(values_t[i].item())

                if not node.is_expanded:
                    node.expand(policy.detach().cpu().numpy())

                # Backup along path
                v = value
                for n in reversed(path):
                    n.backup(v)
                    v = -v

                node.in_flight = False
                sims_done += 1

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

        policy_t = policy_t * legal_mask
        policy_t = policy_t / (policy_t.sum() + 1e-8)

        policy = policy_t.detach().cpu().numpy()
        return policy, value

    def _batch_evaluate_leaves(self, leaf_data: List[Tuple[MCTSNode, List[MCTSNode]]]):
        """Deprecated: kept for compatibility; not used in new batching."""
        return

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
    
    def _state_to_tensor(self, env: GomokuEnv) -> torch.Tensor:
        """Convert environment state to neural network input tensor (legacy)."""
        board_size = env.board_size
        
        # Channel 0: Current player's stones
        own_stones = (env.board == env.current_player).astype(np.float32)
        
        # Channel 1: Opponent's stones  
        opp_stones = (env.board == -env.current_player).astype(np.float32)
        
        # Channel 2: Last move
        last_move = np.zeros((board_size, board_size), dtype=np.float32)
        if env.last_move[0] >= 0:
            last_move[env.last_move[0], env.last_move[1]] = 1.0
        
        # Channel 3: Side to move (1 for current player)
        side_to_move = np.ones((board_size, board_size), dtype=np.float32)
        
        # Channel 4: Pattern maps (simplified - can be enhanced later)
        pattern_maps = np.zeros((board_size, board_size), dtype=np.float32)
        
        # Stack channels
        state = np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps])
        return torch.FloatTensor(state)
    
    def _create_env_from_state(self, state: np.ndarray) -> GomokuEnv:
        """Legacy helper (kept for compatibility); not used in new batching."""
        env = GomokuEnv(board_size=self.env.board_size)
        env.board = state.copy()
        stones = np.sum(state != 0)
        env.current_player = 1 if stones % 2 == 0 else -1
        env.move_count = stones
        return env

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

        return np.stack([own_stones, opp_stones, last_move, side_to_move, pattern_maps])

    def _state_to_tensor_from_node(self, node: MCTSNode) -> torch.Tensor:
        """Convert node state to NN input tensor."""
        return torch.from_numpy(self._state_to_numpy_from_node(node)).float()

    @staticmethod
    def _is_terminal(state: np.ndarray, last_move: Tuple[int, int], board_size: int) -> Tuple[bool, int]:
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
            while 0 <= rr < board_size and 0 <= cc < board_size and state[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < board_size and 0 <= cc < board_size and state[rr, cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 5:
                return True, int(player)
        return False, 0
