"""Generate tactical training examples to improve model's defensive/offensive awareness."""

import numpy as np
from typing import List
from ..selfplay.selfplay import SelfPlayData


def generate_tactical_positions(board_size: int = 15) -> List[SelfPlayData]:
    """Generate synthetic tactical training positions.

    Creates positions with immediate threats that require forced responses.
    This helps the model learn:
    1. Immediate five-in-a-row completion
    2. Open-four defense
    3. Broken-four handling

    Args:
        board_size: Size of the board

    Returns:
        List of SelfPlayData with forced move examples
    """
    examples = []

    # Pattern 1: Complete open-four to five (immediate win)
    examples.extend(_generate_complete_five_patterns(board_size))

    # Pattern 2: Block opponent's open-four (immediate defense)
    examples.extend(_generate_defense_patterns(board_size))

    # Pattern 3: Create open-three patterns
    examples.extend(_generate_open_three_patterns(board_size))

    return examples


def _generate_complete_five_patterns(board_size: int) -> List[SelfPlayData]:
    """Generate positions where completing five in a row wins immediately."""
    examples = []

    # Horizontal open-four
    board = np.zeros((board_size, board_size), dtype=np.int8)
    # Four in a row: X X X X _ X
    board[7, 5:9] = 1  # Four stones
    # Correct move: complete the five
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[7 * board_size + 9] = 1.0  # Complete on the right

    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=1.0,  # Winning move
        current_player=1,
        last_move=(7, 8),
    ))

    # Vertical open-four
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[5:9, 7] = 1  # Four stones vertically
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[9 * board_size + 7] = 1.0
    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=1.0,
        current_player=1,
        last_move=(8, 7),
    ))

    # Diagonal open-four
    board = np.zeros((board_size, board_size), dtype=np.int8)
    for i in range(4):
        board[5 + i, 5 + i] = 1
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[9 * board_size + 9] = 1.0
    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=1.0,
        current_player=1,
        last_move=(8, 8),
    ))

    return examples


def _generate_defense_patterns(board_size: int) -> List[SelfPlayData]:
    """Generate positions requiring immediate defense."""
    examples = []

    # Opponent has open-four horizontally - MUST block
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 5:9] = -1  # Opponent's four stones
    policy = np.zeros(board_size * board_size, dtype=np.float32)

    # MUST block at either end
    policy[7 * board_size + 4] = 0.5  # Block left
    policy[7 * board_size + 9] = 0.5  # Block right

    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),  # Our turn to defend
        policy=policy,
        value=-0.8,  # Bad position, but can defend
        current_player=1,
        last_move=(7, 8),
    ))

    # Opponent has broken-four (X X X _ X) - MUST block the gap
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 5] = -1
    board[7, 6] = -1
    board[7, 7] = -1
    board[7, 9] = -1  # Gap at position 8
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[7 * board_size + 8] = 1.0  # MUST fill the gap

    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=-0.7,
        current_player=1,
        last_move=(7, 9),
    ))

    # Opponent has vertical open-four
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[5:9, 7] = -1
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    policy[4 * board_size + 7] = 0.5
    policy[9 * board_size + 7] = 0.5
    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=-0.8,
        current_player=1,
        last_move=(8, 7),
    ))

    return examples


def _generate_open_three_patterns(board_size: int) -> List[SelfPlayData]:
    """Generate open-three patterns (important tactical positions)."""
    examples = []

    # Create open-three horizontally
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[7, 6:9] = 1  # Three stones: _ X X X _
    policy = np.zeros(board_size * board_size, dtype=np.float32)
    # Good moves: extend to open-four
    policy[7 * board_size + 5] = 0.4  # Left extension
    policy[7 * board_size + 9] = 0.4  # Right extension
    # Other moves get small probability
    legal_moves = (board.reshape(-1) == 0)
    policy[legal_moves] += 0.2 / legal_moves.sum()
    policy /= policy.sum()

    examples.append(SelfPlayData(
        state=_board_to_state(board, 1),
        policy=policy,
        value=0.6,  # Good attacking position
        current_player=1,
        last_move=(7, 8),
    ))

    return examples


def _board_to_state(board: np.ndarray, current_player: int) -> np.ndarray:
    """Convert simple board to 5-channel state tensor.

    Args:
        board: Simple board (H, W) with {-1, 0, 1}
        current_player: Current player (1 or -1)

    Returns:
        State tensor (5, H, W) with [own, opp, last, side, pattern]
    """
    from ..utils.pattern_detector import get_pattern_features

    h, w = board.shape

    # Channel 0: Own stones
    own = (board == current_player).astype(np.float32)

    # Channel 1: Opponent stones
    opp = (board == -current_player).astype(np.float32)

    # Channel 2: Last move (set to zeros for synthetic data)
    last = np.zeros((h, w), dtype=np.float32)

    # Channel 3: Side to move
    side = np.ones((h, w), dtype=np.float32)

    # Channel 4: Pattern maps (NOW ACTUALLY COMPUTED!)
    pattern = get_pattern_features(board, current_player)

    return np.stack([own, opp, last, side, pattern])


def augment_with_tactical_data(
    selfplay_data: List[SelfPlayData],
    board_size: int = 15,
    augmentation_ratio: float = 0.1
) -> List[SelfPlayData]:
    """Augment self-play data with tactical training examples.

    Args:
        selfplay_data: Original self-play data
        board_size: Size of the board
        augmentation_ratio: Ratio of tactical examples to add (0.1 = 10%)

    Returns:
        Augmented data with tactical examples mixed in
    """
    # Generate tactical examples
    tactical_examples = generate_tactical_positions(board_size)

    # Calculate how many tactical examples to add
    num_tactical = int(len(selfplay_data) * augmentation_ratio)

    # Randomly sample tactical examples (with replacement if needed)
    if num_tactical > 0:
        indices = np.random.choice(len(tactical_examples), size=num_tactical, replace=True)
        selected_tactical = [tactical_examples[i] for i in indices]

        # Mix with self-play data
        return selfplay_data + selected_tactical
    else:
        return selfplay_data
