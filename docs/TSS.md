# Threat-Space Search (TSS) Technical Specification

## Overview

Threat-Space Search (TSS) is a specialized search module designed to enhance the strength and tactical awareness of the Gomoku AI by explicitly detecting and exploiting threat sequences such as open-threes, open-fours, double threats, and forced win or forced defense lines. TSS operates alongside the Monte Carlo Tree Search (MCTS) and the endgame αβ solver to improve move selection in complex tactical scenarios, particularly where immediate threats determine the outcome.

## Goals

- Accurately detect critical tactical threats (open-threes, open-fours, broken-fours, double threats) in the current board position.
- Identify forced win or forced defense sequences within a configurable search depth.
- Override or guide the MCTS move selection when forced tactical responses are found.
- Provide configurable depth and time limits to balance strength and inference latency.
- Integrate seamlessly with the existing MCTS and endgame solver pipeline.
- Support difficulty-level adjustments by tuning depth, time caps, and activation thresholds.
- Provide detailed logging for post-game analysis and debugging.

## Integration

- **With MCTS:** TSS is invoked before MCTS rollouts during the search phase. If a forced win or forced defense is detected by TSS, it overrides MCTS, returning the forced move directly.
- **With Endgame Solver:** TSS operates primarily in the midgame tactical phase. When the number of empty cells falls below the endgame solver threshold, the αβ endgame solver takes precedence.
- **Difficulty Levels:** 
  - Easy: TSS disabled or minimal depth (≤2).
  - Medium: TSS depth 3–4, time cap ~100 ms.
  - Strong: TSS depth 6–7, time cap ~300 ms, with priority override on forced lines.
- **Resource Management:** TSS respects configured time and memory budgets to maintain inference latency and memory constraints.

## Algorithm

1. **Threat Detection:**
   - Analyze the board to identify all threat patterns relevant to Gomoku:
     - Open-three (three stones in a row with two open ends).
     - Open-four (four stones with at least one open end).
     - Broken-four, double-threes, double-fours.
   - Use pattern maps and heuristics to prioritize candidate moves.

2. **Threat-Space Exploration:**
   - Perform a depth-limited search exploring sequences of moves that create, block, or extend threats.
   - Alternate between players, simulating forced responses to threats.
   - Detect forced win lines where the current player can guarantee victory regardless of opponent response.
   - Detect forced defense lines where the current player must respond to avoid immediate loss.

3. **Pruning and Heuristics:**
   - Use threat heuristics to prune the search space to relevant moves.
   - Limit branching by focusing on moves that extend or block threats.
   - Employ iterative deepening to respect time caps.

4. **Result:**
   - Return a result object indicating:
     - Whether a forced win or forced defense was found.
     - The best move to play.
     - Search statistics (nodes visited, time used).

## Parameters

- **Search Depth (int):** Maximum ply depth to explore (e.g., 2–7).
- **Time Cap (ms):** Maximum allowed time for the TSS search per invocation.
- **Activation Threshold:** Minimum number of empty cells or game phase to activate TSS.
- **Priority Override (bool):** Whether TSS forced moves override MCTS decisions.
- **Threat Patterns:** Configurable set of threat types to consider.

## Interfaces

### Entry Point

```python
def tss_search(position: Position, depth: int, time_cap_ms: int) -> TSSResult:
    """
    Perform Threat-Space Search on the given position.

    Args:
        position (Position): Current board state and player to move.
        depth (int): Maximum search depth in plies.
        time_cap_ms (int): Time cap in milliseconds.

    Returns:
        TSSResult: Result object containing:
            - forced_move (Optional[Move]): Move to play if forced win/defense found.
            - is_forced_win (bool): True if forced win line detected.
            - is_forced_defense (bool): True if forced defense detected.
            - search_stats (dict): Nodes visited, time used, etc.
    """
```

### Input Definitions

- **Position:** Encapsulates the current board state, player to move, and auxiliary data such as threat pattern maps.
- **Move:** A tuple or object specifying row and column coordinates.

### Output Definitions

- **TSSResult:** Structured data including:
  - `forced_move`: The move recommended by TSS if found, else `None`.
  - `is_forced_win`: Boolean flag.
  - `is_forced_defense`: Boolean flag.
  - `search_stats`: Dictionary with keys like `nodes_visited`, `time_ms`.

## Logging and Monitoring

- Log entries should include:
  - Invocation parameters (depth, time cap).
  - Reason for move selection if TSS forced move is used (`reason='tss_forced'`).
  - Search statistics (nodes visited, time spent).
  - Detected threat types and sequences.
- Monitoring metrics:
  - Average latency per TSS call.
  - Frequency of forced win/defense detections.
  - Memory usage during TSS.
- Logs should be structured and compatible with global system logging, enabling correlation with MCTS and endgame solver logs.

## Deliverables

- **Source Code:** Fully implemented TSS module with clean API.
- **Unit Tests:** Covering threat detection, forced lines, edge cases.
- **Integration Tests:** Validation with MCTS and endgame solver.
- **Performance Benchmarks:** Latency and memory usage under different depths and time caps.
- **Documentation:** Usage guide, configuration parameters, and API reference.
- **Logging Examples:** Sample logs demonstrating TSS forced move overrides and threat detections.
