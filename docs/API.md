# Gomoku API Documentation

## Overview

RESTful API for playing Gomoku against AI with three difficulty levels.

## Base URL

```
http://localhost:8000
```

## Endpoints

### Health Check

**GET** `/health`

Check if the service is ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Create New Game

**POST** `/api/games`

Create a new game session.

**Request Body:**
```json
{
  "difficulty": "medium",  // "easy" | "medium" | "hard"
  "player_color": 1        // 1 (black, goes first) | -1 (white)
}
```

**Response:**
```json
{
  "game_id": "550e8400-e29b-41d4-a716-446655440000",
  "board_size": 15,
  "difficulty": "medium",
  "player_color": 1,
  "current_player": 1,
  "status": "in_progress"
}
```

### Get Game State

**GET** `/api/games/{game_id}`

Get current game state.

**Response:**
```json
{
  "game_id": "550e8400-e29b-41d4-a716-446655440000",
  "board": [[0, 0, ...], ...],  // 15x15 array
  "current_player": -1,
  "status": "in_progress",      // "in_progress" | "player_won" | "ai_won" | "draw"
  "last_move": {"row": 7, "col": 7, "player": 1},
  "move_count": 5,
  "player_time": 120.5,         // seconds used
  "ai_time": 45.2
}
```

### Make Move

**POST** `/api/games/{game_id}/move`

Make a player move. AI will respond automatically.

**Request Body:**
```json
{
  "row": 7,
  "col": 8
}
```

**Response:**
```json
{
  "player_move": {"row": 7, "col": 8, "player": 1},
  "ai_move": {"row": 8, "col": 8, "player": -1},
  "board": [[0, 0, ...], ...],
  "current_player": 1,
  "status": "in_progress",
  "move_count": 7,
  "debug_info": {
    "policy_distribution": [[0.001, 0.002, ...], ...],
    "value_estimate": 0.15,
    "simulations": 128,
    "thinking_time_ms": 450,
    "top_moves": [
      {"row": 8, "col": 8, "probability": 0.35, "visits": 45},
      {"row": 7, "col": 9, "probability": 0.22, "visits": 28}
    ],
    "search_depth": 12,
    "nodes_explored": 450
  }
}
```

### Resign Game

**POST** `/api/games/{game_id}/resign`

Player resigns the current game.

**Response:**
```json
{
  "game_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "ai_won",
  "winner": -1
}
```

### Delete Game

**DELETE** `/api/games/{game_id}`

Delete a game session (cleanup).

**Response:**
```json
{
  "message": "Game deleted successfully"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid move: position already occupied"
}
```

### 404 Not Found
```json
{
  "detail": "Game not found"
}
```

### 422 Validation Error
```json
{
  "detail": "Invalid difficulty level"
}
```

## Difficulty Configurations

| Difficulty | MCTS Sims | TSS      | Endgame Solver | Avg Response Time |
|------------|-----------|----------|----------------|-------------------|
| Easy       | 64        | Disabled | Disabled       | ~100ms           |
| Medium     | 128       | Depth 4  | ≤14 empties    | ~200-500ms       |
| Hard       | 256       | Depth 6  | ≤20 empties    | ~500ms-1s        |

### Unified Search Stack

The AI uses a prioritized search stack for optimal play:

1. **Endgame Solver** (highest priority)
   - Exact alpha-beta search for positions with few empty cells
   - Guarantees optimal play in solved endgames
   - Medium: activates at ≤14 empty cells
   - Hard: activates at ≤20 empty cells

2. **Threat-Space Search (TSS)** (medium priority)
   - Detects forced win/defense sequences
   - Medium: depth 4, 100ms time cap
   - Hard: depth 6, 300ms time cap

3. **MCTS** (fallback for general positions)
   - Neural network guided tree search
   - Handles complex strategic positions

## WebSocket Support (Future)

For real-time updates and notifications:

**WS** `/ws/games/{game_id}`

Not implemented in initial version.