# Gomoku UI & Backend Implementation

Complete technical documentation for the web application implementation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                          │
│                    http://localhost:5173                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ HTTP/REST API
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                           │
│                   (Port 8000)                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Game Manager                                         │  │
│  │  - Session management                                 │  │
│  │  - Board state tracking                              │  │
│  │  - Win detection                                     │  │
│  │  - Timer management                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Inference Engine (Unified Search Stack)             │  │
│  │  1. Endgame Solver (Priority 1)                      │  │
│  │  2. TSS - Threat-Space Search (Priority 2)           │  │
│  │  3. MCTS - Monte Carlo Tree Search (Priority 3)      │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Your Trained Model                                   │  │
│  │  - GomokuNet (DW-ResNet-SE)                          │  │
│  │  - Policy + Value heads                              │  │
│  │  - ~100 epochs trained                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Backend Implementation

### Unified Search Stack

The backend implements a sophisticated three-tier search hierarchy:

#### 1. Endgame Solver (Priority 1)
- **When**: Activates when empty cells ≤ threshold
- **Method**: Exact alpha-beta search with transposition tables
- **Guarantees**: Optimal play in solved positions
- **Configuration**:
  - Easy: Disabled
  - Medium: ≤14 empty cells
  - Hard: ≤20 empty cells
- **Performance**: 50-200ms, solves typical endgames

#### 2. Threat-Space Search (Priority 2)
- **When**: After endgame check, if TSS enabled
- **Method**: Tactical pattern recognition, forced sequence detection
- **Returns**: Forced win/defense moves when found
- **Configuration**:
  - Easy: Disabled
  - Medium: Depth 4, 100ms time cap
  - Hard: Depth 6, 300ms time cap
- **Performance**: Very fast when forced sequence found (<100ms)

#### 3. MCTS (Priority 3 - Fallback)
- **When**: When endgame solver and TSS find nothing
- **Method**: Neural network guided Monte Carlo Tree Search
- **Returns**: Best strategic move
- **Configuration**:
  - Easy: 64 simulations
  - Medium: 128 simulations
  - Hard: 256 simulations
- **Performance**: Scales with simulation count

### API Endpoints

**Core Endpoints:**
- `POST /api/games` - Create new game with difficulty and color
- `GET /api/games/{id}` - Get current game state
- `POST /api/games/{id}/move` - Player move + AI response (single call)
- `POST /api/games/{id}/resign` - Player resigns
- `DELETE /api/games/{id}` - Cleanup game session
- `GET /health` - Health check with model status

**Key Features:**
- Single API call returns both player and AI moves
- Comprehensive debug information in response
- Automatic time tracking for both players
- Game state validation and win detection

### Debug Information

Every AI move returns detailed metrics:

```json
{
  "debug_info": {
    "value_estimate": 0.15,
    "simulations": 128,
    "thinking_time_ms": 450.2,
    "search_depth": 8,
    "nodes_explored": 1250,
    "top_moves": [
      {"row": 7, "col": 8, "probability": 0.35, "visits": 45},
      {"row": 8, "col": 7, "probability": 0.22, "visits": 28}
    ],
    "policy_distribution": [[...], ...]
  }
}
```

### Configuration

Difficulty settings in `apps/backend/app/config.py`:

```python
class DifficultyConfig(BaseModel):
    simulations: int                # MCTS simulation count
    temperature: float              # Move selection temperature
    cpuct: float                   # MCTS exploration parameter
    use_tss: bool                  # Enable Threat-Space Search
    tss_depth: int                 # TSS search depth
    tss_time_cap_ms: int          # TSS time limit
    use_endgame_solver: bool       # Enable endgame solver
    endgame_threshold: int         # Activate at N empty cells

DIFFICULTIES = {
    "easy": DifficultyConfig(
        simulations=64, temperature=0.2,
        use_tss=False, use_endgame_solver=False
    ),
    "medium": DifficultyConfig(
        simulations=128, temperature=0.0,
        use_tss=True, tss_depth=4, tss_time_cap_ms=100,
        use_endgame_solver=True, endgame_threshold=14
    ),
    "hard": DifficultyConfig(
        simulations=256, temperature=0.0,
        use_tss=True, tss_depth=6, tss_time_cap_ms=300,
        use_endgame_solver=True, endgame_threshold=20
    ),
}
```

## Frontend Implementation

### Technology Stack

- **Framework**: Vue 3 with Composition API
- **Language**: TypeScript
- **State Management**: Pinia
- **HTTP Client**: Axios
- **Build Tool**: Vite 6
- **Styling**: Scoped CSS

### Component Architecture

```
App.vue (Main Layout)
├── GameControls.vue (Sidebar)
│   ├── New game settings
│   ├── Game status
│   ├── Turn indicator
│   └── Action buttons
├── GameTimer.vue
│   ├── Player timer
│   └── AI timer
├── GomokuBoard.vue (Main Board)
│   ├── SVG rendering
│   ├── Grid and star points
│   ├── Stone placement
│   ├── Last move indicator
│   └── Hover preview
└── DebugPanel.vue (Collapsible)
    ├── Search metrics
    ├── Top moves
    └── Policy heatmap
```

### Key Components

#### GomokuBoard.vue
- **Rendering**: SVG-based for crisp display at any size
- **Theme**: Modern wood texture (#d4a574)
- **Stones**: Radial gradient for 3D effect
- **Interaction**: Click to place, hover preview
- **Features**:
  - Last move marker (small dot on stone)
  - Star points for traditional Gomoku board
  - Legal move validation
  - Disabled during AI thinking

#### GameControls.vue
- **States**: Setup mode vs playing mode
- **Turn Indicator**: Visual feedback with colors
  - Green: Your turn
  - Yellow: AI's turn
  - Blue: AI thinking (with spinner)
  - Red: Game over
- **Settings**: Difficulty and color selection
- **Actions**: New game, resign buttons

#### GameTimer.vue
- **Display**: MM:SS format
- **Highlighting**: Active player's timer highlighted
- **Icons**: Black/white stone indicators
- **Updates**: Real-time from backend

#### DebugPanel.vue
- **Collapsible**: Click header to expand/collapse
- **Sections**:
  - Search Metrics (grid layout)
  - Top 5 Moves (with probability bars)
  - Policy Heatmap (15×15 visualization)
- **Features**:
  - Value estimate color coding (positive/negative/neutral)
  - Probability bars for top moves
  - Interactive heatmap with hover tooltips
  - Show/hide policy distribution

### State Management

Pinia store (`stores/game.ts`):

```typescript
interface GameStore {
  // State
  gameId: string | null
  board: number[][]
  difficulty: Difficulty
  playerColor: Player
  currentPlayer: Player
  status: GameStatus
  debugInfo: DebugInfo | null
  isAiThinking: boolean
  error: string | null

  // Computed
  isPlayerTurn: boolean
  isGameOver: boolean
  winner: string | null

  // Actions
  createGame(difficulty, color): Promise<void>
  makeMove(row, col): Promise<void>
  resign(): Promise<void>
  resetGame(): void
}
```

### API Integration

API client (`api/client.ts`):

```typescript
const gameApi = {
  createGame(request): Promise<CreateGameResponse>
  getGameState(gameId): Promise<GameState>
  makeMove(gameId, move): Promise<MoveResponse>
  resignGame(gameId): Promise<ResignResponse>
  deleteGame(gameId): Promise<void>
  healthCheck(): Promise<HealthStatus>
}
```

### User Experience Flow

1. **Game Creation**:
   - User selects difficulty and color
   - Frontend calls `POST /api/games`
   - If AI goes first (player is white), AI move included in response
   - Board updates immediately

2. **Making Moves**:
   - User clicks board intersection
   - Frontend validates (position empty, player's turn)
   - Calls `POST /api/games/{id}/move`
   - Shows "AI thinking..." spinner
   - Response includes both player and AI moves
   - Board updates with both moves
   - Debug panel refreshes with AI metrics

3. **Game Over**:
   - Backend detects win/draw
   - Status returned in move response
   - Frontend shows winner message
   - Board becomes non-interactive
   - Options: Resign button disabled, New Game enabled

## File Structure

```
apps/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app & endpoints
│   │   ├── config.py            # Difficulty configurations
│   │   ├── models.py            # Pydantic models
│   │   ├── game_manager.py      # Game state management
│   │   └── inference.py         # Unified search stack
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run.py
│   └── README.md
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── GomokuBoard.vue
    │   │   ├── GameControls.vue
    │   │   ├── GameTimer.vue
    │   │   └── DebugPanel.vue
    │   ├── stores/
    │   │   └── game.ts           # Pinia store
    │   ├── api/
    │   │   └── client.ts         # API client
    │   ├── types/
    │   │   └── game.ts           # TypeScript types
    │   ├── App.vue               # Main app
    │   └── main.ts               # Entry point
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig.json
    ├── package.json
    ├── Dockerfile
    └── README.md
```

## Deployment

### Docker Setup

**docker-compose.yml**:
```yaml
services:
  backend:
    - Python 3.12
    - Installs core package + backend dependencies
    - Mounts code for hot-reload
    - Port 8000

  frontend:
    - Node 20
    - Vite dev server
    - Hot-reload enabled
    - Port 5173
```

### Environment Variables

**Backend**: No env vars needed (configured in config.py)

**Frontend** (`.env`):
```
VITE_API_URL=http://localhost:8000
```

## Performance Characteristics

### Response Times (Apple Silicon M1)

| Difficulty | MCTS    | TSS         | Endgame     | Typical Time | Max Time |
|------------|---------|-------------|-------------|--------------|----------|
| Easy       | 64 sims | Disabled    | Disabled    | 100ms       | 200ms    |
| Medium     | 128 sims| Depth 4     | ≤14 empties | 300ms       | 600ms    |
| Hard       | 256 sims| Depth 6     | ≤20 empties | 650ms       | 1200ms   |

**Notes:**
- TSS forced moves are very fast (<100ms) when found
- Endgame solver adds 50-200ms but guarantees optimal play
- MCTS scales linearly with simulation count

### Memory Usage

- **Backend Container**: ~1-2GB (model + MCTS tree)
- **Frontend Container**: ~100-200MB (Node + Vite)
- **Total**: ~1.5-2.5GB for full stack

## Testing the System

### Quick Test

```bash
# 1. Start everything
docker-compose up --build

# 2. Health check
curl http://localhost:8000/health

# 3. Open browser
open http://localhost:5173

# 4. Play a game and check:
# - Board rendering
# - Move placement
# - AI response
# - Timer updates
# - Debug panel metrics
```

### Testing Each Difficulty

1. **Easy Mode**: Should respond in ~100ms, no TSS/endgame messages
2. **Medium Mode**: Watch for TSS forced moves in tactical positions
3. **Hard Mode**: Deeper search, endgame solver activates in late game

### Debug Panel Validation

- Simulations match difficulty setting
- Thinking time is reasonable
- Top moves show probabilities and visits
- Policy heatmap displays correctly
- Value estimate changes based on position

## Future Enhancements

Potential improvements:

1. **Move History**: Track and display all moves
2. **Undo**: Allow taking back moves
3. **Analysis Mode**: Review games with AI suggestions
4. **Opening Book**: Integrate opening database
5. **Difficulty Tuning**: More granular control
6. **Sound Effects**: Stone placement audio
7. **Animations**: Smooth stone placement
8. **Mobile Support**: Touch-friendly board
9. **Multiplayer**: Human vs human mode
10. **Game Recording**: Save/load games

## Troubleshooting

See `docs/QUICKSTART.md` for common issues and solutions.