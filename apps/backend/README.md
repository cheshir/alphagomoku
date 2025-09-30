# Gomoku API Backend

FastAPI backend for playing Gomoku against AI with three difficulty levels.

## Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Ensure model checkpoint exists:**
```bash
# Make sure you have a trained model at:
# ../checkpoints/model_best.pt
```

3. **Configure settings (optional):**
Edit `app/config.py` to adjust:
- Model path
- Difficulty configurations (simulations: easy=64, medium=128, hard=256)
- CORS origins
- Server host/port

## Running the Server

```bash
cd backend
python run.py
```

Server will start at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Create New Game
```bash
curl -X POST http://localhost:8000/api/games \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium", "player_color": 1}'
```

### Make Move
```bash
curl -X POST http://localhost:8000/api/games/{game_id}/move \
  -H "Content-Type: application/json" \
  -d '{"row": 7, "col": 7}'
```

### Get Game State
```bash
curl http://localhost:8000/api/games/{game_id}
```

### Resign
```bash
curl -X POST http://localhost:8000/api/games/{game_id}/resign
```

## Configuration

### Unified Search Stack

The backend uses a three-tier search system:

1. **Endgame Solver** - Exact alpha-beta for late game (Medium: ≤14 empties, Hard: ≤20)
2. **TSS** - Forced tactical sequences (Medium: depth 4, Hard: depth 6)
3. **MCTS** - Neural network guided search (Easy: 64, Medium: 128, Hard: 256 sims)

### Difficulty Settings

Edit `app/config.py`:

```python
DIFFICULTIES = {
    "easy": DifficultyConfig(
        simulations=64,
        use_tss=False,              # TSS disabled for easy mode
        use_endgame_solver=False,   # Endgame solver disabled
    ),
    "medium": DifficultyConfig(
        simulations=128,
        use_tss=True,               # TSS enabled
        tss_depth=4,
        tss_time_cap_ms=100,
        use_endgame_solver=True,    # Endgame solver enabled
        endgame_threshold=14,       # Activate at ≤14 empty cells
    ),
    "hard": DifficultyConfig(
        simulations=256,
        use_tss=True,               # Deeper TSS
        tss_depth=6,
        tss_time_cap_ms=300,
        use_endgame_solver=True,    # Earlier endgame activation
        endgame_threshold=20,       # Activate at ≤20 empty cells
    ),
}
```

See [../../docs/API.md](../../docs/API.md) for full documentation.

## Development

Run with auto-reload:
```bash
python run.py
```

The server will automatically reload when you modify code files.