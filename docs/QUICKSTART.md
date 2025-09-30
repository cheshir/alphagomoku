# Gomoku AI - Quick Start Guide

Get up and running with the Gomoku AI game in 5 minutes!

## Prerequisites

✅ Docker and Docker Compose installed
✅ Trained model at `./checkpoints/model_best.pt`

## Step 1: Start the Application

```bash
# From project root
docker-compose up --build
```

Wait for both services to start. You should see:
```
backend_1   | Model loaded successfully!
backend_1   | INFO:     Uvicorn running on http://0.0.0.0:8000
frontend_1  | VITE ready in XXX ms
frontend_1  | ➜  Local:   http://localhost:5173/
```

## Step 2: Open the Game

Open your browser and navigate to:
```
http://localhost:5173
```

## Step 3: Play!

1. **Select Difficulty**:
   - Easy: Fast AI (64 simulations, ~100ms) - MCTS only
   - Medium: Balanced (128 sims + TSS + Endgame, ~200-500ms)
   - Hard: Strong AI (256 sims + deeper TSS + Endgame, ~500ms-1s)

2. **Choose Color**:
   - Black: You go first
   - White: AI goes first

3. **Click "Start Game"**

4. **Make Moves**:
   - Click on board intersections to place stones
   - AI responds automatically
   - Watch the timer and debug panel

## Features to Try

### 🎮 Game Controls
- **Resign**: Give up the current game
- **New Game**: Start over with different settings

### ⏱️ Timers
- See how much time you and AI have used
- Active player's timer is highlighted

### 🔍 Debug Panel (Click to Expand)
- **Search Metrics**: Simulations, thinking time, nodes explored
- **Value Estimate**: AI's evaluation of the position
- **Top Moves**: Best alternatives the AI considered
- **Policy Heatmap**: Probability distribution across the board

### 🎯 Visual Feedback
- Last move is marked with a dot
- Hover preview shows where your stone will be placed
- AI thinking indicator during AI's turn
- Game over messages

## AI Search Stack

The AI uses a **unified search stack** for optimal play:

### Easy Mode
- **MCTS only**: 64 simulations (~100ms)
- TSS: Disabled
- Endgame Solver: Disabled
- Best for beginners and fast play

### Medium Mode
- **MCTS**: 128 simulations
- **TSS**: Depth 4, detects forced tactical sequences
- **Endgame Solver**: Activates at ≤14 empty cells
- Balanced strength (~200-500ms)

### Hard Mode
- **MCTS**: 256 simulations
- **TSS**: Depth 6, deep tactical analysis
- **Endgame Solver**: Activates at ≤20 empty cells
- Maximum strength (~500ms-1s)

## Adjusting Difficulty

Want to change AI strength? Edit `apps/backend/app/config.py`:

```python
DIFFICULTIES = {
    "easy": DifficultyConfig(
        simulations=64,
        use_tss=False,              # No TSS for easy mode
        use_endgame_solver=False,   # No endgame solver
    ),
    "medium": DifficultyConfig(
        simulations=128,
        use_tss=True,               # Enable TSS
        tss_depth=4,
        tss_time_cap_ms=100,
        use_endgame_solver=True,    # Enable endgame solver
        endgame_threshold=14,       # Activate at ≤14 empty cells
    ),
}
```

Then restart:
```bash
docker-compose restart backend
```

## Troubleshooting

### Backend won't start
```bash
# Check if model exists
ls -la checkpoints/model_best.pt

# View backend logs
docker-compose logs backend
```

### Frontend won't connect
```bash
# Verify backend is healthy
curl http://localhost:8000/health

# Should return: {"status":"healthy","model_loaded":true}
```

### AI is too slow
- Use lower difficulty (Easy)
- Reduce simulations in config
- Check CPU usage: `docker stats`

### Port already in use
```bash
# Find what's using the port
lsof -i :8000  # or :5173

# Kill the process or change ports in docker-compose.yml
```

## Stopping the Application

```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Next Steps

- Read `DEPLOYMENT.md` for detailed configuration
- Check `API.md` for API documentation
- Adjust difficulty settings for your hardware
- Train your model longer for stronger play!

## Quick Tips

💡 **Better AI**: Train for 200+ epochs with 500 games per epoch
💡 **Faster Response**: Lower simulation counts in config
💡 **Debug Info**: Expand the debug panel to see AI's thinking
💡 **Best Move**: Red bars in debug panel show AI's top choices
💡 **Tactical Play**: Watch for TSS forced moves in medium/hard mode
💡 **Endgame**: Solver activates in late game for perfect play

Have fun playing! 🎮