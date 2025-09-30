# Gomoku AI - Deployment Guide

Complete guide for running the Gomoku AI application with backend and frontend.

## Project Structure

```
alpahgomoku/
├── apps/
│   ├── backend/          # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── game_manager.py
│   │   │   ├── inference.py
│   │   │   └── models.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── frontend/         # Vue 3 frontend
│       ├── src/
│       │   ├── components/
│       │   ├── stores/
│       │   ├── api/
│       │   └── types/
│       ├── Dockerfile
│       └── package.json
├── alphagomoku/          # Core ML package
├── checkpoints/          # Trained models
├── docker-compose.yml
└── docs/
```

## Prerequisites

- Docker and Docker Compose
- Trained model checkpoint at `./checkpoints/model_best.pt`

## Quick Start with Docker

### 1. Build and Start Services

```bash
# From project root
docker-compose up --build
```

This will:
- Build the backend container with FastAPI
- Build the frontend container with Vue 3 + Vite
- Start both services with hot-reload

### 2. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Stop Services

```bash
docker-compose down
```

## Configuration

### Backend Configuration

Edit `apps/backend/app/config.py` to adjust:

```python
# Difficulty settings (MCTS simulations)
DIFFICULTIES = {
    "easy": DifficultyConfig(simulations=64, temperature=0.2),
    "medium": DifficultyConfig(simulations=128, temperature=0.0),
    "hard": DifficultyConfig(simulations=256, temperature=0.0),
}

# Model path
MODEL_PATH = "./checkpoints/model_best.pt"

# MCTS batch size
BATCH_SIZE = 32
```

### Frontend Configuration

Edit `apps/frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000
```

## Running Without Docker

### Backend

```bash
cd apps/backend

# Install dependencies
pip install -r requirements.txt
pip install -r ../../requirements.txt

# Run server
python run.py
```

### Frontend

```bash
cd apps/frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

## Model Requirements

Ensure you have a trained model checkpoint:

```bash
./checkpoints/model_best.pt
```

The checkpoint should contain:
```python
{
    'model_state_dict': {...},  # or just the state dict directly
    'epoch': N,
    # ... other training metadata
}
```

## Performance Tuning

### Adjusting Difficulty

Edit simulation counts in `apps/backend/app/config.py`:

| Difficulty | Simulations | Expected Response Time |
|------------|-------------|------------------------|
| Easy       | 32-64       | ~100ms                |
| Medium     | 128-256     | ~200-500ms            |
| Hard       | 256-512     | ~500ms-1s             |

Lower simulations = faster but weaker AI
Higher simulations = slower but stronger AI

### MCTS Batch Size

Adjust `BATCH_SIZE` in config:
- Larger batch = better GPU utilization but more memory
- Smaller batch = less memory but slower
- Recommended: 16-32 for CPU, 32-64 for MPS/CUDA

### Backend Device

The backend automatically selects:
1. MPS (Apple Silicon)
2. CUDA (NVIDIA GPUs)
3. CPU (fallback)

To force CPU mode:
```python
# In apps/backend/app/inference.py
self.device = torch.device("cpu")
```

## Troubleshooting

### Backend Issues

**Model not loading:**
```bash
# Check model path
ls -la checkpoints/model_best.pt

# Verify in logs
docker-compose logs backend
```

**MCTS too slow:**
- Reduce simulation count in config
- Reduce batch size
- Check device being used (should see "Using device: mps" or "cuda")

**Out of memory:**
- Reduce MCTS simulations
- Reduce batch size
- Check system resources: `docker stats`

### Frontend Issues

**Cannot connect to backend:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check CORS settings in backend config
- Verify VITE_API_URL in frontend .env

**Build errors:**
```bash
# Clean and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Docker Issues

**Port conflicts:**
```bash
# Check what's using ports 8000 or 5173
lsof -i :8000
lsof -i :5173

# Change ports in docker-compose.yml if needed
```

**Permission issues:**
```bash
# Fix checkpoint permissions
chmod -R 755 checkpoints/
```

## Production Deployment

### Build Production Frontend

```bash
cd apps/frontend
npm run build
```

Static files will be in `apps/frontend/dist/`

### Production Backend

```bash
cd apps/backend

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Production Setup

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ..
      dockerfile: ../apps/backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./checkpoints:/app/checkpoints:ro
    restart: unless-stopped

  frontend:
    build:
      context: ../apps/frontend
      dockerfile: Dockerfile.prod
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend
```

### Health Check

```bash
# Backend health
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","model_loaded":true}
```

### Performance Metrics

Check AI response times in debug panel:
- Easy: should be < 150ms
- Medium: should be < 500ms
- Hard: should be < 1.5s

If slower, reduce simulation counts or check system resources.

## API Testing

### Create Game
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

See `docs/API.md` for complete API documentation.

## Next Steps

1. Test with your trained model
2. Adjust difficulty settings for desired performance
3. Play games and monitor debug metrics
4. Fine-tune simulation counts based on hardware
5. Consider training longer for stronger play

Enjoy playing against your AlphaZero-style Gomoku AI! 🎮