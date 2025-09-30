# Gomoku Frontend

Modern Vue 3 + TypeScript frontend for playing Gomoku against AI.

## Features

- 🎨 Beautiful modern wood-themed board
- 🤖 Three difficulty levels (Easy/Medium/Hard)
- ⏱️ Real-time timers for both players
- 🔍 Collapsible debug panel with AI metrics
- 📊 Visual policy distribution heatmap
- 🎯 Last move indicator
- 💭 AI thinking indicator

## Tech Stack

- Vue 3 with Composition API
- TypeScript
- Pinia for state management
- Axios for API calls
- Vite for build tooling

## Development

The frontend runs in Docker via docker-compose. See the root-level docker-compose.yml.

To run locally without Docker:

```bash
cd apps/frontend
npm install
npm run dev
```

Frontend will be available at http://localhost:5173

## Environment Variables

- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)

## Components

- **GomokuBoard**: Main game board with SVG rendering
- **GameControls**: Game setup and controls
- **GameTimer**: Player and AI time tracking
- **DebugPanel**: Collapsible AI metrics and visualization

## State Management

Uses Pinia store (`stores/game.ts`) for:
- Game state management
- API communication
- Move validation
- Timer tracking