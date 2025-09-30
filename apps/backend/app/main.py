"""FastAPI application for Gomoku game server."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import settings
from .models import (
    CreateGameRequest,
    CreateGameResponse,
    MoveRequest,
    MoveResponse,
    ResignResponse,
    Move,
)
from .game_manager import GameManager
from .inference import InferenceEngine


# Global state
game_manager = GameManager()
inference_engine: InferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global inference_engine

    # Startup: Load model
    print("Loading AI model...")
    inference_engine = InferenceEngine(settings.MODEL_PATH)
    print("Model loaded successfully!")

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down...")


app = FastAPI(
    title="Gomoku API",
    description="API for playing Gomoku against AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": inference_engine is not None}


@app.post("/api/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest):
    """Create a new game."""
    game = game_manager.create_game(request.difficulty, request.player_color)

    # If AI goes first (player is white)
    ai_move = None
    debug_info = None

    if request.player_color == -1:
        game.start_ai_timer()
        (row, col), debug_info = inference_engine.get_ai_move(
            game.board, game.current_player, game.difficulty
        )
        game.make_move(row, col, game.ai_color)
        game.stop_ai_timer()
        ai_move = Move(row=row, col=col, player=game.ai_color)

    return CreateGameResponse(
        game_id=game.game_id,
        board_size=game.board_size,
        difficulty=game.difficulty,
        player_color=game.player_color,
        current_player=game.current_player,
        status=game.status,
        ai_move=ai_move,
        debug_info=debug_info,
    )


@app.get("/api/games/{game_id}")
async def get_game_state(game_id: str):
    """Get current game state."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    return game.to_dict()


@app.post("/api/games/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, move_request: MoveRequest):
    """Make a player move and get AI response."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    if game.status != "in_progress":
        raise HTTPException(status_code=400, detail="Game is not in progress")

    # Validate it's player's turn
    if game.current_player != game.player_color:
        raise HTTPException(status_code=400, detail="Not player's turn")

    # Make player move
    game.start_player_timer()
    if not game.make_move(move_request.row, move_request.col, game.player_color):
        game.stop_player_timer()
        raise HTTPException(status_code=400, detail="Invalid move")
    game.stop_player_timer()

    player_move = Move(
        row=move_request.row, col=move_request.col, player=game.player_color
    )

    # Check if game ended
    if game.status != "in_progress":
        return MoveResponse(
            player_move=player_move,
            ai_move=None,
            board=game.board.tolist(),
            current_player=game.current_player,
            status=game.status,
            move_count=game.move_count,
            player_time=game.player_time,
            ai_time=game.ai_time,
            debug_info=None,
        )

    # Get AI move
    game.start_ai_timer()
    (ai_row, ai_col), debug_info = inference_engine.get_ai_move(
        game.board, game.current_player, game.difficulty
    )
    game.make_move(ai_row, ai_col, game.ai_color)
    game.stop_ai_timer()

    ai_move = Move(row=ai_row, col=ai_col, player=game.ai_color)

    return MoveResponse(
        player_move=player_move,
        ai_move=ai_move,
        board=game.board.tolist(),
        current_player=game.current_player,
        status=game.status,
        move_count=game.move_count,
        player_time=game.player_time,
        ai_time=game.ai_time,
        debug_info=debug_info,
    )


@app.post("/api/games/{game_id}/resign", response_model=ResignResponse)
async def resign_game(game_id: str):
    """Player resigns the game."""
    game = game_manager.get_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    if game.status != "in_progress":
        raise HTTPException(status_code=400, detail="Game is not in progress")

    game.resign()

    return ResignResponse(game_id=game.game_id, status=game.status, winner=game.ai_color)


@app.delete("/api/games/{game_id}")
async def delete_game(game_id: str):
    """Delete a game."""
    if not game_manager.delete_game(game_id):
        raise HTTPException(status_code=404, detail="Game not found")

    return {"message": "Game deleted successfully"}