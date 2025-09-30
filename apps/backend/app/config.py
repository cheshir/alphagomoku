"""Configuration for the Gomoku API server."""

from typing import Dict
from pydantic import BaseModel


class DifficultyConfig(BaseModel):
    """Configuration for a difficulty level."""
    simulations: int
    temperature: float = 0.0
    cpuct: float = 1.8
    # TSS settings
    use_tss: bool = False
    tss_depth: int = 0
    tss_time_cap_ms: int = 0
    # Endgame solver settings
    use_endgame_solver: bool = False
    endgame_threshold: int = 0  # Activate when <= this many empty cells


class Settings:
    """Application settings."""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Game settings
    BOARD_SIZE: int = 15

    # Model settings
    MODEL_PATH: str = "./model.pt"

    # Difficulty configurations (easily configurable)
    DIFFICULTIES: Dict[str, DifficultyConfig] = {
        "easy": DifficultyConfig(
            simulations=64,
            temperature=0.2,
            use_tss=False,  # TSS disabled for easy mode
            tss_depth=0,
            tss_time_cap_ms=0,
            use_endgame_solver=False,  # Endgame solver disabled for easy mode
            endgame_threshold=0,
        ),
        "medium": DifficultyConfig(
            simulations=128,
            temperature=0.0,
            use_tss=True,  # TSS enabled for medium
            tss_depth=4,
            tss_time_cap_ms=100,
            use_endgame_solver=True,  # Endgame solver enabled
            endgame_threshold=14,  # Activate at ≤14 empty cells
        ),
        "hard": DifficultyConfig(
            simulations=256,
            temperature=0.0,
            use_tss=True,  # TSS enabled for hard
            tss_depth=6,
            tss_time_cap_ms=300,
            use_endgame_solver=True,  # Endgame solver enabled
            endgame_threshold=20,  # Activate at ≤20 empty cells
        ),
    }

    # MCTS settings
    BATCH_SIZE: int = 32

    # CORS settings - allow all origins
    CORS_ORIGINS: list = ["*"]


settings = Settings()