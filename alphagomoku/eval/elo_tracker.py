"""Elo rating tracker for monitoring model strength over training epochs."""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class EloRecord:
    """Single Elo rating record"""
    epoch: int
    elo: float
    games_played: int
    wins: int
    losses: int
    draws: int
    opponent_elo: float


class EloTracker:
    """Track model Elo rating over training epochs.

    Uses standard Elo formula:
    - Expected score: E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
    - Rating update: new_elo = old_elo + K * (actual_score - expected_score)

    Example:
        >>> tracker = EloTracker(initial_elo=1500)
        >>> new_elo = tracker.update(epoch=10, wins=35, losses=15, draws=0, opponent_elo=1400)
        >>> print(f"New Elo: {new_elo:.0f}")
    """

    def __init__(
        self,
        initial_elo: int = 1500,
        k_factor: int = 32,
        save_path: Optional[str] = None,
    ):
        """Initialize Elo tracker.

        Args:
            initial_elo: Starting Elo rating
            k_factor: K-factor for rating updates (higher = more volatile)
            save_path: Optional path to save Elo history
        """
        self.initial_elo = initial_elo
        self.current_elo = initial_elo
        self.k_factor = k_factor
        self.save_path = Path(save_path) if save_path else None
        self.history: List[EloRecord] = []

        # Load existing history if available
        if self.save_path and self.save_path.exists():
            self._load_history()

    def update(
        self,
        epoch: int,
        wins: int,
        losses: int,
        draws: int,
        opponent_elo: float,
    ) -> float:
        """Update Elo rating based on game results.

        Args:
            epoch: Training epoch
            wins: Number of wins
            losses: Number of losses
            draws: Number of draws
            opponent_elo: Opponent's Elo rating

        Returns:
            New Elo rating
        """
        total_games = wins + losses + draws

        if total_games == 0:
            return self.current_elo

        # Calculate actual score (1 = win, 0.5 = draw, 0 = loss)
        actual_score = (wins + 0.5 * draws) / total_games

        # Calculate expected score using Elo formula
        expected_score = 1.0 / (1.0 + math.pow(10, (opponent_elo - self.current_elo) / 400.0))

        # Update rating
        rating_change = self.k_factor * (actual_score - expected_score)
        new_elo = self.current_elo + rating_change

        # Record
        record = EloRecord(
            epoch=epoch,
            elo=new_elo,
            games_played=total_games,
            wins=wins,
            losses=losses,
            draws=draws,
            opponent_elo=opponent_elo,
        )
        self.history.append(record)
        self.current_elo = new_elo

        # Save if path provided
        if self.save_path:
            self._save_history()

        return new_elo

    def get_current_elo(self) -> float:
        """Get current Elo rating"""
        return self.current_elo

    def get_history(self) -> List[EloRecord]:
        """Get full Elo history"""
        return self.history

    def get_elo_at_epoch(self, epoch: int) -> Optional[float]:
        """Get Elo rating at specific epoch"""
        for record in self.history:
            if record.epoch == epoch:
                return record.elo
        return None

    def get_rating_change(self, start_epoch: int, end_epoch: int) -> Optional[float]:
        """Calculate Elo change between two epochs"""
        start_elo = self.get_elo_at_epoch(start_epoch)
        end_elo = self.get_elo_at_epoch(end_epoch)

        if start_elo is None or end_elo is None:
            return None

        return end_elo - start_elo

    def _save_history(self):
        """Save Elo history to JSON file"""
        if not self.save_path:
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "initial_elo": self.initial_elo,
            "current_elo": self.current_elo,
            "k_factor": self.k_factor,
            "history": [asdict(record) for record in self.history],
        }

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_history(self):
        """Load Elo history from JSON file"""
        if not self.save_path or not self.save_path.exists():
            return

        with open(self.save_path, "r") as f:
            data = json.load(f)

        self.initial_elo = data["initial_elo"]
        self.current_elo = data["current_elo"]
        self.k_factor = data["k_factor"]
        self.history = [EloRecord(**record) for record in data["history"]]

    def print_summary(self):
        """Print Elo history summary"""
        if not self.history:
            print("No Elo history recorded.")
            return

        print(f"\n{'='*60}")
        print(f"Elo Rating History")
        print(f"{'='*60}")
        print(f"Initial Elo: {self.initial_elo:.0f}")
        print(f"Current Elo: {self.current_elo:.0f}")
        print(f"Total change: {self.current_elo - self.initial_elo:+.0f}")
        print(f"\nRecent history:")
        print(f"{'Epoch':<8} {'Elo':<8} {'Change':<10} {'W-L-D':<12} {'Win Rate'}")
        print(f"{'-'*60}")

        for i, record in enumerate(self.history[-10:]):  # Show last 10
            prev_elo = self.history[i - 1].elo if i > 0 else self.initial_elo
            change = record.elo - prev_elo
            total = record.wins + record.losses + record.draws
            win_rate = record.wins / total if total > 0 else 0

            print(
                f"{record.epoch:<8} {record.elo:<8.0f} {change:+9.0f}  "
                f"{record.wins:2d}-{record.losses:2d}-{record.draws:2d}     "
                f"{win_rate:.1%}"
            )

        print(f"{'='*60}\n")
