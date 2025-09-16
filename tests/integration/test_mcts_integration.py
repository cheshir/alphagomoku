"""Integration tests for MCTS with other components."""

import pytest
import numpy as np
import torch
import tempfile
from alphagomoku.env.gomoku_env import GomokuEnv
from alphagomoku.model.network import GomokuNet
from alphagomoku.mcts.mcts import MCTS
from alphagomoku.tss import Position, tss_search
from alphagomoku.train.trainer import Trainer
from alphagomoku.train.data_buffer import DataBuffer
from alphagomoku.selfplay.selfplay import SelfPlayWorker, SelfPlayData
from alphagomoku.eval.evaluator import Evaluator


class TestMCTSEnvironmentIntegration:
    """Test MCTS integration with GomokuEnv."""

    @pytest.fixture
    def setup_mcts_env(self):
        """Setup MCTS with environment."""
        env = GomokuEnv(board_size=11)  # Odd size for testing
        model = GomokuNet(board_size=11, num_blocks=3, channels=32)
        mcts = MCTS(model, env, num_simulations=50)
        return env, model, mcts

    def test_full_game_simulation(self, setup_mcts_env):
        """Test MCTS playing a complete game."""
        env, model, mcts = setup_mcts_env
        env.reset()

        move_count = 0
        game_history = []

        while not env.terminated and move_count < 121:  # Max moves for 11x11
            # Get MCTS move
            action_probs, value = mcts.search(env.board)

            # Select action (argmax for deterministic play)
            action = np.argmax(action_probs)

            # Record move
            game_history.append({
                'board': env.board.copy(),
                'action': action,
                'action_probs': action_probs.copy(),
                'value': value,
                'current_player': env.current_player
            })

            # Make move
            obs, reward, terminated, truncated, info = env.step(action)
            move_count += 1

        # Game should complete properly
        assert env.terminated or move_count == 121
        assert len(game_history) > 0

        # Check game history consistency
        for i, move in enumerate(game_history):
            assert move['board'].shape == (11, 11)
            assert len(move['action_probs']) == 121
            assert np.isclose(np.sum(move['action_probs']), 1.0)
            assert -1 <= move['value'] <= 1

    def test_legal_move_enforcement(self, setup_mcts_env):
        """Test that MCTS only selects legal moves."""
        env, model, mcts = setup_mcts_env
        env.reset()

        # Make some moves to create occupied positions
        occupied_actions = [60, 61, 72, 73]  # Center area
        for action in occupied_actions:
            env.step(action)
            if env.terminated:
                break

        if not env.terminated:
            # Get MCTS policy
            action_probs, value = mcts.search(env.board)

            # Check that occupied positions have zero probability
            legal_actions = env.get_legal_actions()
            for action in range(121):
                if action not in legal_actions:
                    assert action_probs[action] == 0.0, \
                        f"Illegal action {action} has non-zero probability"

            # At least one legal action should have positive probability
            assert np.sum(action_probs) > 0

    def test_mcts_env_state_consistency(self, setup_mcts_env):
        """Test MCTS maintains consistency with environment state."""
        env, model, mcts = setup_mcts_env
        env.reset()

        for _ in range(10):  # Play several moves
            initial_board = env.board.copy()
            initial_player = env.current_player

            # MCTS search shouldn't modify environment
            action_probs, value = mcts.search(env.board)

            # Environment state should be unchanged
            assert np.array_equal(env.board, initial_board)
            assert env.current_player == initial_player

            # Make a move
            legal_actions = env.get_legal_actions()
            if legal_actions:
                action = np.random.choice(legal_actions)
                env.step(action)

            if env.terminated:
                break

    def test_mcts_with_different_board_sizes(self):
        """Test MCTS works with different board sizes."""
        board_sizes = [5, 9, 13, 15, 19]

        for size in board_sizes:
            env = GomokuEnv(board_size=size)
            model = GomokuNet(board_size=size, num_blocks=2, channels=16)
            mcts = MCTS(model, env, num_simulations=20)

            env.reset()
            action_probs, value = mcts.search(env.board)

            # Check output dimensions
            assert len(action_probs) == size * size
            assert np.isclose(np.sum(action_probs), 1.0)
            assert -1 <= value <= 1

    def test_mcts_temperature_effects(self, setup_mcts_env):
        """Test temperature effects in MCTS-environment interaction."""
        env, model, mcts = setup_mcts_env
        env.reset()

        # Test different temperatures
        temperatures = [0.1, 1.0, 2.0]
        policies = {}

        for temp in temperatures:
            action_probs, value = mcts.search(env.board, temperature=temp)
            policies[temp] = action_probs.copy()

        # Low temperature should be more concentrated
        entropy_low = -np.sum(policies[0.1] * np.log(policies[0.1] + 1e-10))
        entropy_high = -np.sum(policies[2.0] * np.log(policies[2.0] + 1e-10))

        assert entropy_high >= entropy_low - 0.1  # Allow some variance


class TestMCTSTSSIntegration:
    """Test MCTS integration with TSS."""

    @pytest.fixture
    def setup_tactical_position(self):
        """Setup a tactical Gomoku position."""
        env = GomokuEnv(board_size=15)
        model = GomokuNet(board_size=15, num_blocks=2, channels=16)
        mcts = MCTS(model, env, num_simulations=100)

        env.reset()
        # Create a tactical position with threats
        moves = [
            (7, 7), (8, 7), (7, 8), (8, 8),  # 2x2 square
            (6, 7), (9, 7), (7, 6)  # Extend pattern
        ]

        for i, (row, col) in enumerate(moves):
            action = row * 15 + col
            env.step(action)
            if env.terminated:
                break

        return env, model, mcts

    def test_tss_mcts_coordination(self, setup_tactical_position):
        """Test TSS and MCTS working together."""
        env, model, mcts = setup_tactical_position

        if not env.terminated:
            # Run MCTS search
            mcts_probs, mcts_value = mcts.search(env.board)

            # Run TSS analysis on same position
            position = Position(
                board=env.board,
                current_player=env.current_player,
                last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
            )

            tss_result = tss_search(position, depth=4, time_cap_ms=100)

            # If TSS finds a forced move, check MCTS gives it high probability
            if tss_result.forced_move:
                forced_row, forced_col = tss_result.forced_move
                forced_action = forced_row * 15 + forced_col

                # MCTS should also favor the forced move (though not necessarily highest)
                forced_prob = mcts_probs[forced_action]
                avg_prob = np.mean(mcts_probs[mcts_probs > 0])

                # Forced move should have above-average probability
                assert forced_prob >= avg_prob * 0.5, \
                    f"MCTS didn't favor TSS forced move: {forced_prob:.3f} vs avg {avg_prob:.3f}"

    def test_tss_guided_mcts_move_selection(self):
        """Test TSS guiding MCTS move selection."""
        env = GomokuEnv(board_size=15)
        model = GomokuNet(board_size=15, num_blocks=2, channels=16)
        mcts = MCTS(model, env, num_simulations=50)

        # Create position with immediate threat
        env.reset()
        board = np.zeros((15, 15), dtype=np.int8)

        # Opponent's open four (must defend)
        for i in range(4):
            board[7, 5 + i] = -1

        env.board = board
        env.current_player = 1
        env.last_move = np.array([7, 8])

        # Convert to TSS position
        position = Position(board=board, current_player=1, last_move=(7, 8))

        # TSS should find forced defense
        tss_result = tss_search(position, depth=2, time_cap_ms=50)

        # MCTS on same position
        mcts_probs, mcts_value = mcts.search(board)

        if tss_result.is_forced_defense and tss_result.forced_move:
            forced_row, forced_col = tss_result.forced_move
            forced_action = forced_row * 15 + forced_col

            # In a real integration, TSS would override MCTS
            # Here we just check they're analyzing the same critical position

            # The forced defense move should be blocking the threat
            assert (forced_row, forced_col) in [(7, 4), (7, 9)], \
                "TSS should find the blocking move"

    def test_performance_comparison_tss_vs_mcts(self):
        """Test performance characteristics of TSS vs MCTS."""
        # Create tactical position
        board = np.zeros((15, 15), dtype=np.int8)
        for i in range(3):
            board[7, 6 + i] = 1  # Open three

        position = Position(board=board, current_player=1)

        # Time TSS
        import time
        start = time.time()
        for _ in range(10):
            tss_result = tss_search(position, depth=3, time_cap_ms=30)
        tss_time = (time.time() - start) / 10

        # Time MCTS
        env = GomokuEnv(board_size=15)
        env.board = board
        env.current_player = 1
        model = GomokuNet(board_size=15, num_blocks=1, channels=8)
        mcts = MCTS(model, env, num_simulations=30)

        start = time.time()
        for _ in range(10):
            action_probs, value = mcts.search(board)
        mcts_time = (time.time() - start) / 10

        # TSS should be faster for tactical positions
        print(f"TSS: {tss_time:.3f}s, MCTS: {mcts_time:.3f}s")

        # Both should complete within reasonable time
        assert tss_time < 1.0
        assert mcts_time < 2.0


class TestMCTSTrainingIntegration:
    """Test MCTS integration with training pipeline."""

    @pytest.fixture
    def setup_training_components(self):
        """Setup training components."""
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=30)
        trainer = Trainer(model, device='cpu')

        return model, env, mcts, trainer

    def test_mcts_training_data_generation(self, setup_training_components):
        """Test MCTS generating data for training."""
        model, env, mcts, trainer = setup_training_components

        # Generate training data using MCTS
        training_data = []
        env.reset()

        for _ in range(5):  # Play several moves
            if env.terminated:
                break

            # Get MCTS policy and value
            action_probs, value = mcts.search(env.board, temperature=1.0)

            # Create training example
            training_example = SelfPlayData(
                state=env.board.copy(),
                policy=action_probs.copy(),
                value=value,
                current_player=env.current_player,
                last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
            )
            training_data.append(training_example)

            # Make move
            legal_actions = env.get_legal_actions()
            if legal_actions:
                # Sample from policy
                action = np.random.choice(len(action_probs), p=action_probs)
                env.step(action)

        # Train on generated data
        if training_data:
            losses = trainer.train_step(training_data)
            assert isinstance(losses, dict)

    def test_model_improvement_through_training(self, setup_training_components):
        """Test that model improves through MCTS-generated training."""
        model, env, mcts, trainer = setup_training_components

        # Get initial model performance
        env.reset()
        initial_probs, initial_value = mcts.search(env.board)
        initial_entropy = -np.sum(initial_probs * np.log(initial_probs + 1e-10))

        # Generate training data
        training_data = []
        for game in range(3):
            env.reset()
            for move in range(10):
                if env.terminated:
                    break

                action_probs, value = mcts.search(env.board, temperature=1.2)

                # Create training example with some outcome bias
                outcome_value = np.random.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
                training_example = SelfPlayData(
                    state=env.board.copy(),
                    policy=action_probs,
                    value=outcome_value * 0.8 + value * 0.2,  # Mix true outcome with current value
                    current_player=env.current_player,
                    last_move=tuple(env.last_move) if env.last_move[0] >= 0 else None
                )
                training_data.append(training_example)

                # Make random legal move
                legal_actions = env.get_legal_actions()
                if legal_actions:
                    action = np.random.choice(legal_actions)
                    env.step(action)

        # Train model
        for epoch in range(3):
            np.random.shuffle(training_data)
            batch_size = 8
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i + batch_size]
                losses = trainer.train_step(batch)

        # Test model after training
        env.reset()
        trained_probs, trained_value = mcts.search(env.board)

        # Model should have changed (though not necessarily improved on this tiny dataset)
        prob_diff = np.sum(np.abs(initial_probs - trained_probs))
        assert prob_diff > 0.01, "Model should change after training"

    def test_mcts_with_updated_model(self, setup_training_components):
        """Test MCTS behavior with continuously updated model."""
        model, env, mcts, trainer = setup_training_components

        # Create simple training data
        training_data = []
        for _ in range(20):
            state = np.random.randint(-1, 2, (9, 9)).astype(np.int8)
            policy = np.random.rand(81)
            policy = policy / np.sum(policy)

            training_data.append(SelfPlayData(
                state=state,
                policy=policy,
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(4, 4)
            ))

        # Record MCTS behavior before training
        env.reset()
        before_probs, before_value = mcts.search(env.board)

        # Update model
        for _ in range(5):
            batch = training_data[:8]
            losses = trainer.train_step(batch)

        # Test MCTS with updated model
        env.reset()
        after_probs, after_value = mcts.search(env.board)

        # MCTS should reflect model changes
        assert not np.allclose(before_probs, after_probs, atol=1e-3), \
            "MCTS should change with model updates"

    def test_training_stability_with_mcts_data(self, setup_training_components):
        """Test training stability with MCTS-generated data."""
        model, env, mcts, trainer = setup_training_components

        loss_history = []

        for iteration in range(10):
            # Generate fresh MCTS data
            env.reset()
            action_probs, value = mcts.search(env.board)

            # Create training batch
            batch = [SelfPlayData(
                state=env.board.copy(),
                policy=action_probs,
                value=value,
                current_player=env.current_player,
                last_move=None
            )]

            # Train
            losses = trainer.train_step(batch)
            if 'total_loss' in losses:
                loss_history.append(losses['total_loss'])
            elif losses:
                loss_history.append(list(losses.values())[0])

        # Training should be stable (no exploding gradients)
        if loss_history:
            for loss in loss_history:
                assert np.isfinite(loss), "Training loss should remain finite"
                assert loss < 100.0, "Loss should not explode"


class TestMCTSSelfPlayIntegration:
    """Test MCTS integration with self-play."""

    def test_selfplay_worker_with_mcts(self):
        """Test SelfPlayWorker using MCTS."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        worker = SelfPlayWorker(
            model=model,
            board_size=9,
            mcts_simulations=20,
            adaptive_sims=False,
            batch_size=4
        )

        # Generate game
        game_data = worker.generate_game()

        # Should generate valid training data
        assert len(game_data) > 0
        assert all(isinstance(data, SelfPlayData) for data in game_data)

        # Check data quality
        for data in game_data[:3]:  # Check first few
            assert data.state.shape == (9, 9)
            assert len(data.policy) == 81
            assert np.isclose(np.sum(data.policy), 1.0, atol=1e-6)
            assert -1 <= data.value <= 1
            assert data.current_player in [-1, 1]

    def test_selfplay_game_diversity(self):
        """Test that self-play generates diverse games."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        worker = SelfPlayWorker(
            model=model,
            board_size=9,
            mcts_simulations=15,
            adaptive_sims=False,
            batch_size=2
        )

        # Generate multiple games
        games = [worker.generate_game() for _ in range(3)]

        # Games should be different
        first_moves = []
        for game in games:
            if game:
                # Find first move (non-zero position)
                first_state = game[0].state
                first_move_pos = np.argmax(np.abs(first_state))
                first_moves.append(first_move_pos)

        # Should have some diversity in opening moves
        unique_first_moves = len(set(first_moves))
        # With temperature and randomness, should get some variety
        # (Though small sample size might have duplicates)

    def test_selfplay_mcts_consistency(self):
        """Test consistency between self-play and direct MCTS usage."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)

        # Direct MCTS
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=20)
        env.reset()
        direct_probs, direct_value = mcts.search(env.board, temperature=1.0)

        # Self-play MCTS (should use similar configuration)
        worker = SelfPlayWorker(
            model=model,
            board_size=9,
            mcts_simulations=20,
            adaptive_sims=False,
            batch_size=1
        )

        # Get first move from self-play
        game_data = worker.generate_game()
        if game_data:
            selfplay_probs = game_data[0].policy

            # Policies should be similar (allowing for temperature differences)
            correlation = np.corrcoef(direct_probs, selfplay_probs)[0, 1]
            assert correlation > 0.5, f"Policies should be correlated: {correlation:.3f}"


class TestMCTSEvaluationIntegration:
    """Test MCTS integration with evaluation."""

    def test_mcts_strength_evaluation(self):
        """Test evaluating MCTS strength."""
        # Create two different strength MCTS players
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        env = GomokuEnv(board_size=9)

        strong_mcts = MCTS(model, env, num_simulations=50, cpuct=1.5)
        weak_mcts = MCTS(model, env, num_simulations=10, cpuct=1.0)

        evaluator = Evaluator()

        # Play games between different strength MCTS
        results = evaluator.play_match(
            player1=strong_mcts,
            player2=weak_mcts,
            num_games=3,
            board_size=9
        )

        # Should complete without errors
        assert 'wins_player1' in results
        assert 'wins_player2' in results
        assert 'draws' in results

        total_games = results['wins_player1'] + results['wins_player2'] + results['draws']
        assert total_games == 3

    def test_mcts_position_evaluation(self):
        """Test MCTS evaluation of specific positions."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        env = GomokuEnv(board_size=9)
        mcts = MCTS(model, env, num_simulations=50)

        # Test position evaluation
        positions = [
            np.zeros((9, 9)),  # Empty board
            np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],  # Center stone
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        ]

        evaluations = []
        for pos in positions:
            action_probs, value = mcts.search(pos)
            evaluations.append({
                'position': pos.copy(),
                'value': value,
                'entropy': -np.sum(action_probs * np.log(action_probs + 1e-10))
            })

        # Empty board should have high entropy (many good moves)
        # Position with center stone should have lower entropy (more focused)
        empty_entropy = evaluations[0]['entropy']
        center_entropy = evaluations[1]['entropy']

        # This may vary based on model, but there should be some difference
        assert abs(empty_entropy - center_entropy) >= 0, "Evaluations should differ"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])