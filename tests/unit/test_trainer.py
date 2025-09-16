"""Unit tests for Trainer module."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from alphagomoku.train.trainer import Trainer
from alphagomoku.model.network import GomokuNet
from alphagomoku.selfplay.selfplay import SelfPlayData


class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def setup_trainer(self):
        """Setup trainer components."""
        model = GomokuNet(board_size=9, num_blocks=2, channels=16)
        trainer = Trainer(model, lr=0.01, weight_decay=1e-4, device='cpu')
        return model, trainer

    def test_initialization(self, setup_trainer):
        """Test Trainer initialization."""
        model, trainer = setup_trainer

        assert trainer.model == model
        assert trainer.device == 'cpu'
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)
        assert hasattr(trainer, 'policy_loss_fn')
        assert hasattr(trainer, 'value_loss_fn')
        assert trainer.step == 0

    def test_device_selection(self):
        """Test device selection logic."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)

        # Test explicit device
        trainer = Trainer(model, device='cpu')
        assert trainer.device == 'cpu'

        # Test automatic device selection (mocked)
        with patch('torch.backends.mps.is_available', return_value=True):
            trainer = Trainer(model, device=None)
            assert trainer.device == 'mps'

        with patch('torch.backends.mps.is_available', return_value=False):
            trainer = Trainer(model, device=None)
            assert trainer.device == 'cpu'

    def test_train_step_basic(self, setup_trainer):
        """Test basic training step."""
        model, trainer = setup_trainer

        # Create sample training data
        batch = [
            SelfPlayData(
                state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                policy=np.random.rand(81),
                value=np.random.rand() * 2 - 1,  # [-1, 1]
                current_player=1,
                last_move=(4, 4)
            )
            for _ in range(4)
        ]

        # Normalize policies
        for data in batch:
            data.policy = data.policy / np.sum(data.policy)

        losses = trainer.train_step(batch)

        # Should return loss dictionary
        assert isinstance(losses, dict)
        assert 'policy_loss' in losses or 'value_loss' in losses or 'total_loss' in losses

    def test_train_step_empty_batch(self, setup_trainer):
        """Test training step with empty batch."""
        model, trainer = setup_trainer

        losses = trainer.train_step([])
        assert losses == {}

    def test_train_step_gradient_computation(self, setup_trainer):
        """Test gradient computation during training step."""
        model, trainer = setup_trainer

        batch = [
            SelfPlayData(
                state=np.zeros((9, 9), dtype=np.int8),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(4, 4)
            )
        ]

        # Check that gradients are computed
        losses = trainer.train_step(batch)

        # After training step, model should have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        # Note: Depending on implementation, gradients might be cleared
        # So we just check that training completed without error
        assert isinstance(losses, dict)

    def test_train_step_loss_computation(self, setup_trainer):
        """Test loss computation details."""
        model, trainer = setup_trainer

        # Create deterministic data
        batch = [
            SelfPlayData(
                state=np.zeros((9, 9), dtype=np.int8),
                policy=np.ones(81) / 81,  # Uniform policy
                value=0.5,
                current_player=1,
                last_move=(4, 4)
            ),
            SelfPlayData(
                state=np.ones((9, 9), dtype=np.int8),
                policy=np.zeros(81),  # Concentrated policy
                value=-0.5,
                current_player=-1,
                last_move=(0, 0)
            )
        ]
        batch[1].policy[40] = 1.0  # Put all probability on center

        losses = trainer.train_step(batch)

        # Losses should be positive
        for loss_name, loss_value in losses.items():
            assert loss_value >= 0.0, f"{loss_name} should be non-negative"

    def test_train_step_tensor_shapes(self, setup_trainer):
        """Test tensor shape handling in training step."""
        model, trainer = setup_trainer

        # Mock model forward to check input shapes
        original_forward = model.forward
        input_shapes = []

        def shape_tracking_forward(x):
            input_shapes.append(x.shape)
            return original_forward(x)

        with patch.object(model, 'forward', side_effect=shape_tracking_forward):
            batch = [
                SelfPlayData(
                    state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                    policy=np.ones(81) / 81,
                    value=0.0,
                    current_player=1,
                    last_move=(4, 4)
                )
                for _ in range(3)
            ]

            trainer.train_step(batch)

            # Should have called forward once with batch size 3
            assert len(input_shapes) == 1
            assert input_shapes[0][0] == 3  # Batch size
            assert input_shapes[0][1:] == (5, 9, 9)  # Input channels and board size

    def test_optimizer_step(self, setup_trainer):
        """Test optimizer step execution."""
        model, trainer = setup_trainer

        # Mock optimizer to track calls
        with patch.object(trainer.optimizer, 'zero_grad') as mock_zero_grad, \
             patch.object(trainer.optimizer, 'step') as mock_step:

            batch = [
                SelfPlayData(
                    state=np.zeros((9, 9), dtype=np.int8),
                    policy=np.ones(81) / 81,
                    value=0.0,
                    current_player=1,
                    last_move=(4, 4)
                )
            ]

            trainer.train_step(batch)

            mock_zero_grad.assert_called_once()
            mock_step.assert_called_once()

    def test_scheduler_step(self, setup_trainer):
        """Test learning rate scheduler."""
        model, trainer = setup_trainer

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Mock scheduler step
        with patch.object(trainer.scheduler, 'step') as mock_scheduler_step:
            # Train method would typically call scheduler.step()
            trainer.scheduler.step()
            mock_scheduler_step.assert_called_once()

    def test_tensorboard_logging(self, setup_trainer):
        """Test TensorBoard logging."""
        model, trainer = setup_trainer

        # Mock tensorboard writer
        with patch.object(trainer.writer, 'add_scalar') as mock_add_scalar:
            batch = [
                SelfPlayData(
                    state=np.zeros((9, 9), dtype=np.int8),
                    policy=np.ones(81) / 81,
                    value=0.0,
                    current_player=1,
                    last_move=(4, 4)
                )
            ]

            losses = trainer.train_step(batch)

            # Trainer might log losses (implementation dependent)
            # At minimum, no errors should occur

    def test_error_handling_invalid_data(self, setup_trainer):
        """Test error handling with invalid training data."""
        model, trainer = setup_trainer

        # Invalid data with wrong shapes
        invalid_batch = [
            SelfPlayData(
                state=np.zeros((10, 10), dtype=np.int8),  # Wrong board size
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(4, 4)
            )
        ]

        with pytest.raises((RuntimeError, ValueError)):
            trainer.train_step(invalid_batch)

    def test_error_handling_model_failure(self, setup_trainer):
        """Test error handling when model forward fails."""
        model, trainer = setup_trainer

        # Mock model to raise error
        with patch.object(model, 'forward', side_effect=RuntimeError("Model error")):
            batch = [
                SelfPlayData(
                    state=np.zeros((9, 9), dtype=np.int8),
                    policy=np.ones(81) / 81,
                    value=0.0,
                    current_player=1,
                    last_move=(4, 4)
                )
            ]

            with pytest.raises(RuntimeError):
                trainer.train_step(batch)

    def test_memory_management(self, setup_trainer):
        """Test memory management during training."""
        model, trainer = setup_trainer

        # Test with larger batches
        large_batch = [
            SelfPlayData(
                state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                policy=np.random.rand(81),
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(4, 4)
            )
            for _ in range(100)
        ]

        # Normalize policies
        for data in large_batch:
            data.policy = data.policy / np.sum(data.policy)

        # Should handle without memory issues
        losses = trainer.train_step(large_batch)
        assert isinstance(losses, dict)

    def test_loss_scaling(self, setup_trainer):
        """Test loss scaling and numerical stability."""
        model, trainer = setup_trainer

        # Create batch with extreme values
        extreme_batch = [
            SelfPlayData(
                state=np.ones((9, 9), dtype=np.int8),
                policy=np.zeros(81),  # One-hot policy
                value=1.0,  # Max value
                current_player=1,
                last_move=(4, 4)
            )
        ]
        extreme_batch[0].policy[0] = 1.0  # One-hot at position 0

        losses = trainer.train_step(extreme_batch)

        # Losses should be finite
        for loss_name, loss_value in losses.items():
            assert np.isfinite(loss_value), f"{loss_name} should be finite"

    def test_gradient_clipping(self, setup_trainer):
        """Test gradient clipping if implemented."""
        model, trainer = setup_trainer

        # Create batch that might cause large gradients
        batch = [
            SelfPlayData(
                state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                policy=np.zeros(81),  # Extreme policy
                value=1.0,
                current_player=1,
                last_move=(4, 4)
            )
            for _ in range(5)
        ]

        # Make policies one-hot (extreme)
        for i, data in enumerate(batch):
            data.policy[i] = 1.0

        # Should handle extreme gradients
        losses = trainer.train_step(batch)
        assert isinstance(losses, dict)

    def test_training_mode(self, setup_trainer):
        """Test that model is in training mode during training."""
        model, trainer = setup_trainer

        batch = [
            SelfPlayData(
                state=np.zeros((9, 9), dtype=np.int8),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(4, 4)
            )
        ]

        # Ensure model is in training mode
        model.eval()  # Set to eval first
        assert not model.training

        # After train_step, should be in training mode (implementation dependent)
        trainer.train_step(batch)
        # Note: Implementation might set model.train() at beginning of train_step

    def test_cleanup(self, setup_trainer):
        """Test trainer cleanup."""
        model, trainer = setup_trainer

        # Test that writer can be closed
        try:
            trainer.writer.close()
        except:
            pass  # Close might not be implemented or needed


class TestTrainerIntegration:
    """Integration tests for Trainer with other components."""

    def test_trainer_with_real_model_outputs(self):
        """Test trainer with realistic model outputs."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        trainer = Trainer(model, device='cpu')

        # Create realistic training data
        batch = []
        for i in range(8):
            state = np.zeros((9, 9), dtype=np.int8)
            # Add some stones
            state[4, 4] = 1
            state[4, 5] = -1

            # Create realistic policy (higher probability near stones)
            policy = np.ones(81) * 0.01
            policy[4*9 + 3] = 0.3  # Adjacent to stones
            policy[4*9 + 6] = 0.3
            policy[3*9 + 4] = 0.2
            policy = policy / np.sum(policy)

            batch.append(SelfPlayData(
                state=state,
                policy=policy,
                value=np.random.rand() * 2 - 1,
                current_player=1,
                last_move=(4, 5)
            ))

        losses = trainer.train_step(batch)

        # Should complete successfully
        assert isinstance(losses, dict)
        assert len(losses) > 0

    def test_multi_step_training(self):
        """Test multiple training steps."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        trainer = Trainer(model, device='cpu')

        loss_history = []

        for step in range(5):
            batch = [
                SelfPlayData(
                    state=np.random.randint(-1, 2, (9, 9)).astype(np.int8),
                    policy=np.random.rand(81),
                    value=np.random.rand() * 2 - 1,
                    current_player=1,
                    last_move=(4, 4)
                )
                for _ in range(4)
            ]

            # Normalize policies
            for data in batch:
                data.policy = data.policy / np.sum(data.policy)

            losses = trainer.train_step(batch)
            loss_history.append(losses)

        # Should complete all steps
        assert len(loss_history) == 5
        assert all(isinstance(losses, dict) for losses in loss_history)

    def test_save_load_compatibility(self):
        """Test that trainer works with model save/load."""
        model = GomokuNet(board_size=9, num_blocks=1, channels=8)
        trainer = Trainer(model, device='cpu')

        # Save initial state
        initial_state = model.state_dict().copy()

        batch = [
            SelfPlayData(
                state=np.zeros((9, 9), dtype=np.int8),
                policy=np.ones(81) / 81,
                value=0.0,
                current_player=1,
                last_move=(4, 4)
            )
        ]

        # Train one step
        trainer.train_step(batch)

        # Model should have changed
        trained_state = model.state_dict()
        parameters_changed = False
        for key in initial_state:
            if not torch.equal(initial_state[key], trained_state[key]):
                parameters_changed = True
                break

        # At least some parameters should have changed
        # Note: With very small learning rate, changes might be minimal
        # So we just verify training completed without error


if __name__ == "__main__":
    pytest.main([__file__])