#!/usr/bin/env python3
"""Validation script for training improvements.

Run this BEFORE starting training to verify all changes work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_proximity_penalty():
    """Test proximity penalty implementation."""
    print("=" * 60)
    print("TEST 1: Proximity Penalty")
    print("=" * 60)

    from alphagomoku.utils.pattern_detector import compute_proximity_mask, get_pattern_features
    import numpy as np

    # Test 1: Board with one stone
    board = np.zeros((15, 15), dtype=np.int8)
    board[7, 7] = 1
    mask = compute_proximity_mask(board)

    assert 0.95 < mask[7, 8] <= 1.0, f"Adjacent cell should be 1.0, got {mask[7, 8]}"
    assert 0.45 < mask[7, 9] < 0.55, f"Distance 2 should be ~0.5, got {mask[7, 9]}"
    assert mask[0, 0] < 0.1, f"Far corner should be <0.1, got {mask[0, 0]}"

    print(f"✓ Adjacent to stone: {mask[7, 8]:.3f} (expected: 1.0)")
    print(f"✓ 2 cells away: {mask[7, 9]:.3f} (expected: 0.5)")
    print(f"✓ Corner (far): {mask[0, 0]:.3f} (expected: <0.1)")

    # Test 2: Pattern features include proximity
    board = np.zeros((15, 15), dtype=np.int8)
    board[7, 5:9] = 1  # Four in a row
    features = get_pattern_features(board, 1)

    assert features[7, 9] > 0.8, f"Next to pattern should be high, got {features[7, 9]}"
    assert features[0, 0] < 0.1, f"Far from pattern should be low, got {features[0, 0]}"

    print(f"✓ Next to pattern: {features[7, 9]:.3f} (expected: >0.8)")
    print(f"✓ Far from pattern: {features[0, 0]:.3f} (expected: <0.1)")

    # Test 3: Opening exception
    board = np.zeros((15, 15), dtype=np.int8)
    board[7, 7] = 1  # Just one stone
    mask = compute_proximity_mask(board)
    center_mask = mask[7, 7:10].mean()

    print(f"✓ Opening favor center: {center_mask:.3f}")

    print("\n✅ Proximity penalty tests PASSED!\n")
    return True


def test_temperature_scheduler():
    """Test temperature scheduler implementation."""
    print("=" * 60)
    print("TEST 2: Temperature Scheduler")
    print("=" * 60)

    from alphagomoku.selfplay.temperature import TemperatureScheduler
    import numpy as np

    # Test for different epochs
    print("\nTemperature schedule by epoch and move:")
    for epoch in [0, 30, 60, 100]:
        scheduler = TemperatureScheduler(epoch=epoch)
        temps = [scheduler.get_temperature(move) for move in [0, 3, 6, 10, 20]]
        temp_str = [f"{t:.2f}" for t in temps]
        print(f"  Epoch {epoch:3d}: moves [0,3,6,10,20] → {temp_str}")

        # Validate temperature decreases with moves
        assert temps[0] >= temps[2], "Temperature should decrease over game"
        assert temps[-1] == 0.0, "Late game should be deterministic"

    # Test critical position override
    scheduler = TemperatureScheduler(epoch=50)
    temp_normal = scheduler.get_temperature(5, is_critical=False)
    temp_critical = scheduler.get_temperature(5, is_critical=True)

    assert temp_critical == 0.0, f"Critical positions should be temp=0, got {temp_critical}"
    assert temp_normal > 0.3, f"Normal positions should have exploration, got {temp_normal}"

    print(f"\n✓ Normal position (move 5): temp = {temp_normal:.2f}")
    print(f"✓ Critical position (move 5): temp = {temp_critical:.2f} (forced deterministic)")

    # Test temperature application
    policy = np.array([0.5, 0.3, 0.2])
    temp_policy = scheduler.apply_temperature(policy, temperature=0.0)
    assert temp_policy[0] == 1.0, "Temperature 0 should give one-hot on best"
    assert temp_policy[1] == 0.0 and temp_policy[2] == 0.0, "Others should be zero"

    print(f"✓ Temperature application: {temp_policy} (one-hot on best)")

    print("\n✅ Temperature scheduler tests PASSED!\n")
    return True


def test_imports():
    """Test all imports work."""
    print("=" * 60)
    print("TEST 3: Import Validation")
    print("=" * 60)

    try:
        from alphagomoku.model.network import GomokuNet
        print("✓ GomokuNet import")

        from alphagomoku.utils.pattern_detector import compute_proximity_mask, get_pattern_features
        print("✓ Pattern detector import")

        from alphagomoku.selfplay.temperature import TemperatureScheduler
        print("✓ Temperature scheduler import")

        from alphagomoku.selfplay.selfplay import SelfPlayWorker
        print("✓ SelfPlayWorker import")

        from alphagomoku.train.tactical_augmentation import augment_with_tactical_data
        print("✓ Tactical augmentation import")

        print("\n✅ All imports PASSED!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        print("Solution: Run 'pip install -e .' from project root")
        return False


def test_model_loading():
    """Test that model can still load old checkpoints."""
    print("=" * 60)
    print("TEST 4: Model Checkpoint Loading")
    print("=" * 60)

    import torch
    from alphagomoku.model.network import GomokuNet

    # Find latest checkpoint
    checkpoint_dir = project_root / "checkpoints"
    checkpoints = list(checkpoint_dir.glob("model_epoch_*.pt"))

    if not checkpoints:
        print("⚠️  No checkpoints found - skipping this test")
        print("   (This is OK if starting fresh training)")
        return True

    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    print(f"Testing checkpoint: {latest_checkpoint.name}")

    try:
        model = GomokuNet(board_size=15, num_blocks=12, channels=64)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', -1)
            print(f"✓ Loaded checkpoint from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint (legacy format)")

        # Test inference with new pattern features
        import numpy as np
        from alphagomoku.utils.pattern_detector import get_pattern_features

        board = np.zeros((15, 15), dtype=np.int8)
        board[7, 7] = 1

        # Build 5-channel input
        own = (board == 1).astype(np.float32)
        opp = (board == -1).astype(np.float32)
        last = np.zeros_like(board, dtype=np.float32)
        side = np.ones_like(board, dtype=np.float32)
        pattern = get_pattern_features(board, 1)

        state = torch.from_numpy(np.stack([own, opp, last, side, pattern])).float().unsqueeze(0)

        model.eval()
        with torch.no_grad():
            policy, value = model(state)

        assert torch.all(torch.isfinite(policy)), "Policy has non-finite values"
        assert torch.all(torch.isfinite(value)), "Value has non-finite values"

        print(f"✓ Model inference works with new pattern features")
        print(f"  Policy shape: {policy.shape}, Value: {value.item():.3f}")

        print("\n✅ Model checkpoint loading PASSED!\n")
        return True

    except Exception as e:
        print(f"\n❌ Checkpoint loading failed: {e}")
        print("   This might be OK if model architecture changed")
        print("   Consider starting fresh training")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("AlphaGomoku Training Improvements - Validation Suite")
    print("=" * 60)
    print("\nThis script validates that all improvements are working correctly.")
    print("Run this BEFORE starting training.\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Proximity Penalty", test_proximity_penalty()))
    results.append(("Temperature Scheduler", test_temperature_scheduler()))
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if all_passed:
        print("\n🎉 All validations PASSED! Ready to start training.")
        print("\nNext step:")
        print("  make train")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")
        print("\nCheck IMPLEMENTATION_COMPLETE.md for troubleshooting.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
