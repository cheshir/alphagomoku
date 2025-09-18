"""Unit tests for learning rate schedulers."""

import math
import torch

from alphagomoku.train.schedulers import WarmupCosineScheduler


def test_warmup_cosine_basic_progression():
    """Warmup increases to base_lr, then cosine decays towards min_lr."""
    model = torch.nn.Linear(4, 2)
    base_lr = 0.1
    min_lr = 0.001
    warmup_epochs = 2
    max_epochs = 10

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr)
    sched = WarmupCosineScheduler(
        opt,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        base_lr=base_lr,
        min_lr=min_lr,
        start_epoch=0,
    )

    # Epoch 0 (after init): linear warmup 1/2 of base_lr
    lr_e0 = opt.param_groups[0]["lr"]
    assert math.isclose(lr_e0, base_lr * 1 / warmup_epochs, rel_tol=1e-6)

    # Epoch 1: should reach base_lr after warmup
    sched.step()  # epoch -> 1
    lr_e1 = opt.param_groups[0]["lr"]
    assert math.isclose(lr_e1, base_lr * 2 / warmup_epochs, rel_tol=1e-6)

    # Epoch 2: first cosine step equals base_lr
    sched.step()  # epoch -> 2
    lr_e2 = opt.param_groups[0]["lr"]
    assert math.isclose(lr_e2, base_lr, rel_tol=1e-6)

    # Progress a few epochs; LR should be within [min_lr, base_lr]
    for _ in range(3):
        sched.step()
        lr = opt.param_groups[0]["lr"]
        assert min_lr - 1e-9 <= lr <= base_lr + 1e-9

    # Advance to end; LR should approach min_lr (not necessarily equal)
    while sched.epoch < max_epochs:
        sched.step()
    lr_end = opt.param_groups[0]["lr"]
    assert lr_end >= min_lr - 1e-9


def test_warmup_cosine_state_dict_roundtrip():
    """State dict save/load preserves epoch and LR."""
    model = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    sched = WarmupCosineScheduler(opt, warmup_epochs=1, max_epochs=5, base_lr=0.05, min_lr=0.005)

    # Step a few epochs
    sched.step()  # epoch 1
    sched.step()  # epoch 2
    lr_before = opt.param_groups[0]["lr"]

    # Save state
    state = sched.state_dict()

    # New optimizer + scheduler; then load
    opt2 = torch.optim.SGD(model.parameters(), lr=0.01)
    sched2 = WarmupCosineScheduler(opt2, warmup_epochs=3, max_epochs=99, base_lr=0.01, min_lr=0.0)
    sched2.load_state_dict(state)

    # Should match restored epoch and LR
    assert sched2.epoch == sched.epoch
    assert math.isclose(opt2.param_groups[0]["lr"], lr_before, rel_tol=1e-9)

