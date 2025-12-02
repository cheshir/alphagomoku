# AlphaGomoku Makefile - Optimized Training Configurations
# Based on refactoring plan - using model presets and proper hyperparameters

# Load environment variables from .env file if it exists
-include .env
export

venv:
	conda activate alphagomoku

# =============================================================================
# Fast Iteration (Small Model) - For Experiments
# Quick experiments and validation - pure MCTS (AlphaZero style)
# Expected: 5-10 min/epoch on MPS, 3-5 min/epoch on CUDA
# =============================================================================

train-fast:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset small \
		--epochs 50 \
		--selfplay-games 50 \
		--mcts-simulations 200 \
		--batch-size 256 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 5 \
		--lr-schedule cosine \
		--map-size-gb 16 \
		--buffer-max-size 2000000 \
		--batch-size-mcts 64 \
		--parallel-workers 4 \
		--difficulty easy \
		--device auto \
		--resume auto \
		--eval-frequency 5 \
		--eval-games 50

# =============================================================================
# RECOMMENDED: Balanced Training (Small Model) - AlphaZero Style
# Best balance of speed and strength
# Pure MCTS (no TSS) - network learns threats through self-play
# Expected: 10-20 min/epoch on MPS, 5-10 min/epoch on CUDA
# =============================================================================

train:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset small \
		--epochs 200 \
		--selfplay-games 100 \
		--mcts-simulations 400 \
		--batch-size 256 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--lr-schedule cosine \
		--map-size-gb 32 \
		--buffer-max-size 5000000 \
		--batch-size-mcts 64 \
		--parallel-workers 4 \
		--difficulty easy \
		--device auto \
		--resume auto \
		--eval-frequency 5 \
		--eval-games 50

# =============================================================================
# Production Training (Medium Model) - AlphaZero Style
# Pure MCTS training (no TSS) - faster and follows AlphaZero methodology
# TSS is used during inference/evaluation, not training
# Expected: 20-30 min/epoch on CUDA, 15-30 epochs/day
# =============================================================================

train-production:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset medium \
		--epochs 200 \
		--selfplay-games 200 \
		--mcts-simulations 400 \
		--batch-size 1024 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--lr-schedule cosine \
		--map-size-gb 32 \
		--buffer-max-size 5000000 \
		--batch-size-mcts 128 \
		--parallel-workers 1 \
		--difficulty easy \
		--device auto \
		--resume auto \
		--eval-frequency 10 \
		--eval-games 100

# =============================================================================
# Legacy configs (for backward compatibility)
# Note: These use OLD hyperparameters (low MCTS sims, small buffer, no eval)
# =============================================================================

train-legacy:
	@echo "⚠️  WARNING: Using legacy configuration (not recommended)"
	@echo "    Consider using 'make train' or 'make train-fast' instead"
	@echo ""
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--epochs 200 \
		--selfplay-games 200 \
		--mcts-simulations 150 \
		--batch-size 512 \
		--lr 1e-3 \
		--min-lr 5e-4 \
		--warmup-epochs 0 \
		--lr-schedule cosine \
		--map-size-gb 12 \
		--buffer-max-size 500000 \
		--batch-size-mcts 64 \
		--parallel-workers 4 \
		--difficulty medium \
		--device auto \
		--resume auto

# =============================================================================
# Device-Specific Training (Advanced)
# =============================================================================

train-cuda:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset small \
		--epochs 200 \
		--selfplay-games 100 \
		--mcts-simulations 400 \
		--batch-size 1024 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--lr-schedule cosine \
		--map-size-gb 32 \
		--buffer-max-size 5000000 \
		--batch-size-mcts 128 \
		--parallel-workers 1 \
		--difficulty medium \
		--device cuda \
		--resume auto \
		--eval-frequency 5

train-mps:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset small \
		--epochs 200 \
		--selfplay-games 100 \
		--mcts-simulations 400 \
		--batch-size 256 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--lr-schedule cosine \
		--map-size-gb 32 \
		--buffer-max-size 5000000 \
		--batch-size-mcts 64 \
		--parallel-workers 4 \
		--difficulty medium \
		--device mps \
		--resume auto \
		--eval-frequency 5

train-cpu:
	OMP_NUM_THREADS=4 \
	python scripts/train.py \
		--model-preset small \
		--epochs 100 \
		--selfplay-games 50 \
		--mcts-simulations 200 \
		--batch-size 128 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 5 \
		--lr-schedule cosine \
		--map-size-gb 16 \
		--buffer-max-size 2000000 \
		--batch-size-mcts 32 \
		--parallel-workers 8 \
		--difficulty medium \
		--device cpu \
		--resume auto \
		--eval-frequency 10

# =============================================================================
# Evaluation and Testing
# =============================================================================

evaluate-latest:
	@latest=$$(ls -t checkpoints/model_epoch_*.pt | head -1); \
	echo "Evaluating latest checkpoint: $$latest"; \
	python scripts/evaluate.py $$latest

test-tactical-latest:
	@latest=$$(ls -t checkpoints/model_epoch_*.pt | head -1); \
	echo "Testing tactical awareness: $$latest"; \
	python scripts/test_tactical.py $$latest

test:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	MKL_THREADING_LAYER=SEQUENTIAL \
	KMP_DUPLICATE_LIB_OK=TRUE \
	KMP_AFFINITY=disabled \
	KMP_INIT_AT_FORK=FALSE \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	pytest tests/unit -v

test-all:
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	MKL_THREADING_LAYER=SEQUENTIAL \
	KMP_DUPLICATE_LIB_OK=TRUE \
	KMP_AFFINITY=disabled \
	KMP_INIT_AT_FORK=FALSE \
	PYTORCH_ENABLE_MPS_FALLBACK=1 \
	pytest tests/ -v

# =============================================================================
# Utilities
# =============================================================================

clean-checkpoints:
	@echo "Removing all checkpoints except latest 5..."
	@ls -t checkpoints/model_epoch_*.pt | tail -n +6 | xargs rm -f
	@echo "Done"

clean-data:
	@echo "⚠️  WARNING: This will delete all training data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/replay_buffer; \
		echo "Training data deleted"; \
	fi

show-config:
	@echo "Current Training Configurations:"
	@echo ""
	@echo "FAST (Development):"
	@echo "  Model: small (1.2M params)"
	@echo "  Difficulty: easy (pure MCTS, no TSS)"
	@echo "  Time: ~5-10 min/epoch on CUDA"
	@echo "  Usage: make train-fast"
	@echo ""
	@echo "BALANCED (Recommended):"
	@echo "  Model: small (1.2M params)"
	@echo "  Difficulty: easy (pure MCTS, AlphaZero style)"
	@echo "  Time: ~10-20 min/epoch on MPS, ~5-10 on CUDA"
	@echo "  Usage: make train"
	@echo ""
	@echo "PRODUCTION (Maximum Strength):"
	@echo "  Model: medium (3M params)"
	@echo "  Difficulty: easy (pure MCTS, network learns threats)"
	@echo "  Time: ~20-30 min/epoch on CUDA"
	@echo "  Usage: make train-production"
	@echo ""
	@echo "NOTE: All configs use 'difficulty: easy' during training"
	@echo "      TSS is used during inference/evaluation for stronger play"
	@echo "      This follows AlphaZero methodology (pure self-play)"
	@echo ""
	@echo "DYNAMIC (Hardware-optimized):"
	@echo "  Automatically detects your hardware"
	@echo "  Recommends optimal settings"
	@echo "  Usage: make show-hardware-config"

show-hardware-config:
	@echo "Detecting your hardware and recommending settings..."
	@echo ""
	@python scripts/show_recommended_config.py

show-hardware-config-speed:
	@echo "Optimizing for SPEED..."
	@echo ""
	@python scripts/show_recommended_config.py --prefer-speed

show-hardware-config-strength:
	@echo "Optimizing for STRENGTH..."
	@echo ""
	@python scripts/show_recommended_config.py --prefer-strength

help:
	@echo "AlphaGomoku Training Makefile"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make show-hardware-config     - ⭐ Get recommended settings for YOUR hardware"
	@echo "  make train-fast               - Fast iteration (small model, 50 epochs)"
	@echo "  make train                    - Balanced training (recommended)"
	@echo "  make train-production         - Production training (strongest model)"
	@echo ""
	@echo "⚙️  Hardware-Specific Recommendations:"
	@echo "  make show-hardware-config        - Balanced recommendation"
	@echo "  make show-hardware-config-speed  - Optimize for speed"
	@echo "  make show-hardware-config-strength - Optimize for strength"
	@echo ""
	@echo "🖥️  Device-Specific Training:"
	@echo "  make train-cuda       - CUDA GPU training"
	@echo "  make train-mps        - Apple Silicon (MPS) training"
	@echo "  make train-cpu        - CPU training"
	@echo ""
	@echo "📊 Evaluation:"
	@echo "  make evaluate-latest  - Evaluate latest checkpoint"
	@echo "  make test-tactical-latest - Test tactical awareness"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test             - Run unit tests"
	@echo "  make test-all         - Run all tests"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make show-config      - Show configuration summary"
	@echo "  make clean-checkpoints - Keep only 5 latest checkpoints"
	@echo "  make clean-data       - Delete all training data (WARNING)"
	@echo ""
	@echo "🌐 Distributed Training:"
	@echo "  make distributed-help - Show distributed training guide"
	@echo "  make distributed-selfplay-cpu-workers - Start 6 CPU self-play workers"
	@echo "  make distributed-selfplay-mps-worker  - Start 1 MPS self-play worker"
	@echo "  make distributed-training-gpu         - Start GPU training worker"
	@echo "  make distributed-monitor              - Monitor queue status"
	@echo "  make distributed-migrate-queue        - Migrate old queue data (dry-run)"
	@echo "  make distributed-migrate-queue-apply  - Apply queue migration"
	@echo ""
	@echo "📖 For more info: see README.md or WHATS_NEW.md"

# =============================================================================
# Distributed Training (Mac Self-Play + Colab Training)
# =============================================================================

# Distributed Self-Play Workers (CPU) - Run on Mac M1 Pro
# Uses separate PROCESSES (not threads) for TRUE parallel execution
# Bypasses Python's GIL - Expected CPU usage: ~600% with 6 workers
# Batches 500 positions (~10 games) before pushing to Redis
distributed-selfplay-cpu-workers:
	OMP_NUM_THREADS=1 \
	python scripts/distributed_selfplay_manager.py \
		--redis-url "$$REDIS_URL" \
		--num-workers 8 \
		--model-preset medium \
		--mcts-simulations 100 \
		--device cpu \
		--positions-per-push 1000 \
		--model-update-frequency 10 \
		--mcts-batch-size 128

# Distributed Training Worker (GPU) - Run on Colab T4
distributed-training-gpu:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/distributed_training_worker.py \
		--redis-url "$$REDIS_URL" \
		--model-preset medium \
		--batch-size 1024 \
		--device cuda \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--publish-frequency 5 \
		--checkpoint-dir ./checkpoints_distributed \
		--min-games-for-training 50 \
		--games-per-training-batch 50

# Monitor distributed queue status
distributed-monitor:
	python scripts/monitor_queue.py \
		--redis-url "$$REDIS_URL" \
		--refresh 5

# Migrate Redis queue from old format to new position-based format
distributed-migrate-queue:
	@echo "🔍 Running migration in DRY RUN mode first..."
	@echo ""
	python scripts/migrate_redis_queue.py \
		--redis-url "$$REDIS_URL" \
		--positions-per-batch 1000 \
		--dry-run
	@echo ""
	@echo "To apply migration, run: make distributed-migrate-queue-apply"

# =============================================================================
# Distributed Training - Combined Commands
# =============================================================================

distributed-help:
	@echo "Distributed Training Setup for AlphaGomoku"
	@echo ""
	@echo "Architecture (Producer-Consumer Pattern):"
	@echo "  Worker Processes → Local Queue → Accumulator → Redis → Training Worker"
	@echo "  Model Updater → Checkpoints (version-based sync)"
	@echo ""
	@echo "Setup Steps:"
	@echo "  1. Setup Redis"
	@echo "  2. Set REDIS_URL in .env file:"
	@echo "     REDIS_URL=redis://:password@queue.cheshir.me:6379/0"
	@echo "  3. Start self-play workers on Mac:"
	@echo "     make distributed-selfplay-cpu-workers  # 8 CPU workers (multiprocess)"
	@echo "     OR"
	@echo "     make distributed-selfplay-mps-worker   # 1 MPS worker (multiprocess)"
	@echo "  4. Start training worker on Colab:"
	@echo "     make distributed-training-gpu"
	@echo "  5. Monitor queue status:"
	@echo "     make distributed-monitor"
	@echo ""
	@echo "Commands:"
	@echo "  make distributed-selfplay-cpu-workers  - Start 8 CPU worker processes (bypasses GIL)"
	@echo "  make distributed-selfplay-mps-worker   - Start 1 MPS worker process"
	@echo "  make distributed-training-gpu          - Start GPU training worker"
	@echo "  make distributed-monitor               - Monitor queue status"
	@echo ""
	@echo "Architecture Benefits:"
	@echo "  ✓ Multiprocessing: Each worker is a separate process (bypasses Python's GIL!)"
	@echo "  ✓ True parallelism: 8 workers = ~800% CPU (8 cores fully utilized)"
	@echo "  ✓ Producer-consumer pattern for efficient batching"
	@echo "  ✓ Version-based model sync (lock-free, eventual consistency)"
	@echo "  ✓ Unified dashboard with live statistics"
	@echo "  ✓ Each worker has own model copy (no race conditions)"
	@echo ""
	@echo "Resource Allocation:"
	@echo "  Mac M1 Pro (8 CPU worker processes):"
	@echo "    - Medium model, 50 MCTS sims"
	@echo "    - Each worker: separate process with own interpreter"
	@echo "    - Expected: ~800% CPU usage (8 cores @ 100%)"
	@echo "    - 2-4x faster than threaded version"
	@echo ""
	@echo "  Colab T4 GPU (1 training worker):"
	@echo "    - Medium model, batch 1024"
	@echo "    - ~3-5 min per 50-game training batch"
	@echo "    - Can process ~600-1000 games/hour"
	@echo ""
	@echo "For more info: see docs/TRAINING.md#distributed-training"

.PHONY: venv train train-fast train-production train-legacy train-cuda train-mps train-cpu \
        evaluate-latest test-tactical-latest test test-all \
        clean-checkpoints clean-data show-config help \
        distributed-selfplay-cpu-workers distributed-selfplay-mps-worker \
        distributed-training-gpu distributed-monitor distributed-help
