# AlphaGomoku Makefile - Optimized Training Configurations
# Based on refactoring plan - using model presets and proper hyperparameters

venv:
	conda activate alphagomoku

# =============================================================================
# RECOMMENDED: Fast Iteration (Small Model)
# Best for development and quick experiments
# Expected: 10-15 min/epoch, 80+ epochs/day
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
		--batch-size-mcts 32 \
		--parallel-workers 4 \
		--difficulty medium \
		--device auto \
		--resume auto \
		--eval-frequency 5 \
		--eval-games 50

# =============================================================================
# RECOMMENDED: Balanced Training (Small Model)
# Best balance of speed and strength
# Expected: 15-25 min/epoch, 50+ epochs/day
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
		--difficulty medium \
		--device auto \
		--resume auto \
		--eval-frequency 5 \
		--eval-games 50

# =============================================================================
# Production Training (Medium Model)
# For final strong model - slower but stronger
# Expected: 40-60 min/epoch, 24-36 epochs/day
# =============================================================================

train-production:
	PYTORCH_ENABLE_MPS_FALLBACK=0 \
	OMP_NUM_THREADS=1 \
	python scripts/train.py \
		--model-preset medium \
		--epochs 200 \
		--selfplay-games 200 \
		--mcts-simulations 600 \
		--batch-size 512 \
		--lr 1e-3 \
		--min-lr 1e-6 \
		--warmup-epochs 10 \
		--lr-schedule cosine \
		--map-size-gb 32 \
		--buffer-max-size 5000000 \
		--batch-size-mcts 96 \
		--parallel-workers 1 \
		--difficulty medium \
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
	@echo "  Time: ~10-15 min/epoch"
	@echo "  Usage: make train-fast"
	@echo ""
	@echo "BALANCED (Recommended):"
	@echo "  Model: small (1.2M params)"
	@echo "  Time: ~15-25 min/epoch"
	@echo "  Usage: make train"
	@echo ""
	@echo "PRODUCTION (Maximum Strength):"
	@echo "  Model: medium (3M params)"
	@echo "  Time: ~40-60 min/epoch"
	@echo "  Usage: make train-production"
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
	@echo "📖 For more info: see README.md or WHATS_NEW.md"

.PHONY: venv train train-fast train-production train-legacy train-cuda train-mps train-cpu \
        evaluate-latest test-tactical-latest test test-all \
        clean-checkpoints clean-data show-config help
