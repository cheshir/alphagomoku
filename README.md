# AlphaGomoku: AlphaZero-style Gomoku AI

A strong Gomoku (15×15) AI implementation using AlphaZero methodology with self-play training, Monte Carlo Tree Search (MCTS), and deep neural networks.

## 🎯 Project Overview

- **Goal**: Build a competitive Gomoku AI that can beat experienced human players
- **Architecture**: DW-ResNet-SE + MCTS + Threat-Space Search + Endgame Solver
- **Training**: PyTorch with MPS acceleration on Apple Silicon
- **Inference**: ONNX Runtime for cross-platform deployment
- **Board Size**: 15×15 (classic Gomoku rules)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS with Apple Silicon (for training) or any platform (for inference)
- Conda or pip environment manager

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd alphagomoku
conda create -n alphagomoku python=3.12
conda activate alphagomoku
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install package in development mode:**
```bash
pip install -e .
```

### Verify Installation

```bash
# Test the training pipeline (currently has known issues with SelfPlayWorker API)
# python scripts/test_training.py

# Run unit tests
python -m pytest tests/unit/ -v

# Alternative: Test basic imports work
python -c "from alphagomoku.model.network import GomokuNet; print('✓ Core imports working')"
```

## 🏋️ Training Pipeline

### Quick Training Test

Start with a minimal training run to verify everything works:

```bash
python scripts/train.py \
    --epochs 5 \
    --selfplay-games 10 \
    --batch-size 128 \
    --lr 0.01
```

### Recommended Training Configuration

For serious training to achieve strong play:

```bash
python scripts/train.py \
    --epochs 200 \
    --selfplay-games 500 \
    --batch-size 512 \
    --lr 0.001 \
    --data-dir ./training_data \
    --checkpoint-dir ./checkpoints
```

**Training Parameters Explained:**

- `--epochs`: Number of training iterations (200+ recommended)
- `--selfplay-games`: Games per epoch for data generation (500+ for strong play)
- `--batch-size`: Training batch size (512 optimal for 16GB RAM)
- `--lr`: Learning rate (0.001 works well with AdamW)
- `--data-dir`: Directory for replay buffer storage
- `--checkpoint-dir`: Model checkpoint storage
- `--map-size-gb`: LMDB database size limit in GB (default: 64GB)

### Resume Training

```bash
python scripts/train.py \
    --resume ./checkpoints/model_epoch_50.pt \
    --epochs 200
```

## 📊 Training Recommendations

### Hardware Requirements

**Minimum (Testing):**
- 8GB RAM
- Apple Silicon M1/M2 or CUDA GPU
- 10GB disk space

**Recommended (Production):**
- 16GB+ RAM
- Apple Silicon M1 Pro/Max or RTX 3080+
- 50GB+ disk space for replay buffer

### Training Strategy

1. **Phase 1 - Bootstrap (Epochs 1-50):**
   - Small games count (100-200 per epoch)
   - Higher learning rate (0.01)
   - Focus on basic patterns

2. **Phase 2 - Strengthening (Epochs 51-150):**
   - Increase games (300-500 per epoch)
   - Reduce learning rate (0.001)
   - Longer MCTS simulations (800+)

3. **Phase 3 - Fine-tuning (Epochs 151+):**
   - Maximum games (500+ per epoch)
   - Lower learning rate (0.0005)
   - Add evaluation against previous versions

### Expected Training Time

- **M1 Pro (16GB)**: ~2-3 hours per epoch with 500 games
- **M1 Max (32GB)**: ~1-2 hours per epoch with 500 games
- **Total for strong model**: 200-400 hours of training

## 🎮 Model Evaluation

### Test Model Strength

```bash
# Quick strength test
python -c "
from alphagomoku.model.network import GomokuNet
from alphagomoku.eval.evaluator import Evaluator
import torch

model = GomokuNet()
model.load_state_dict(torch.load('checkpoints/model_best.pt')['model_state_dict'])
evaluator = Evaluator(model)

# Test against different simulation counts
results = evaluator.evaluate_strength(test_sims=800, baseline_sims=400, num_games=50)
print(f'Win rate: {results[\"win_rate\"]:.2%}')
"
```

## 📁 Project Structure

```
alphagomoku/
├── alphagomoku/           # Core package
│   ├── env/              # Gomoku environment
│   ├── model/            # Neural network architecture
│   ├── mcts/             # Monte Carlo Tree Search
│   ├── selfplay/         # Self-play data generation
│   ├── train/            # Training pipeline
│   ├── eval/             # Evaluation framework
│   └── tss/              # Threat-Space Search
├── scripts/              # Training and utility scripts
├── tests/                # Unit and integration tests
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── docs/                 # Technical specifications
├── configs/              # Configuration files
├── data/                 # Training data directory
├── checkpoints/          # Model checkpoints
└── runs/                 # Training run logs
```

## 🔧 Configuration

### Model Architecture

The default configuration uses:
- **Blocks**: 12 residual blocks
- **Channels**: 64 feature channels
- **Parameters**: ~2.7M (efficient for training)

For stronger models, increase parameters:
```python
model = GomokuNet(board_size=15, num_blocks=16, channels=128)  # ~10M parameters
```

### MCTS Settings

Training defaults:
- **Simulations**: 800 per move
- **CPUCT**: 1.8 (exploration parameter)
- **Temperature**: 1.0 for first 8 moves, then 0.0

## 🐛 Troubleshooting

### Known Issues

**SelfPlayWorker API Issue:**
```bash
# Current test_training.py fails with:
# TypeError: SelfPlayWorker.__init__() got an unexpected keyword argument 'num_simulations'
# Work-around: Use train.py directly for training instead
```

### Common Issues

**1. Out of Memory during training:**
```bash
# Reduce batch size
python scripts/train.py --batch-size 256

# Or reduce model size
# Edit model creation in scripts/train.py:
# model = GomokuNet(num_blocks=8, channels=32)
```

**2. OpenMP library conflict (macOS):**
```bash
# Quick fix - set environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
python scripts/train.py --epochs 5 --selfplay-games 10 --batch-size 128 --lr 0.01

# Permanent fix - add to shell profile
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc
source ~/.zshrc
```

**3. MPS not available:**
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Force CPU training if needed
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

**4. LMDB database corruption:**
```bash
# Remove and restart training
rm -rf ./data/replay_buffer
python scripts/train.py --epochs 200
```

### Performance Optimization

**Training Speed:**
- Use MPS on Apple Silicon: ~3x faster than CPU
- Batch size 512 optimal for 16GB RAM
- SSD storage recommended for replay buffer

**Memory Usage:**
- Replay buffer: ~2-5GB for 5M positions
- Model training: ~1-2GB GPU memory
- MCTS tree: ~500MB-1GB during self-play

## 📈 Expected Results

### Training Progress

After proper training (200+ epochs, 500 games/epoch):

- **Epoch 50**: Beats random play consistently
- **Epoch 100**: Understands basic tactics (blocks threats)
- **Epoch 150**: Recognizes complex patterns
- **Epoch 200+**: Competitive with experienced humans

### Model Strength Indicators

- **Policy Accuracy**: >60% on training positions
- **Value MAE**: <0.3 (mean absolute error)
- **Win Rate**: >70% vs 400-simulation baseline

## 🚧 Roadmap

### Completed ✅
- [x] Gomoku environment (Gymnasium interface)
- [x] DW-ResNet-SE neural network
- [x] MCTS with neural network guidance
- [x] Self-play data generation
- [x] Training pipeline with replay buffer
- [x] Basic evaluation framework

### In Progress 🔄
- [x] Threat-Space Search (TSS) implementation (basic version complete)
- [ ] SelfPlayWorker API fixes (constructor parameter issues)
- [ ] Alpha-beta endgame solver
- [ ] ONNX model export
- [ ] Inference API server

### Planned 📋
- [ ] Opening book integration
- [ ] Advanced evaluation positions
- [ ] Multi-difficulty inference modes
- [ ] Web interface for human play

## 📚 References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Gomoku Rules](https://en.wikipedia.org/wiki/Gomoku)
- [MCTS Survey](https://ieeexplore.ieee.org/document/6145622)

## 🤝 Contributing

1. Follow the documentation-first rule: check `docs/` before implementing
2. Write tests for new features
3. Use black for code formatting: `black alphagomoku/`
4. Run tests before submitting: `pytest tests/`

## 📄 License

MIT License - see LICENSE file for details.