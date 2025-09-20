# AlphaGomoku Development Guide

## Project Overview
AlphaZero-style Gomoku AI with training and inference components.

## Key Project Information
- **Target**: 15×15 Gomoku, single model for all difficulties
- **Training**: Apple Silicon M1 Pro, PyTorch + MPS
- **Inference**: arm64 Linux, 4 vCPU, 8GB RAM, CPU-only
- **Architecture**: DW-ResNet-SE + MCTS + TSS + Endgame Solver
- **Framework Stack**: PyTorch, Gymnasium, ONNX Runtime

## Development Rules

### 1. Documentation-First Rule
**ALWAYS check the `docs/` directory before implementing any feature.**

Required documentation files:
- `docs/PROJECT_DESCRIPTION.md` - Complete technical specification
- `docs/TSS.md` - Threat-Space Search specification
- Additional specs may be added to `docs/` during development

### 2. Implementation Priority
1. Training pipeline first (self-play → MCTS → neural network training)
2. Core components: Environment → Model → MCTS → Self-play → Training
3. Advanced components: TSS → Endgame Solver → Evaluation
4. Inference API last

### 3. Code Standards
- Minimal, focused implementations
- Unit + integration tests required
- Follow specifications exactly
- No verbose code that doesn't contribute to solution

### 4. Module Structure
```
alphagomoku/
├── env/          # Gymnasium Gomoku environment
├── model/        # DW-ResNet-SE neural network
├── mcts/         # Monte Carlo Tree Search
├── tss/          # Threat-Space Search
├── endgame/      # Alpha-beta endgame solver
├── selfplay/     # Self-play data generation
├── train/        # Training pipeline
├── eval/         # Evaluation and testing
└── utils/        # Shared utilities
```

## Current Implementation Status
- ✅ Project structure and requirements
- ✅ Gomoku environment (Gymnasium interface)
- ✅ DW-ResNet-SE neural network model
- ✅ MCTS implementation
- 🔄 Self-play data generation (in progress)
- ⏳ TSS implementation (pending)
- ⏳ Training pipeline (pending)
- ⏳ Evaluation framework (pending)

## Next Steps
Continue with self-play implementation, then proceed to training pipeline.