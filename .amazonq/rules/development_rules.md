# AlphaGomoku Development Rules

## Documentation-First Implementation Rule

**CRITICAL**: Before implementing ANY feature or component, ALWAYS:

1. Check `docs/PROJECT_DESCRIPTION.md` for complete technical specifications
2. Check `docs/TSS.md` for Threat-Space Search specifications  
3. Look for any additional documentation in `docs/` directory
4. Verify implementation matches specifications exactly

## Project Context

- **Goal**: AlphaZero-style Gomoku AI (15×15 board)
- **Training**: PyTorch + MPS on Apple Silicon
- **Inference**: ONNX Runtime on arm64 Linux
- **Architecture**: DW-ResNet-SE + MCTS + TSS + Endgame Solver
- **Approach**: Training pipeline first, then inference API

## Implementation Standards

- Write minimal, focused code that directly addresses requirements
- Follow specifications in `docs/` exactly
- Implement both unit and integration tests
- Maintain monorepo structure with clear module separation