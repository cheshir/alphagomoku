# Archived Documentation

This directory contains historical documentation that describes the evolution of the AlphaGomoku project.

## Current Documentation (Use These)

- **Main**: `../README.md` - Project overview and quick start
- **Training**: `../WHATS_NEW.md` - Latest training configurations
- **Details**: `../REFACTORING_SUMMARY.md` - Comprehensive changes
- **Plan**: `../REFACTORING_PLAN.md` - Refactoring details

## Archived Files (Historical Reference Only)

The following files describe **previous iterations** of the project and may contain **outdated information**:

### Superseded by Refactoring
- `MODEL_UPGRADE.md` - Described upgrade to 5M model (now we use presets)
- `TRAINING_FIXES.md` - Old training issues (fixed in refactoring)
- `TRAINING_IMPROVEMENTS.md` - Old improvements (superseded)
- `GRADIENT_CHECKPOINTING.md` - Now handled by model presets
- `DEVICE_TRAINING.md` - Now handled by `--device auto`
- `MEMORY_DEBUG.md` - Old memory issues (resolved)
- `DATA_FILTERING.md` - Old filtering approach (simplified)
- `PROXIMITY_PENALTY.md` - Old data filtering (removed)

### Still Relevant (But May Need Updates)
- `PROJECT_DESCRIPTION.md` - Original spec (mostly still valid)
- `TSS.md` - TSS specification (still valid)
- `TSS_*.md` - TSS implementation details (still valid)
- `API.md` - Backend API (still valid)
- `UI_IMPLEMENTATION.md` - Frontend details (still valid)
- `DEPLOYMENT.md` - Deployment guide (still valid)
- `TESTING.md` - Test guide (still valid)
- `OPENING_STRATEGY.md` - Opening book (future work)
- `FORCED_WINS_VERIFIED.md` - Endgame solver verification (still valid)

### Development Guides (Still Useful)
- `QUICKSTART.md` - Quick start guide
- `DEVELOPMENT.md` - Development setup

## Recommendation

If you're looking for current information, start with:
1. `../README.md` - Overview
2. `../WHATS_NEW.md` - What changed
3. `../REFACTORING_SUMMARY.md` - Technical details
4. `make help` - Training commands

For specific topics:
- **TSS**: See `TSS.md` and related files
- **API/UI**: See `API.md` and `UI_IMPLEMENTATION.md`
- **Testing**: See `TESTING.md`
- **Deployment**: See `DEPLOYMENT.md`
