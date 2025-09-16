# AlphaGomoku Testing Framework

This document describes the comprehensive testing framework for the AlphaGomoku project.

## Test Coverage Overview

Our testing framework now covers all major components with comprehensive test scenarios:

### Previous Coverage (4 test files, 21 tests)
- Basic environment functionality
- Basic model functionality
- Basic TSS functionality
- Basic TSS integration

### New Coverage (12+ test files, 200+ tests)
- **Unit Tests**: Individual component testing with error scenarios
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Regression and benchmark testing
- **Error Handling**: Edge cases and failure scenarios
- **Multiprocessing**: Parallel components testing

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_env.py            # Environment tests (existing)
│   ├── test_model.py          # Neural network tests (existing)
│   ├── test_mcts.py           # MCTS comprehensive tests (NEW)
│   ├── test_tss.py            # TSS tests (existing)
│   ├── test_trainer.py        # Training pipeline tests (NEW)
│   ├── test_data_buffer.py    # Data buffer tests (NEW)
│   ├── test_parallel_selfplay.py # Multiprocessing tests (NEW)
│   ├── test_evaluator.py      # Evaluation tests (NEW)
│   └── test_error_handling.py # Error scenarios (NEW)
├── integration/                # Integration tests
│   ├── test_tss_integration.py # TSS integration (existing)
│   └── test_mcts_integration.py # MCTS integration (NEW)
└── performance/                # Performance regression tests
    └── test_performance_regression.py # Benchmarks (NEW)
```

## Running Tests

### Quick Start

```bash
# Check dependencies
python run_tests.py --check-deps

# Run all tests
python run_tests.py

# Run fast tests only (exclude slow performance tests)
python run_tests.py --fast
```

### Specific Test Categories

```bash
# Unit tests only
python run_tests.py --unit

# Integration tests only
python run_tests.py --integration

# Performance tests only
python run_tests.py --performance

# With coverage reporting
python run_tests.py --coverage
```

### Component-Specific Tests

```bash
# Test specific components
python run_tests.py --component mcts
python run_tests.py --component tss
python run_tests.py --component trainer
python run_tests.py --component parallel
python run_tests.py --component errors
```

### Direct pytest Usage

```bash
# Run all tests
pytest

# Run with specific markers
pytest -m "unit and not slow"
pytest -m "mcts or tss"
pytest -m "performance"

# Run specific files
pytest tests/unit/test_mcts.py -v
pytest tests/integration/ -v

# Run with coverage
pytest --cov=alphagomoku --cov-report=html
```

## Test Categories and Markers

### Markers
- `unit`: Unit tests for individual components
- `integration`: Integration tests between components
- `performance`: Performance and regression tests
- `slow`: Slow-running tests (can be excluded)
- `gpu`: Tests requiring GPU
- `parallel`: Tests involving multiprocessing
- `mcts`: MCTS-specific tests
- `tss`: TSS-specific tests
- `training`: Training pipeline tests

### Test Types

#### Unit Tests
- **Environment Tests** (`test_env.py`): Board mechanics, move validation, win detection
- **Model Tests** (`test_model.py`): Network architecture, forward pass, parameter handling
- **MCTS Tests** (`test_mcts.py`): Tree search, batching, UCT scores, error handling
- **TSS Tests** (`test_tss.py`): Threat detection, forced moves, search algorithms
- **Trainer Tests** (`test_trainer.py`): Loss computation, optimization, gradient handling
- **Data Buffer Tests** (`test_data_buffer.py`): LMDB operations, data augmentation, sampling
- **Parallel Tests** (`test_parallel_selfplay.py`): Multiprocessing, worker management
- **Evaluator Tests** (`test_evaluator.py`): Game evaluation, strength measurement
- **Error Handling Tests** (`test_error_handling.py`): Edge cases, failure recovery

#### Integration Tests
- **TSS-MCTS Integration**: Tactical search coordination with tree search
- **Training Pipeline Integration**: Self-play → data buffer → training cycle
- **Model-MCTS Integration**: Neural network evaluation within tree search
- **Evaluation Integration**: End-to-end game playing and strength assessment

#### Performance Tests
- **Model Inference Benchmarks**: Speed and memory usage across model sizes
- **MCTS Performance**: Scaling with simulation counts and batch sizes
- **TSS Timing**: Search depth vs. time tradeoffs
- **Training Speed**: Batch processing efficiency
- **Memory Management**: Leak detection and resource usage
- **Parallel Efficiency**: Multiprocessing overhead and scaling

## Test Configuration

### pytest.ini
- Test discovery patterns
- Marker definitions
- Output formatting
- Timeout settings
- Warning filters

### conftest.py Fixtures
- `small_model`, `medium_model`: Different model sizes
- `small_env`, `medium_env`: Different board sizes
- `sample_training_data`: Mock training data
- `tactical_position`, `winning_position`: Specific board states
- `performance_monitor`: Performance tracking

## Expected Test Outcomes

### Unit Tests (80+ tests)
- **Environment**: 15+ tests covering board mechanics, validation, edge cases
- **Model**: 10+ tests covering architecture, inference, error handling
- **MCTS**: 25+ tests covering search, batching, tree operations, errors
- **TSS**: 20+ tests covering threat detection, search, integration
- **Trainer**: 15+ tests covering optimization, loss computation, stability
- **Data Buffer**: 20+ tests covering storage, sampling, persistence
- **Parallel**: 15+ tests covering multiprocessing, resource management
- **Error Handling**: 30+ tests covering edge cases across all components

### Integration Tests (20+ tests)
- **Component Interactions**: Verify components work together correctly
- **Data Flow**: Test data propagation through training pipeline
- **Performance Consistency**: Ensure integration doesn't degrade performance

### Performance Tests (15+ tests)
- **Regression Benchmarks**: Ensure performance doesn't degrade
- **Scaling Tests**: Verify performance scales appropriately
- **Memory Tests**: Detect memory leaks and excessive usage
- **Timeout Tests**: Ensure operations complete within time limits

## Quality Metrics

### Coverage Goals
- **Unit Test Coverage**: >90% line coverage for core components
- **Integration Coverage**: All major component interactions tested
- **Error Scenario Coverage**: All expected failure modes tested

### Performance Benchmarks
- **Model Inference**: <100ms for medium models, <1s for large models
- **MCTS Search**: Scales linearly with simulation count
- **TSS Analysis**: <200ms for tactical positions at depth 6
- **Training Step**: <50ms for typical batch sizes
- **Memory Usage**: <4GB total for training, <2GB for inference

### Reliability Standards
- **Error Recovery**: All components handle failures gracefully
- **Resource Cleanup**: No memory leaks or resource locks
- **Reproducibility**: Tests produce consistent results with fixed seeds
- **Parallel Safety**: Multiprocessing components are thread-safe

## CI/CD Integration

### GitHub Actions Integration
```yaml
- name: Run Fast Tests
  run: python run_tests.py --fast

- name: Run Full Test Suite
  run: python run_tests.py --skip-performance

- name: Performance Regression Tests
  run: python run_tests.py --performance
```

### Local Development Workflow
1. **Pre-commit**: Run fast tests (`python run_tests.py --fast`)
2. **Before PR**: Run full test suite (`python run_tests.py`)
3. **Performance Check**: Run performance tests periodically
4. **Coverage Check**: Generate coverage reports for code review

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure PYTHONPATH includes project root
2. **GPU Tests**: Skip with `-m "not gpu"` if no GPU available
3. **Slow Tests**: Use `--fast` flag to exclude performance tests
4. **Memory Issues**: Reduce batch sizes in test fixtures
5. **Timeout**: Increase timeout in pytest.ini for slow hardware

### Debug Commands
```bash
# Verbose output with no capture
pytest tests/unit/test_mcts.py -v -s

# Stop on first failure
pytest --maxfail=1

# Run specific test method
pytest tests/unit/test_mcts.py::TestMCTS::test_search_basic -v

# Debug with pdb
pytest --pdb tests/unit/test_mcts.py::TestMCTS::test_search_basic
```

## Future Enhancements

### Planned Additions
1. **Property-based Testing**: Use hypothesis for broader test coverage
2. **Fuzzing Tests**: Random input testing for robustness
3. **Load Testing**: High-throughput scenario testing
4. **Integration with Benchmarking Tools**: Automated performance tracking
5. **Visual Test Reports**: Dashboard for test results and trends

### Monitoring
- **Performance Tracking**: Automated benchmarking in CI
- **Coverage Tracking**: Coverage trend monitoring
- **Test Reliability**: Flaky test detection and resolution
- **Resource Usage**: Memory and CPU usage profiling during tests

This comprehensive testing framework ensures the reliability, performance, and maintainability of the AlphaGomoku codebase across all its components.