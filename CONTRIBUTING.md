# Contributing to AlphaGomoku

Thank you for your interest in contributing to AlphaGomoku! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- **Bug Reports**: Submit detailed bug reports with reproduction steps
- **Feature Requests**: Propose new features or improvements
- **Code Contributions**: Fix bugs, implement features, improve performance
- **Documentation**: Improve docs, add examples, fix typos
- **Testing**: Add tests, improve test coverage
- **Research**: Share insights on training strategies, architectures

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/alphagomoku.git
   cd alphagomoku
   ```

2. **Create Environment**
   ```bash
   conda create -n alphagomoku python=3.12
   conda activate alphagomoku
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify Setup**
   ```bash
   make test
   ```

### Development Workflow

1. **Create a branch** for your feature/fix
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style

3. **Add tests** for new functionality

4. **Run tests** to ensure nothing breaks
   ```bash
   make test
   ```

5. **Commit your changes** with clear messages
   ```bash
   git commit -m "feat: add feature description"
   git commit -m "fix: fix bug description"
   git commit -m "docs: update documentation"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style

### Python Code

- Follow **PEP 8** style guide
- Use **type hints** for function signatures
- Maximum line length: **88 characters** (Black formatter)
- Use **docstrings** for public functions/classes

Example:
```python
def evaluate_position(
    board: np.ndarray,
    player: int,
    depth: int = 4
) -> float:
    """Evaluate board position for given player.

    Args:
        board: 15x15 board state
        player: Player to evaluate for (1 or -1)
        depth: Search depth for evaluation

    Returns:
        Position score (-1 to 1, higher is better)
    """
    # Implementation
    pass
```

### Formatting

We use automated formatters:

```bash
# Format code
black alphagomoku/

# Sort imports
isort alphagomoku/

# Check linting
flake8 alphagomoku/ --max-line-length=88
```

## 🧪 Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names: `test_what_it_does`
- Test edge cases and error conditions
- Aim for >80% code coverage

Example test structure:
```python
def test_model_forward_pass_shape():
    """Test model outputs correct shapes"""
    model = GomokuNet.from_preset("small")
    batch = torch.randn(4, 5, 15, 15)

    policy, value = model(batch)

    assert policy.shape == (4, 225)  # 15x15
    assert value.shape == (4,)
```

### Running Tests

```bash
# All unit tests
make test

# Specific test file
pytest tests/unit/test_model.py -v

# With coverage
pytest tests/ --cov=alphagomoku --cov-report=html
```

## 📊 Performance Guidelines

When optimizing performance:

1. **Benchmark first**: Measure before optimizing
2. **Profile carefully**: Use `cProfile` or `line_profiler`
3. **Document trade-offs**: Speed vs memory vs accuracy
4. **Test thoroughly**: Ensure correctness isn't sacrificed

Example:
```python
# Add benchmarks for performance-critical code
import time

def benchmark_mcts_search():
    model = GomokuNet.from_preset("small")
    env = GomokuEnv()

    start = time.time()
    for _ in range(100):
        policy, value = mcts.search(env.board)
    elapsed = time.time() - start

    print(f"MCTS: {elapsed/100:.3f}s per search")
```

## 🎓 Research Contributions

If you're contributing research findings:

1. **Document methodology**: Clear description of experiments
2. **Share results**: Include metrics, graphs, comparisons
3. **Provide reproducibility**: Scripts, configs, seeds
4. **Discuss limitations**: What works, what doesn't

Example contribution:
- New training schedule that improves Elo by 100 points
- Include: training config, Elo progression graph, comparison
- Document: hardware used, training time, final strength

## 📖 Documentation Guidelines

### Writing Documentation

- Use **clear, concise language**
- Include **code examples** where helpful
- Add **links** to related docs
- Keep formatting consistent

### Documentation Types

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document public APIs
3. **Markdown Docs**: Guides, tutorials, specifications
4. **README Updates**: Keep main README current

## 🔄 Pull Request Process

### Before Submitting

- [ ] Code follows project style
- [ ] Tests added for new functionality
- [ ] All tests pass (`make test`)
- [ ] Documentation updated if needed
- [ ] Commit messages are clear

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or noted)
```

### Review Process

1. **Automated checks** run (tests, linting)
2. **Maintainer review** (usually within 1-3 days)
3. **Feedback addressed** via additional commits
4. **Approval and merge** when ready

## 🐛 Bug Reports

### Good Bug Reports Include

1. **Clear title**: "MCTS crashes with empty board"
2. **Environment**: OS, Python version, PyTorch version
3. **Steps to reproduce**: Minimal code example
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Logs/errors**: Full error messages

Example:
```markdown
## Bug: Model loading fails on Windows

**Environment:**
- OS: Windows 11
- Python: 3.12
- PyTorch: 2.0.0

**Steps to Reproduce:**
python
model = GomokuNet.from_preset("small")
model.load_state_dict(torch.load("checkpoint.pt"))


**Error:**
RuntimeError: Error(s) in loading state_dict...


**Expected:** Model loads successfully
```

## 💡 Feature Requests

### Good Feature Requests Include

1. **Use case**: Why is this needed?
2. **Proposed solution**: How might it work?
3. **Alternatives considered**: Other approaches?
4. **Additional context**: Examples, references

## 🤝 Code of Conduct

### Our Standards

- **Be respectful**: Treat everyone with respect
- **Be constructive**: Provide helpful feedback
- **Be collaborative**: Work together toward better code
- **Be patient**: Remember everyone was a beginner once

### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Publishing others' private information
- Trolling, insulting, or inflammatory comments
- Other unprofessional conduct

## 📞 Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainers directly (see README)
- **Development help**: Check docs/ folder

## 🎉 Recognition

Contributors are recognized in:
- Project README (major contributions)
- Release notes (feature contributions)
- Git history (all contributions)

Thank you for contributing to AlphaGomoku! 🚀
