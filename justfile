# ROMA-DSPy Justfile
# Common development commands

# Default recipe
default:
    @just --list

# Install dependencies
install:
    pip install -e .

# Run all tests
test:
    pytest tests/

# Run specific test file
test-file file:
    pytest tests/{{file}}

# Run tests with coverage
test-coverage:
    pytest tests/ --cov=src/roma_dspy --cov-report=html --cov-report=term

# Run tests in verbose mode
test-verbose:
    pytest tests/ -v

# Run unit tests
test-unit:
    pytest tests/unit/ -v

# Run integration tests
test-integration:
    pytest tests/integration/ -v

# Run linting
lint:
    ruff check src/ tests/

# Format code
format:
    ruff format src/ tests/

# Type check
typecheck:
    mypy src/roma_dspy

# Clean cache and build artifacts
clean:
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -rf .coverage
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Run the CLI
cli *args:
    python -m src.roma_dspy.cli {{args}}

# Start interactive Python session with imports
repl:
    python -c "from src.roma_dspy.engine.solve import solve, RecursiveSolver; from src.roma_dspy.modules import *; print('ROMA-DSPy modules loaded'); import IPython; IPython.start_ipython()"

# Run example notebooks
example name:
    jupyter nbconvert --to notebook --execute examples/{{name}}.ipynb --output {{name}}_executed.ipynb

# Start Jupyter notebook
notebook:
    jupyter notebook

# Build package
build:
    python -m build

# Run pre-commit checks
pre-commit:
    just format
    just lint
    just typecheck
    just test

# Setup development environment
setup:
    pip install -e .
    pip install pytest mypy ruff jupyter ipython coverage
    @echo "Development environment ready!"