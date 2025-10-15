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

# Setup complete environment (one-command setup)
setup profile="":
    @echo "Running complete ROMA-DSPy setup..."
    @if [ -n "{{profile}}" ]; then \
        ./setup.sh --profile {{profile}}; \
    else \
        ./setup.sh; \
    fi

# Setup development environment (Python only)
setup-dev:
    pip install -e .
    pip install pytest mypy ruff jupyter ipython coverage
    @echo "Development environment ready!"

# ==============================================================================
# CLI Commands (via Docker)
# ==============================================================================

# Solve a task using ROMA-DSPy CLI
# Usage: just solve <task> [profile] [max_depth] [verbose] [output]
# Example: just solve "your task" crypto_agent 3 false text
solve task profile="crypto_agent" max_depth="3" verbose="false" output="text":
    @if [ "{{verbose}}" = "true" ]; then \
        docker exec -it roma-dspy-api roma-dspy solve --profile {{profile}} --max-depth {{max_depth}} --output {{output}} --verbose "{{task}}"; \
    else \
        docker exec -it roma-dspy-api roma-dspy solve --profile {{profile}} --max-depth {{max_depth}} --output {{output}} "{{task}}"; \
    fi

# Run any CLI command in the container (Docker)
cli-docker *args:
    docker exec -it roma-dspy-api roma-dspy {{args}}

# Visualize task execution DAG
# Usage: just visualize <execution_id> [visualizer_type] [format]
# Visualizer types: tree, timeline, statistics, context_flow, llm_trace
# Formats: text, json
# Example: just visualize abc123 tree text
# Example: just visualize abc123 llm_trace json
visualize execution_id visualizer_type="tree" format="text":
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type {{visualizer_type}} --format {{format}}

# Quick visualize with tree (default)
viz-tree execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type tree

# Visualize with timeline
viz-timeline execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type timeline

# Visualize with statistics
viz-stats execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type statistics

# Visualize with context flow
viz-context execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type context_flow

# Visualize with LLM trace (detailed execution trace)
viz-trace execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type llm_trace

# Export visualization as JSON
viz-json execution_id visualizer_type="tree":
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --visualizer-type {{visualizer_type}} --format json

# Ultra-detailed tree view (verbose with IDs, timing, tokens, no truncation)
viz-detailed execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --type tree --verbose --show-ids

# Full execution trace with no goal truncation
viz-full execution_id:
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --type llm_trace --max-goal-length 0 --verbose

# Tree view with custom goal length
viz-tree-custom execution_id max_length="100":
    docker exec -it roma-dspy-api roma-dspy visualize {{execution_id}} --type tree --max-goal-length {{max_length}}

# ==============================================================================
# Quick Setup Commands
# ==============================================================================

# One-command setup with everything
quick-start:
    @echo "Starting quick setup..."
    ./setup.sh

# Setup with specific profile
setup-profile profile:
    ./setup.sh --profile {{profile}}

# Setup without optional components
setup-minimal:
    ./setup.sh --skip-e2b --skip-s3

# List available profiles
list-profiles:
    @echo "Available configuration profiles:"
    @cd config/profiles && ls -1 *.yaml 2>/dev/null | sed 's/\.yaml$//' | sed 's/^/  - /'

# ==============================================================================
# Docker Commands
# ==============================================================================

# Build Docker image
docker-build:
    docker build -t roma-dspy:latest -f Dockerfile .

# Build Docker image with no cache
docker-build-clean:
    docker build --no-cache -t roma-dspy:latest -f Dockerfile .

# Start all services with docker-compose
docker-up:
    docker-compose up -d

# Start services with observability (MLflow)
docker-up-full:
    docker-compose --profile observability up -d

# Stop all services
docker-down:
    docker-compose down

# Stop and remove volumes
docker-down-clean:
    docker-compose down -v

# View logs for all services
docker-logs:
    docker-compose logs -f

# View logs for specific service
docker-logs-service service:
    docker-compose logs -f {{service}}

# Restart all services
docker-restart:
    docker-compose restart

# Check service status
docker-ps:
    docker-compose ps

# Execute command in roma-api container
docker-exec *args:
    docker-compose exec roma-api {{args}}

# Open shell in roma-api container
docker-shell:
    docker-compose exec roma-api bash

# Run database migrations
docker-migrate:
    docker-compose exec roma-api alembic upgrade head

# ==============================================================================
# S3 Storage Setup
# ==============================================================================

# Mount S3 bucket locally (requires goofys and AWS credentials)
s3-mount:
    @echo "Mounting S3 bucket for local development..."
    bash scripts/setup_local.sh

# Unmount S3 bucket
s3-unmount:
    @echo "Unmounting S3 bucket..."
    umount ${STORAGE_BASE_PATH:-${HOME}/.roma/s3_mount} || true

# Check S3 mount status
s3-status:
    @echo "Checking S3 mount status..."
    @mount | grep ${STORAGE_BASE_PATH:-${HOME}/.roma/s3_mount} || echo "S3 not mounted"

# ==============================================================================
# E2B Template Management
# ==============================================================================

# Build E2B sandbox template
e2b-build:
    @echo "Building E2B sandbox template..."
    cd docker/e2b && e2b template build

# List E2B templates
e2b-list:
    e2b template list

# Delete E2B template (use with caution)
e2b-delete template_id:
    e2b template delete {{template_id}}

# Test E2B sandbox connection (quick check)
e2b-test:
    @echo "Testing E2B sandbox creation..."
    python -c "from e2b_code_interpreter import Sandbox; s = Sandbox(); print(f'Sandbox created: {s.id}'); s.kill(); print('Test successful!')"

# Validate E2B template (comprehensive integration test)
e2b-validate:
    @echo "Running E2B template validation tests..."
    pytest tests/integration/test_e2b_template_validation.py -v

# ==============================================================================
# Production Deployment
# ==============================================================================

# Deploy to production (build and start all services)
deploy:
    @echo "Deploying ROMA-DSPy to production..."
    just docker-build
    just s3-mount
    just docker-up
    @echo "Deployment complete!"

# Full production deployment with observability
deploy-full:
    @echo "Deploying ROMA-DSPy with full observability stack..."
    just docker-build
    just s3-mount
    just docker-up-full
    @echo "Deployment complete!"

# Health check
health-check:
    @echo "Checking service health..."
    curl -f http://localhost:8000/health || echo "API not responding"
    docker-compose ps