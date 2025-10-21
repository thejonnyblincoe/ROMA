# ROMA-DSPy Application Dockerfile
# Multi-stage build for optimized image size

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyproject.toml and README.md for dependency installation
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies (core + e2b + api + boto3 for S3/MinIO support)
RUN pip install --no-cache-dir -e ".[e2b,api]" boto3

# ============================================================================
# Final stage
# ============================================================================

FROM python:3.12-slim

# Install runtime dependencies including goofys for S3 mounting and Node.js for MCP servers
RUN apt-get update && apt-get install -y \
    curl \
    git \
    fuse \
    ca-certificates \
    postgresql-client \
    wget \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install goofys for S3 mounting
RUN curl -L https://github.com/kahing/goofys/releases/latest/download/goofys -o /usr/local/bin/goofys \
    && chmod +x /usr/local/bin/goofys

# Enable FUSE for non-root users
RUN echo "user_allow_other" >> /etc/fuse.conf

# Create application user
RUN useradd -m -u 1000 roma && mkdir -p /opt/sentient && chown roma:roma /opt/sentient

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=roma:roma . .

# Install the package in development mode (no deps, already copied)
RUN pip install --no-deps -e .

# Create necessary directories
RUN mkdir -p /app/.checkpoints /app/.cache /app/logs /app/executions \
    && chown -R roma:roma /app/.checkpoints /app/.cache /app/logs /app/executions \
    && mkdir -p /mlflow/artifacts \
    && chown -R roma:roma /mlflow

# Switch to non-root user
USER roma

# Set Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run API server
CMD ["roma-dspy", "server", "start", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
