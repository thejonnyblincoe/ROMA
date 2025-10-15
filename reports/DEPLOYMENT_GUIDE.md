# ROMA-DSPy Crypto Agent Deployment Guide

Complete guide for deploying ROMA-DSPy crypto analysis agent with Docker, E2B code execution, and S3 storage.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [E2B Template Setup](#e2b-template-setup)
4. [S3 Storage Setup](#s3-storage-setup)
5. [Docker Deployment](#docker-deployment)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

1. **Docker & Docker Compose**
   ```bash
   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Just** (command runner)
   ```bash
   # macOS
   brew install just

   # Linux
   curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash
   ```

3. **goofys** (S3 filesystem)
   ```bash
   # macOS
   brew install goofys

   # Linux
   wget https://github.com/kahing/goofys/releases/latest/download/goofys
   chmod +x goofys
   sudo mv goofys /usr/local/bin/
   ```

4. **E2B CLI** (for sandbox templates)
   ```bash
   npm install -g @e2b/cli
   # or
   pnpm install -g @e2b/cli
   ```

5. **AWS CLI** (for S3 access)
   ```bash
   # macOS
   brew install awscli

   # Configure credentials
   aws configure
   ```

### Required API Keys

You will need the following API keys:

- **Anthropic API Key** - For Claude Sonnet 4.5 (executor)
- **Google API Key** - For Gemini 2.5 Flash (other agents)
- **E2B API Key** - For code execution sandboxes
- **AWS Credentials** - For S3 storage
- **Crypto API Keys** (optional but recommended):
  - Arkham Intelligence API Key
  - CoinGecko API Key (Pro optional)
  - DefiLlama API Key (Pro optional)
  - Binance API Key + Secret (optional, for trading data)

---

## Configuration

### 1. Create Environment File

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

### 2. Essential Configuration

Edit `.env` file with your values:

```bash
# ============================================================================
# LLM API Keys (REQUIRED)
# ============================================================================

# Claude Sonnet 4.5 for high-quality execution
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here

# Gemini 2.5 Flash for fast planning/verification
GOOGLE_API_KEY=your_google_api_key_here

# ============================================================================
# E2B Configuration (REQUIRED for code execution)
# ============================================================================

# Get from: https://e2b.dev/dashboard
E2B_API_KEY=your_e2b_api_key_here

# Will be set after building E2B template (see next section)
E2B_TEMPLATE_ID=roma-dspy-sandbox

# ============================================================================
# AWS & S3 Storage (REQUIRED)
# ============================================================================

# S3 bucket for persistent storage
ROMA_S3_BUCKET=your-roma-storage-bucket

# AWS credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Storage mount path (same on host and E2B)
STORAGE_BASE_PATH=/opt/sentient

# ============================================================================
# Database (REQUIRED for Docker deployment)
# ============================================================================

POSTGRES_ENABLED=true
POSTGRES_DB=roma_dspy
POSTGRES_USER=postgres
POSTGRES_PASSWORD=choose_secure_password_here
POSTGRES_PORT=5432

# ============================================================================
# Crypto API Keys (OPTIONAL but recommended)
# ============================================================================

ARKHAM_API_KEY=your_arkham_key_here
COINGECKO_API_KEY=your_coingecko_key_here
DEFILLAMA_API_KEY=your_defillama_key_here
BINANCE_API_KEY=your_binance_key_here
BINANCE_API_SECRET=your_binance_secret_here
```

### 3. Model Configuration

The crypto agent profile (`config/profiles/crypto_agent.yaml`) is pre-configured with:

- **Executor**: `claude-sonnet-4-5` (Anthropic's latest, best for coding and tools)
- **Planner/Aggregator/Verifier**: `gemini-2.5-flash` (Google's latest, fast and efficient)

**Model Names Verified (as of 2025):**
- Claude: `anthropic/claude-sonnet-4-5` or `anthropic/claude-sonnet-4-5-20250929`
- Gemini: `gemini/gemini-2.5-flash` or `gemini/gemini-2.5-flash-latest`

The config uses DSPy model naming conventions which will be automatically resolved.

---

## E2B Template Setup

E2B sandboxes require a custom template with S3 mounting capabilities.

### 1. Build E2B Template

```bash
# Using just
just e2b-build

# Or manually
cd docker/e2b
e2b template build
```

This will:
- Build Docker image with goofys and S3 support
- Create E2B template
- Output a template ID like: `abc123def456`

### 2. Update Template ID

Copy the template ID from the build output and update your `.env`:

```bash
E2B_TEMPLATE_ID=abc123def456  # Your actual template ID
```

### 3. Verify Template

```bash
# List templates
just e2b-list

# Test sandbox creation
just e2b-test
```

**Expected output:**
```
Sandbox created: sandbox_xyz123
Test successful!
```

---

## S3 Storage Setup

### 1. Create S3 Bucket

```bash
# Using AWS CLI
aws s3 mb s3://your-roma-storage-bucket --region us-east-1

# Set lifecycle policy (optional, for cost optimization)
aws s3api put-bucket-lifecycle-configuration \
  --bucket your-roma-storage-bucket \
  --lifecycle-configuration file://s3-lifecycle.json
```

### 2. Configure Bucket Permissions

The bucket should allow:
- Read/write access for your AWS credentials
- Private access (not public)

Example IAM policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-roma-storage-bucket",
        "arn:aws:s3:::your-roma-storage-bucket/*"
      ]
    }
  ]
}
```

### 3. Mount S3 Locally (for local development)

```bash
# Using just
just s3-mount

# Or manually
bash scripts/setup_local.sh
```

This will:
- Mount S3 bucket using goofys
- Create symlink to `/opt/sentient` (or your STORAGE_BASE_PATH)
- Verify write access

### 4. Verify S3 Mount

```bash
# Check mount status
just s3-status

# Test write access
echo "test" > /opt/sentient/executions/.test
cat /opt/sentient/executions/.test
rm /opt/sentient/executions/.test
```

---

## Docker Deployment

### 1. Build Docker Image

```bash
# Using just
just docker-build

# Or manually
docker build -t roma-dspy:latest -f Dockerfile .
```

### 2. Start Services

```bash
# Start core services (Postgres + API)
just docker-up

# Or with MLflow observability
just docker-up-full
```

Services started:
- **postgres**: PostgreSQL database (port 5432)
- **roma-api**: ROMA-DSPy API server (port 8000)
- **mlflow**: MLflow tracking server (port 5000, optional)

### 3. Verify Services

```bash
# Check service status
just docker-ps

# View logs
just docker-logs

# Health check
just health-check
```

**Expected output:**
```
Checking service health...
{"status":"healthy","version":"0.1.0","database":"connected"}
NAME                 IMAGE              STATUS
roma-dspy-postgres   postgres:16-alpine Up (healthy)
roma-dspy-api        roma-dspy:latest   Up (healthy)
```

### 4. Run Database Migrations

```bash
just docker-migrate
```

---

## Verification

### 1. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/api/v1/models

# Submit a test task
curl -X POST http://localhost:8000/api/v1/executions \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Get the current price of Bitcoin",
    "config_profile": "crypto_agent"
  }'
```

### 2. Test E2B Code Execution

```bash
# Open shell in container
just docker-shell

# Test E2B from inside container
python -c "
from roma_dspy.tools.core.e2b import E2BToolkit
toolkit = E2BToolkit(config={'timeout': 60})
result = toolkit.run_python_code('print(2 + 2)')
print(result)
"
```

### 3. Test S3 Storage

```bash
# Create test file in container
just docker-exec bash -c "echo 'test' > /opt/sentient/executions/test.txt"

# Verify it appears in S3
aws s3 ls s3://your-roma-storage-bucket/executions/

# Verify it appears locally (if mounted)
cat /opt/sentient/executions/test.txt
```

### 4. Test Full Crypto Agent

```bash
# Using CLI from container
just docker-exec roma-dspy solve \
  --profile crypto_agent \
  "Analyze Bitcoin's price trend over the last 7 days and provide a summary"
```

---

## Troubleshooting

### E2B Issues

**Problem**: "E2B sandbox creation failed"

**Solutions**:
1. Verify E2B API key is correct
2. Check template ID matches the built template
3. Ensure environment variables are passed correctly:
   ```bash
   just docker-exec env | grep E2B
   ```

**Problem**: "S3 mount failed in E2B sandbox"

**Solutions**:
1. Check AWS credentials in `.env`
2. Verify S3 bucket exists and is accessible
3. Check E2B logs for goofys errors:
   ```bash
   just docker-exec roma-dspy solve --debug "test task"
   ```

### S3 Mount Issues

**Problem**: "S3 not mounted locally"

**Solutions**:
1. Check goofys is installed: `which goofys`
2. Verify AWS credentials: `aws s3 ls`
3. Check mount point exists: `ls -la /opt/sentient`
4. Unmount and remount:
   ```bash
   just s3-unmount
   just s3-mount
   ```

**Problem**: "Permission denied on S3"

**Solutions**:
1. Check IAM permissions for the AWS credentials
2. Verify bucket policy allows access
3. Check FUSE permissions: `cat /etc/fuse.conf` (should have `user_allow_other`)

### Docker Issues

**Problem**: "API not responding"

**Solutions**:
1. Check container logs: `just docker-logs-service roma-api`
2. Verify all services are healthy: `just docker-ps`
3. Check database connection: `just docker-exec env | grep DATABASE_URL`
4. Restart services: `just docker-restart`

**Problem**: "Database connection failed"

**Solutions**:
1. Wait for postgres to be healthy: `docker-compose ps postgres`
2. Check postgres logs: `just docker-logs-service postgres`
3. Verify DATABASE_URL is correct in `.env`
4. Manually test connection:
   ```bash
   just docker-exec psql postgresql://postgres:password@postgres:5432/roma_dspy
   ```

### Model/API Issues

**Problem**: "LLM API call failed"

**Solutions**:
1. Verify API keys are set correctly
2. Check API key validity:
   ```bash
   # Test Anthropic
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model":"claude-sonnet-4-5","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'

   # Test Google
   curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY"
   ```

3. Check rate limits and quotas
4. Verify model names in config match current API versions

### Crypto Toolkit Issues

**Problem**: "Crypto API calls failing"

**Solutions**:
1. Verify crypto API keys are set
2. Check rate limits for each service
3. Enable debug logging:
   ```bash
   # In .env
   LOG_LEVEL=DEBUG
   ```
4. Test individual toolkits:
   ```python
   from roma_dspy.tools.crypto.coingecko import CoinGeckoToolkit
   toolkit = CoinGeckoToolkit(config={})
   result = toolkit.get_coin_price(coin_id="bitcoin", vs_currency="usd")
   print(result)
   ```

---

## Production Checklist

Before deploying to production:

- [ ] All API keys configured and tested
- [ ] S3 bucket created with proper permissions
- [ ] E2B template built and ID configured
- [ ] Database password changed from default
- [ ] Health checks passing
- [ ] Logs configured for production (JSON format)
- [ ] Observability (MLflow) configured if needed
- [ ] Resource limits set in docker-compose.yaml
- [ ] Backup strategy for PostgreSQL data
- [ ] Monitoring/alerting configured
- [ ] SSL/TLS configured for API endpoints
- [ ] Firewall rules configured

---

## Quick Reference Commands

```bash
# Deployment
just deploy                  # Full deployment
just deploy-full            # With observability

# Development
just docker-build           # Build image
just docker-up              # Start services
just docker-down            # Stop services
just docker-restart         # Restart services
just docker-shell           # Open shell

# Monitoring
just docker-logs            # View all logs
just docker-ps              # Service status
just health-check           # Test health

# S3
just s3-mount               # Mount S3 locally
just s3-status              # Check mount status

# E2B
just e2b-build              # Build template
just e2b-test               # Test sandbox
just e2b-list               # List templates

# Maintenance
just docker-down-clean      # Remove volumes
just docker-migrate         # Run migrations
```

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/roma-dspy/issues
- Documentation: Check CLAUDE.md for development guidelines
- E2B Docs: https://e2b.dev/docs
- DSPy Docs: https://dspy-docs.vercel.app/