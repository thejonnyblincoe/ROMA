# ROMA-DSPy Crypto Agent - Quick Start

Get running in 5 minutes on your local machine!

---

## Prerequisites

Install these first:

```bash
# macOS
brew install docker goofys

# Linux
# Install Docker: https://docs.docker.com/engine/install/
# Install goofys: wget https://github.com/kahing/goofys/releases/latest/download/goofys && chmod +x goofys && sudo mv goofys /usr/local/bin/
```

---

## Step 1: Configure Environment (2 minutes)

```bash
# Create .env from template
cp .env.example .env

# Edit .env and add these required values:
nano .env  # or use your preferred editor
```

**Required configuration in .env:**

```bash
# LLM Access (OpenRouter - easiest option)
OPENROUTER_API_KEY=your_openrouter_key_here

# E2B Code Execution
E2B_API_KEY=your_e2b_api_key_here

# AWS S3 Storage (using existing roma-shared bucket)
ROMA_S3_BUCKET=roma-shared
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Database (change password)
POSTGRES_PASSWORD=choose_secure_password
```

**Get API keys:**
- OpenRouter: https://openrouter.ai/keys
- E2B: https://e2b.dev/dashboard
- AWS: Should already have these for roma-shared bucket

---

## Step 2: Build E2B Template (1 minute)

```bash
# Build custom E2B sandbox with S3 support
just e2b-build

# Copy the template ID from output and add to .env:
# Example output: "Template ID: abc123def456"
# Add to .env: E2B_TEMPLATE_ID=abc123def456
```

---

## Step 3: Mount S3 (30 seconds)

```bash
# Mount roma-shared bucket locally
just s3-mount

# Verify it worked
ls -la /opt/sentient/executions
```

**Note:** This mounts the existing `roma-shared` S3 bucket to `/opt/sentient` on your Mac.

---

## Step 4: Deploy (1 minute)

```bash
# Build and start all services
just deploy

# Wait for services to start (~30 seconds)
```

---

## Step 5: Test (30 seconds)

```bash
# Check health
curl http://localhost:8000/health

# Test with crypto query
curl -X POST http://localhost:8000/api/v1/executions \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Get the current price of Bitcoin in USD",
    "config_profile": "crypto_agent"
  }'
```

---

## ✅ You're Running!

**Services:**
- API: http://localhost:8000
- PostgreSQL: localhost:5432
- Storage: /opt/sentient (mounted from roma-shared)

**Quick Commands:**
```bash
# CLI Usage (runs in Docker container)
just solve "Get Bitcoin price"                      # Run task with default profile
just solve "Analyze Ethereum" crypto_agent          # With specific profile
just cli config show                                # Run any CLI command
just visualize <execution_id>                       # Visualize execution DAG

# Docker Management
just docker-logs      # View logs
just docker-ps        # Check status
just docker-shell     # Open shell in container
just docker-down      # Stop services
just s3-status        # Check S3 mount
```

---

## Common Issues

### "S3 mount failed"
```bash
# Check AWS credentials
aws s3 ls s3://roma-shared

# Remount
just s3-unmount
just s3-mount
```

### "E2B template not found"
```bash
# List templates
just e2b-list

# Rebuild
just e2b-build
```

### "API not responding"
```bash
# Check logs
just docker-logs-service roma-api

# Restart
just docker-restart
```

---

## What's Running?

```
Your Mac
├── S3 Mount: /opt/sentient → s3://roma-shared (via goofys)
└── Docker Containers
    ├── postgres (database)
    └── roma-api (API server)
        └── Uses E2B sandboxes
            └── Each sandbox mounts same S3 bucket
```

**All file operations** (from host, Docker, or E2B) **write to the same S3 bucket** for persistence!

---

## Next Steps

**Test crypto toolkits:**
```bash
# Via CLI (easiest)
just solve "Analyze Bitcoin price trend over last 7 days"

# Or via API
curl -X POST http://localhost:8000/api/v1/executions \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Analyze Bitcoin price trend over last 7 days and calculate moving averages",
    "config_profile": "crypto_agent"
  }'

# Check execution files
ls -la /opt/sentient/executions/
```

**Monitor execution:**
```bash
# Watch logs in real-time
just docker-logs

# Check specific service
just docker-logs-service roma-api
```

**Customize:**
- Models: Edit `config/profiles/crypto_agent.yaml`
- Toolkits: Enable/disable in same file
- Settings: Adjust in `.env`

---

## Full Documentation

- **Deployment Guide:** DEPLOYMENT_GUIDE.md (comprehensive instructions)
- **Deployment Summary:** DEPLOYMENT_SUMMARY.md (architecture overview)
- **Commands:** Run `just` to see all available commands
- **Config:** .env.example for all configuration options

---

**Need help?** Check DEPLOYMENT_GUIDE.md for detailed troubleshooting!
