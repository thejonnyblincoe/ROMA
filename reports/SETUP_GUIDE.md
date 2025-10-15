# ROMA-DSPy Setup Guide

## Overview

ROMA-DSPy now features a fully dynamic, one-command setup system that automatically discovers and configures all components based on your chosen profile.

## Quick Start

### One-Command Setup

```bash
# Interactive setup with profile selection
just setup

# Or run the script directly
./setup.sh

# Setup with specific profile
just setup crypto_agent
```

### Available Commands

```bash
# List all available profiles
just list-profiles

# Quick start with interactive prompts
just quick-start

# Minimal setup (skip optional components)
just setup-minimal

# Setup with specific profile
just setup-profile crypto_agent
```

## Dynamic Features

### Profile Discovery

The setup script automatically discovers all available profiles from `config/profiles/`:

- No hardcoded profile names
- Automatically detects new profiles you add
- Parses profiles to determine required environment variables
- Shows profile descriptions from YAML comments

### Environment Variable Discovery

The script intelligently parses your selected profile to determine what API keys are needed:

- **LLM Providers**: Detects OpenRouter, OpenAI, Anthropic, Google references
- **Toolkits**: Identifies E2B, Arkham, DefiLlama, Binance, CoinGecko requirements
- **Storage**: Detects S3/AWS requirements
- **Optional Services**: MLflow, custom configurations

### Docker Compose Discovery

Automatically finds and uses appropriate Docker Compose files:

- `docker-compose.yaml` (base)
- `docker-compose.override.yaml` (local overrides)
- `docker-compose.{environment}.yaml` (environment-specific)
- Profile-based compose profiles (e.g., observability)

## Adding Custom Profiles

### Create a New Profile

1. Create a new YAML file in `config/profiles/`:

```yaml
# config/profiles/my_custom.yaml
# description: My custom agent configuration

agents:
  executor:
    model: "openai/gpt-4"
    temperature: 0.7

toolkit:
  enabled_toolkits:
    - e2b_toolkit
    - web_search
```

2. Run setup with your profile:

```bash
just setup my_custom
```

The setup script will:
- Detect your new profile automatically
- Parse it to find required API keys
- Prompt only for what's needed
- Configure the environment accordingly

## Command Line Options

### Setup Script Options

```bash
./setup.sh [OPTIONS]

Options:
  --profile NAME       Use specific profile
  --env ENV           Set environment (development, production)
  --skip-e2b          Skip E2B template building
  --skip-s3           Skip S3 mount setup
  --skip-COMPONENT    Skip any component dynamically
  --help              Show help message
```

### Examples

```bash
# Production setup with crypto profile
./setup.sh --profile crypto_agent --env production

# Development setup without E2B
./setup.sh --skip-e2b

# Minimal setup for testing
./setup.sh --skip-e2b --skip-s3 --profile lightweight
```

## Environment Configuration

### Automatic Detection

The setup script automatically:

1. **Checks existing .env**: Preserves configured values
2. **Detects placeholders**: Identifies unconfigured values like `your_api_key_here`
3. **Prompts selectively**: Only asks for missing/placeholder values
4. **Creates backups**: Saves existing .env before reconfiguration

### Dynamic Variables

Based on your profile, the script will request:

- **Core Variables**: Always required (database password)
- **Profile-Specific**: Based on models and toolkits in profile
- **Optional Services**: S3, MLflow, custom variables
- **Custom Variables**: Add any additional variables interactively

## Advanced Usage

### Custom Docker Images

Set environment variables before running setup:

```bash
export DOCKER_IMAGE_NAME=my-roma
export DOCKER_IMAGE_TAG=v1.0
./setup.sh
```

### Multiple Environments

```bash
# Development
./setup.sh --env development

# Staging
./setup.sh --env staging

# Production
./setup.sh --env production
```

The script will look for corresponding Docker Compose files:
- `docker-compose.development.yaml`
- `docker-compose.staging.yaml`
- `docker-compose.production.yaml`

### CI/CD Integration

For automated deployments:

```bash
# Non-interactive setup with environment variables
export SELECTED_PROFILE=crypto_agent
export ROMA_ENV=production
export OPENROUTER_API_KEY=your_key
export E2B_API_KEY=your_key
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

./setup.sh --profile crypto_agent
```

## Extending the Setup

### Adding New Discovery Logic

The setup script uses discovery functions that can be extended:

- `discover_profiles()`: Finds available profiles
- `discover_required_env_vars()`: Parses profiles for requirements
- `discover_docker_compose_files()`: Finds compose configurations
- `discover_examples()`: Finds example configurations

### Custom Setup Steps

Add new steps to the main flow:

1. Create a new function in `setup.sh`
2. Add it to the main() function flow
3. Use `--skip-yourcomponent` to make it optional

### Profile Templates

Create template profiles in `config/examples/`:

```yaml
# config/examples/research_template.yaml
# Template for research-focused agents
# Copy to config/profiles/ and customize

agents:
  planner:
    model: "anthropic/claude-3.5-sonnet"
  executor:
    model: "openai/gpt-4"
```

## Troubleshooting

### Common Issues

1. **Docker daemon not running**
   ```bash
   # Start Docker first
   systemctl start docker  # Linux
   open /Applications/Docker.app  # macOS
   ```

2. **Profile not found**
   ```bash
   # Check available profiles
   just list-profiles

   # Verify profile file exists
   ls config/profiles/
   ```

3. **API key validation fails**
   ```bash
   # Check .env file
   cat .env | grep API_KEY

   # Reconfigure
   ./setup.sh --profile your_profile
   ```

### Debug Mode

Run with bash debugging:

```bash
bash -x ./setup.sh
```

### Manual Steps

If automated setup fails:

```bash
# 1. Copy and configure .env
cp .env.example .env
# Edit .env with your keys

# 2. Build Docker image
docker build -t roma-dspy:latest .

# 3. Start services
docker compose up -d

# 4. Check health
curl http://localhost:8000/health
```

## Best Practices

1. **Profile Organization**
   - Keep profiles focused and purposeful
   - Use descriptive names
   - Add comments for documentation
   - Store sensitive profiles in `.gitignore`'d directory

2. **Environment Management**
   - Use `.env.example` as template
   - Never commit `.env` with real keys
   - Use different profiles for dev/prod
   - Backup `.env` before updates

3. **Continuous Integration**
   - Use environment variables for secrets
   - Run setup in non-interactive mode
   - Validate deployment automatically
   - Keep profiles in version control

## Support

For issues or questions:

1. Check logs: `docker compose logs`
2. Review setup output for warnings
3. Verify prerequisites: `./setup.sh --help`
4. Check profile syntax: `yq eval config/profiles/your_profile.yaml`

The dynamic setup system ensures that ROMA-DSPy adapts to your configuration needs without requiring any hardcoded values or manual script modifications.