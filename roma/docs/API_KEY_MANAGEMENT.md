# API Key Management in ROMA

ROMA uses a smart, on-demand API key management system that only requires API keys for the LLM providers you actually use in your configuration.

## Key Principles

1. **On-Demand Loading**: API keys are only checked when creating models that need them
2. **Graceful Degradation**: Models are created even without API keys (with warnings)
3. **Flexible Configuration**: API keys can be provided via environment variables or model config
4. **Provider-Specific**: Only set keys for providers you actually use

## How It Works

### Model Creation Process

1. **Model Config Check**: If `api_key` is provided in the model config, use it directly
2. **Environment Lookup**: If no config key, look for provider-specific environment variable
3. **Warning on Missing**: If key is required but missing, create model with warning
4. **No Key Needed**: Some models (like Ollama) don't require API keys

### Provider Detection

ROMA automatically detects the required API key based on the model ID:

| Model ID Pattern | Provider | Required Env Var |
|------------------|----------|------------------|
| `gpt-4`, `gpt-3.5-turbo` | OpenAI | `OPENAI_API_KEY` |
| `claude-3-5-sonnet` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini-pro` | Google | `GOOGLE_API_KEY` |
| `groq/llama3-70b` | Groq | `GROQ_API_KEY` |
| `ollama/llama2` | Ollama | None required |

## Configuration Options

### Option 1: Environment Variables (Recommended)

```bash
# Copy .env.example to .env and uncomment what you need
cp .env.example .env

# Edit .env to add only the keys you need
OPENAI_API_KEY=sk-your-actual-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key  # Only if using Claude models
```

### Option 2: Model Config (Per-Model)

```yaml
# In your agent config
model:
  provider: litellm
  model_id: gpt-4o
  api_key: sk-your-api-key-here  # Overrides environment
  temperature: 0.7
```

### Option 3: Mixed Approach

```yaml
# Use environment for most providers
model:
  provider: litellm
  model_id: gpt-4o
  temperature: 0.7

# Override for specific models
special_model:
  provider: litellm
  model_id: claude-3-5-sonnet
  api_key: sk-different-key-here  # Override for this model only
```

## Common Use Cases

### Minimal Setup (Single Provider)

If you only use OpenAI models:

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key
```

All other provider keys can be omitted.

### Multi-Provider Setup

```bash
# .env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
# Add more as needed
```

### Local Models Only

For Ollama or other local models:

```yaml
model:
  provider: litellm
  model_id: ollama/llama2
  api_base: http://localhost:11434
  # No API key needed
```

### Development/Testing

```bash
# .env
OPENAI_API_KEY=sk-test-dummy-key  # Will warn but won't fail
ROMA_TEST_MODE=true  # Use mock responses
```

## Best Practices

1. **Use Environment Variables**: Store real API keys in `.env`, not in config files
2. **Principle of Least Privilege**: Only set keys for providers you actually use
3. **Rotate Keys Regularly**: Update API keys periodically for security
4. **Monitor Usage**: Use provider dashboards to track API usage and costs
5. **Test Mode**: Use `ROMA_TEST_MODE=true` for development without real API calls

## Troubleshooting

### Warning: API key required but not found

```
API key required for model 'gpt-4o' but OPENAI_API_KEY not found in environment.
Model creation will proceed, but API calls may fail.
```

**Solution**: Set the required environment variable or provide `api_key` in model config.

### Model creation succeeds but API calls fail

This happens when the model is created without an API key. The failure occurs during actual LLM calls.

**Solution**: Add the appropriate API key for the provider you're using.

### Multiple keys for same provider

If you have multiple API keys for the same provider, use model config:

```yaml
production_model:
  api_key: sk-prod-key

development_model:
  api_key: sk-dev-key
```

## Security Notes

- Never commit API keys to version control
- Use `.env` files and add `.env` to `.gitignore`
- Rotate keys regularly
- Monitor API usage for unauthorized access
- Consider using provider-specific access controls and quotas