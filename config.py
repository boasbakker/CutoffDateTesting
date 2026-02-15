"""
Configuration constants for the LLM cutoff date testing project.
"""

# Debug mode (can be set via --debug flag in main script)
DEBUG = False

# LLM Settings
TEMPERATURE = 0

# Max tokens per model type/thinking configuration
MAX_TOKENS_OPENAI = 5
MAX_TOKENS_CLAUDE = 2
MAX_TOKENS_GEMINI_MINIMAL = 5
MAX_TOKENS_LOW_REASONING = 200  # Used for all models when --reasoning is enabled, and gemini-3-pro (always low)
MAX_TOKENS_REASONING_RETRY = 800  # Max tokens to retry with if LOW_REASONING is not enough (OpenAI)

SYSTEM_PROMPT = 'Answer "Yes" or "No"'
SYSTEM_PROMPT_REASONING = 'Answer "Yes" or "No". Produce minimal reasoning.'

# Prompt template - use .format(name=..., description=...) to fill in
PROMPT_TEMPLATE = 'Is {name}, {description}, still alive?'

# Structured output schema: forces LLM to respond with {"answer": true/false}.
# answer=true means "alive" (doesn't know about death), answer=false means "dead" (knows about death).
STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "boolean"}
    },
    "required": ["answer"],
    "additionalProperties": False
}

# Default models
DEFAULT_MODEL_OPENAI = "gpt-5.2"
DEFAULT_MODEL_GEMINI = "gemini-3-flash-preview"
DEFAULT_MODEL_CLAUDE = "claude-opus-4-5-20250929"
