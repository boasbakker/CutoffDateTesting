"""
Interactive model selection interface.

Guides the user through selecting an LLM model using single-keypress input at every step:
  1. Provider  (OpenAI / Google / Anthropic)
  2. Series    (e.g. GPT-4.x, Gemini 2.x, Claude 3.x, …)
  3. Level     (e.g. Flash / Pro, Haiku / Sonnet / Opus, …)
  4. Version   (specific dated snapshot like claude-sonnet-4-20250514)

Can be imported (`from select_model import select_model`) or run directly.
"""

import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Cross-platform single-keypress input
# ---------------------------------------------------------------------------

def _getch():
    """Read a single keypress without requiring Enter. Cross-platform."""
    try:
        import msvcrt
        ch = msvcrt.getch()
        # Handle special keys (arrow keys etc.) — ignore them
        if ch in (b'\x00', b'\xe0'):
            msvcrt.getch()  # consume the second byte
            return None
        return ch.decode('utf-8', errors='ignore')
    except ImportError:
        # Unix/macOS fallback
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def keypress_select(options, prompt_title="Select an option"):
    """
    Display a numbered menu and wait for a single keypress to select.

    Parameters:
        options: list of (label, value) tuples.
        prompt_title: header text shown above the menu.

    Returns:
        The `value` from the selected (label, value) tuple.

    Uses keys 1-9 then a-z for up to 35 options. If only one option exists,
    auto-selects it.
    """
    if not options:
        print("  No options available.")
        return None

    # Auto-select if only one option
    if len(options) == 1:
        label, value = options[0]
        print(f"\n{prompt_title}")
        print(f"  Auto-selected: {label}")
        return value

    # Build key mapping: 1-9, then a-z
    keys = [str(i) for i in range(1, 10)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
    if len(options) > len(keys):
        # Extremely unlikely but handle gracefully: truncate
        print(f"  Warning: too many options ({len(options)}), showing first {len(keys)}")
        options = options[:len(keys)]

    key_map = {}
    print(f"\n{prompt_title}")
    for i, (label, value) in enumerate(options):
        key = keys[i]
        key_map[key] = value
        print(f"  [{key}] {label}")

    print()
    while True:
        ch = _getch()
        if ch is None:
            continue
        ch = ch.lower()
        if ch in key_map:
            # Echo the selection
            # Find the label for display
            idx = keys.index(ch)
            if idx < len(options):
                print(f"  → {options[idx][0]}")
            return key_map[ch]
        # Ctrl+C / Escape
        if ch in ('\x03', '\x1b'):
            print("\n  Cancelled.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Model fetching — one function per provider
# ---------------------------------------------------------------------------

# OpenAI model ID prefixes/suffixes that are NOT text→text
_OPENAI_EXCLUDE_PREFIXES = ('dall-e', 'tts-', 'whisper-', 'text-embedding', 'sora-', 'omni-moderation')
_OPENAI_EXCLUDE_SUFFIXES = ('-tts', '-transcribe', '-realtime', '-image')
_OPENAI_EXCLUDE_CONTAINS = ('-tts-', '-transcribe-', '-realtime-', '-audio', '-image-', 'chatgpt-image')


def _is_openai_text_model(model_id):
    """Return True if an OpenAI model ID is a text→text model (not image/audio/embedding/moderation)."""
    mid = model_id.lower()
    for prefix in _OPENAI_EXCLUDE_PREFIXES:
        if mid.startswith(prefix):
            return False
    for suffix in _OPENAI_EXCLUDE_SUFFIXES:
        if mid.endswith(suffix):
            return False
    for pattern in _OPENAI_EXCLUDE_CONTAINS:
        if pattern in mid:
            return False
    return True


def _fetch_openai_models():
    """Fetch text→text model IDs from OpenAI (excludes image/audio/embedding/moderation models)."""
    from openai import OpenAI
    client = OpenAI()
    models = client.models.list()
    return sorted(m.id for m in models if _is_openai_text_model(m.id))


# Google model name patterns that are NOT text→text
_GOOGLE_EXCLUDE_CONTAINS = ('imagen', 'veo-', 'embedding', '-tts', 'native-audio', '-image')


def _is_google_text_model(model):
    """Return True if a Google model supports text→text generation (based on supported_actions and name)."""
    actions = model.supported_actions or []
    name = (model.name or '').lower()
    # Must support generateContent (the standard text generation action)
    if 'generateContent' not in actions:
        return False
    # Exclude models that only have bidiGenerateContent (live/streaming audio)
    if 'bidiGenerateContent' in actions and 'createCachedContent' not in actions:
        return False
    # Exclude by name patterns
    for pattern in _GOOGLE_EXCLUDE_CONTAINS:
        if pattern in name:
            return False
    # Exclude the special 'aqa' model (question-answering only)
    if name.endswith('/aqa') or name == 'aqa':
        return False
    return True


def _fetch_google_models():
    """Fetch text→text model IDs from Google (excludes image/video/embedding/audio-only models)."""
    from google import genai
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    models = list(client.models.list())
    ids = []
    for m in models:
        if not _is_google_text_model(m):
            continue
        name = m.name
        if name.startswith('models/'):
            name = name[len('models/'):]
        ids.append(name)
    return sorted(ids)


def _fetch_anthropic_models():
    """Fetch all model IDs from Anthropic. Returns a sorted list of model ID strings."""
    import anthropic
    client = anthropic.Anthropic()
    result = client.models.list()
    return sorted(m.id for m in result.data)


# ---------------------------------------------------------------------------
# Classification helpers — map model IDs → series and level
# ---------------------------------------------------------------------------

def _classify_openai(model_id):
    """
    Classify an OpenAI model ID into (series, level).

    Series: 'GPT-3.x', 'GPT-4.x', 'GPT-5.x', 'o-series', 'Other'
    Level:  'mini', 'standard', 'pro', 'other'
    """
    mid = model_id.lower()

    # o-series models (o1, o2, o3, o4, …)
    if re.match(r'^o\d', mid):
        series = 'o-series'
        if 'mini' in mid:
            level = 'mini'
        elif 'pro' in mid:
            level = 'pro'
        else:
            level = 'standard'
        return series, level

    # GPT models
    if 'gpt' in mid or mid in ('babbage-002', 'davinci-002'):
        # Determine major version
        ver_match = re.search(r'gpt-?(\d+)', mid)
        if ver_match:
            major = int(ver_match.group(1))
        elif 'babbage' in mid or 'davinci' in mid:
            major = 3  # legacy GPT-3 era
        else:
            major = 0

        if major <= 3:
            series = 'GPT-3.x (legacy)'
        elif major == 4:
            series = 'GPT-4.x'
        elif major >= 5:
            series = f'GPT-{major}.x'
        else:
            series = 'GPT (other)'

        if 'mini' in mid:
            level = 'mini'
        elif 'pro' in mid:
            level = 'pro'
        else:
            level = 'standard'
        return series, level

    # Catch-all
    return 'Other', 'other'


def _classify_google(model_id):
    """
    Classify a Google model ID into (series, level).

    Series: 'Gemini 1.x', 'Gemini 2.x', 'Gemini 3.x', 'Other'
    Level:  'flash-lite', 'flash', 'pro', 'nano', 'other'
    """
    mid = model_id.lower()

    if 'gemini' in mid:
        ver_match = re.search(r'gemini[- ]?(\d+)', mid)
        if ver_match:
            major = int(ver_match.group(1))
            series = f'Gemini {major}.x'
        else:
            series = 'Gemini (other)'

        if 'flash-lite' in mid or 'flash_lite' in mid:
            level = 'flash-lite'
        elif 'flash' in mid:
            level = 'flash'
        elif 'pro' in mid:
            level = 'pro'
        elif 'nano' in mid:
            level = 'nano'
        else:
            level = 'other'
        return series, level

    return 'Other', 'other'


def _classify_anthropic(model_id):
    """
    Classify an Anthropic model ID into (series, level).

    Series: 'Claude 1.x', 'Claude 2.x', 'Claude 3.x', 'Claude 4.x', 'Other'
    Level:  'haiku', 'sonnet', 'opus', 'other'
    """
    mid = model_id.lower()

    if 'claude' in mid:
        # Try to find a version number like claude-3-5 or claude-3.5 or claude-4
        # Anthropic uses patterns like claude-sonnet-4-... or claude-3-5-sonnet-...
        # We want the major family version
        
        # Pattern: claude-<tier>-<major>-... (newer naming: claude-sonnet-4-20250514)
        ver_match = re.search(r'claude-(?:haiku|sonnet|opus)-(\d+)', mid)
        if not ver_match:
            # Pattern: claude-<major>-... (older naming: claude-3-5-sonnet-...)  
            ver_match = re.search(r'claude-(\d+)', mid)
        
        if ver_match:
            major = int(ver_match.group(1))
            series = f'Claude {major}.x'
        else:
            series = 'Claude (other)'

        if 'haiku' in mid:
            level = 'haiku'
        elif 'sonnet' in mid:
            level = 'sonnet'
        elif 'opus' in mid:
            level = 'opus'
        else:
            level = 'other'
        return series, level

    return 'Other', 'other'


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------

# Provider definitions: (display name, fetch function, classify function)
PROVIDERS = [
    ('OpenAI', _fetch_openai_models, _classify_openai),
    ('Google (Gemini)', _fetch_google_models, _classify_google),
    ('Anthropic (Claude)', _fetch_anthropic_models, _classify_anthropic),
]


def select_model():
    """
    Interactive model selection.

    Walks the user through provider → series → level → version using
    single-keypress input at every step.

    Returns:
        str: the selected model ID (e.g. 'claude-sonnet-4-20250514').
    """

    # Step 1: provider
    provider_options = [(name, i) for i, (name, _, _) in enumerate(PROVIDERS)]
    provider_idx = keypress_select(provider_options, "Select provider")
    if provider_idx is None:
        return None

    provider_name, fetch_fn, classify_fn = PROVIDERS[provider_idx]

    # Step 2: fetch models
    print(f"\n  Fetching models from {provider_name}...")
    try:
        model_ids = fetch_fn()
    except Exception as e:
        print(f"  Error fetching models: {e}")
        print("  Make sure the API key environment variable is set.")
        return None

    if not model_ids:
        print("  No models returned by the API.")
        return None

    print(f"  Found {len(model_ids)} models.")

    # Classify all models into (series, level)
    classified = []  # list of (model_id, series, level)
    for mid in model_ids:
        series, level = classify_fn(mid)
        classified.append((mid, series, level))

    # Step 3: series selection
    series_groups = defaultdict(list)
    for mid, series, level in classified:
        series_groups[series].append((mid, level))

    series_options = [
        (f"{s} ({len(models)} models)", s)
        for s, models in sorted(series_groups.items())
    ]
    selected_series = keypress_select(series_options, "Select series")
    if selected_series is None:
        return None

    models_in_series = series_groups[selected_series]

    # Step 4: level/tier selection (skip if all models share one level)
    level_groups = defaultdict(list)
    for mid, level in models_in_series:
        level_groups[level].append(mid)

    if len(level_groups) > 1:
        level_options = [
            (f"{lvl} ({len(mids)} models)", lvl)
            for lvl, mids in sorted(level_groups.items())
        ]
        selected_level = keypress_select(level_options, "Select level/tier")
        if selected_level is None:
            return None
        remaining = level_groups[selected_level]
    else:
        # Only one level — skip the prompt
        remaining = list(level_groups.values())[0]

    # Step 5: specific version
    version_options = [(mid, mid) for mid in sorted(remaining)]
    selected = keypress_select(version_options, "Select specific model")
    return selected


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    result = select_model()
    if result:
        print(f"\n✓ Selected model: {result}")
    else:
        print("\n✗ No model selected.")
