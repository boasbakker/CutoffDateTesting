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
# Each returns only text→text models, filtered using API-provided metadata.
# ---------------------------------------------------------------------------

def _probe_openai_chat(client, model_id):
    """
    Probe whether an OpenAI model supports text→text generation.

    Tries chat completions first, then falls back to the Responses API
    for models that only support v1/responses (e.g. gpt-5-pro, o1-pro).
    Returns True if either endpoint works, False otherwise.
    """
    msg = [{"role": "user", "content": "hi"}]

    # Errors that indicate the model DOES support chat, it just hit a limit
    CHAT_SUPPORT_HINTS = ["max_tokens", "max_completion_tokens", "output limit"]

    def _is_limit_error(error_str):
        """Check if error indicates model supports chat but hit token limit."""
        return any(hint in error_str for hint in CHAT_SUPPORT_HINTS)

    def _try_responses_api():
        """Try the Responses API (v1/responses) as a fallback."""
        try:
            client.responses.create(
                model=model_id,
                input="hi",
                max_output_tokens=2,
            )
            return True
        except Exception as e2:
            return _is_limit_error(str(e2).lower())

    # Try chat completions first (max_completion_tokens for newer models)
    try:
        client.chat.completions.create(
            model=model_id, messages=msg, max_completion_tokens=2
        )
        return True
    except Exception as e:
        err = str(e).lower()
        if _is_limit_error(err):
            return True
        # Model is responses-only → try that API
        if "not supported" in err or "v1/responses" in err:
            return _try_responses_api()
        # Parameter name error → retry with old parameter name
        if "unsupported_parameter" in err:
            try:
                client.chat.completions.create(
                    model=model_id, messages=msg, max_tokens=2
                )
                return True
            except Exception as e2:
                err2 = str(e2).lower()
                if _is_limit_error(err2):
                    return True
                if "not supported" in err2 or "v1/responses" in err2:
                    return _try_responses_api()
                return False
        return False


def _fetch_openai_models():
    """
    Fetch text→text model IDs from OpenAI.

    OpenAI's list endpoint provides no capability metadata, so we probe each
    model with a minimal chat completion call to check support. Uses
    concurrent threads for speed.
    """
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = OpenAI()
    all_models = sorted(m.id for m in client.models.list())

    print(f"  Probing {len(all_models)} models for chat support...")
    text_models = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_probe_openai_chat, client, mid): mid
            for mid in all_models
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            mid = futures[future]
            if future.result():
                text_models.append(mid)
            # Progress indicator
            print(f"\r  Probed {done_count}/{len(all_models)} models...", end="", flush=True)
    print()  # newline after progress
    return sorted(text_models)


def _fetch_google_models():
    """
    Fetch text→text model IDs from Google Gemini.

    Uses the `supported_actions` metadata field to filter for models that
    support `generateContent` (i.e. text generation).
    """
    from google import genai
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    ids = []
    for m in client.models.list():
        # Only include models that support text generation
        actions = getattr(m, 'supported_actions', None) or []
        if 'generateContent' not in actions:
            continue
        name = m.name
        if name.startswith('models/'):
            name = name[len('models/'):]
        ids.append(name)
    return sorted(ids)


def _fetch_anthropic_models():
    """
    Fetch text→text model IDs from Anthropic.

    All models returned by Anthropic's list endpoint are chat models
    (text→text), so no additional filtering is needed.
    """
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
