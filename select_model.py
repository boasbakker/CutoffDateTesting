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
    model with a minimal chat completion call to check support. Results are
    cached in a JSON file (openai_model_cache.json) next to this script so
    already-probed models are not re-tested on subsequent runs.
    """
    import json as _json
    from openai import OpenAI
    from concurrent.futures import ThreadPoolExecutor, as_completed

    CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "openai_model_cache.json")

    # Load existing cache: { model_id: bool (true=text-capable) }
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = _json.load(f)
        except (ValueError, OSError):
            cache = {}

    client = OpenAI()
    all_models = sorted(m.id for m in client.models.list())

    # Determine which models need probing
    to_probe = [mid for mid in all_models if mid not in cache]

    if to_probe:
        print(f"  Probing {len(to_probe)} new models for chat support "
              f"({len(all_models) - len(to_probe)} cached)...")
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(_probe_openai_chat, client, mid): mid
                for mid in to_probe
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                mid = futures[future]
                cache[mid] = future.result()
                print(f"\r  Probed {done_count}/{len(to_probe)} models...",
                      end="", flush=True)
        print()

        # Save updated cache
        try:
            with open(CACHE_FILE, 'w') as f:
                _json.dump(cache, f, indent=2, sort_keys=True)
        except OSError as e:
            print(f"  Warning: could not save cache: {e}")
    else:
        print(f"  All {len(all_models)} models found in cache.")

    return sorted(mid for mid in all_models if cache.get(mid, False))


def _probe_google_text_support(client, model_id):
    """
    Probe whether a Google model supports text→text generation.
    
    Returns True if the model successfully generates text from a text prompt.
    Returns False if it raises a modality error (e.g. "Audio only") or returns no text.
    """
    from google.genai import types
    
    try:
        # We use a very simple prompt with a small token limit to save cost/time
        response = client.models.generate_content(
            model=model_id,
            contents="hi",
            config=types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=0.0
            )
        )
        
        # Check if we got text back
        if response.text and response.text.strip():
            return True
            
        # If response.text is None, it might be an image/audio-only model returning blobs
        return False
        
    except Exception as e:
        # Check for specific error messages regarding supported capabilities
        err_str = str(e).lower()
        if "response modalities" in err_str and "text" in err_str and "not supported" in err_str:
            return False
        if "requires the use of the computer use tool" in err_str:
            return False
        
        # Other errors (e.g. 500, quota) -> assume True to be safe? 
        # Or False to be strict? User said "Models which don't support text->text must be ignored"
        # If we can't verify it, we probably shouldn't show it.
        # However, transient errors shouldn't delist valid models.
        # But for "invalid argument" type errors, it's definitely False.
        if "invalid_argument" in err_str or "not found" in err_str:
            return False
            
        # For other errors, we might want to log but maybe default to inclusion 
        # specifically if it feels like a network flake? 
        # Actually safer to exclude if we are strict about "must be ignored".
        return False


def _fetch_google_models():
    """
    Fetch text→text model IDs from Google Gemini.
    
    Probes models to ensure they actually support text→text generation,
    filtering out audio-only (TTS), image-only (Imagen), etc.
    Results are cached in google_model_cache.json.
    """
    import json as _json
    from google import genai
    from concurrent.futures import ThreadPoolExecutor, as_completed

    CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "google_model_cache.json")

    # Load existing cache: { model_id: bool (true=text-capable) }
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = _json.load(f)
        except (ValueError, OSError):
            cache = {}

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        # Fallback if no key, just return nothing or crash later.
        # But commonly we might want to just let the client creation fail or return empty.
        # The original code crashed on client creation if key was missing? 
        # No, client creation doesn't crash until you use it usually, but let's be safe.
        pass

    client = genai.Client(api_key=api_key)
    
    # 1. List all models
    all_models = []
    try:
        for m in client.models.list():
            name = m.name
            if name.startswith('models/'):
                name = name[len('models/'):]
            
            # fast-path filtering based on obvious names to avoid probing useless stuff
            if 'embedding' in name or 'bison' in name:
                continue
                
            # Filter by supported_actions broadly first
            actions = getattr(m, 'supported_actions', None) or []
            if 'generateContent' not in actions:
                continue
                
            all_models.append(name)
    except Exception:
        # if list fails (e.g. auth), return empty
        return []

    all_models.sort()

    # 2. Determine which need probing
    to_probe = [mid for mid in all_models if mid not in cache]

    if to_probe:
        print(f"  Probing {len(to_probe)} new Google models for text support...")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_probe_google_text_support, client, mid): mid
                for mid in to_probe
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                mid = futures[future]
                is_text = future.result()
                cache[mid] = is_text
                # Optional: print progress
                # print(f"\r  Probed {done_count}/{len(to_probe)}...", end="", flush=True)

        # Save updated cache
        try:
            with open(CACHE_FILE, 'w') as f:
                _json.dump(cache, f, indent=2, sort_keys=True)
        except OSError:
            pass
    
    # 3. Return only those that are True in cache
    # Also double check they are still in the list (in case they were deleted from API but in cache)
    # actually cache might have old models that don't exist anymore, which is fine, 
    # but we only return ones that are in `all_models` AND valid.
    
    # But wait, if we only return intersection, we lose the benefit of cache for models 
    # that might be transiently missing? No, `client.models.list()` is the source of truth for "existence".
    # Cache is source of truth for "capability".
    
    valid_ids = []
    for mid in all_models:
        if cache.get(mid, False):
            valid_ids.append(mid)
            
    return sorted(valid_ids)


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
    Classify an OpenAI model ID into (series, level, sublevel).

    Series:   'GPT-3.x (legacy)', 'GPT-4.x', 'GPT-5.x', 'o-series', 'Other'
    Level:    'nano', 'mini', 'standard', 'pro'
    Sublevel: '4'/'4o'/'4.1' for GPT-4.x standard, '5'/'5.1'/'5.2' for GPT-5.x standard, else None
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
        return series, level, None

    # GPT models (including chatgpt-* aliases)
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

        # Determine level/tier (nano/mini/standard/pro)
        if 'nano' in mid:
            level = 'nano'
        elif 'mini' in mid:
            level = 'mini'
        elif 'pro' in mid:
            level = 'pro'
        else:
            level = 'standard'

        # Determine sublevel (within standard tier)
        sublevel = None
        if level == 'standard':
            if major == 4:
                if '4o' in mid:
                    sublevel = '4o'
                elif '4.1' in mid:
                    sublevel = '4.1'
                else:
                    sublevel = '4'
            elif major >= 5:
                minor_match = re.search(r'gpt-?\d+\.(\d+)', mid)
                if minor_match:
                    sublevel = f'{major}.{minor_match.group(1)}'
                else:
                    sublevel = str(major)

        return series, level, sublevel

    # Catch-all
    return 'Other', 'other', None


def _classify_google(model_id):
    """
    Classify a Google model ID into (series, level).

    Series examples:
      Gemini 2.x, Gemini 3.x, Gemini (latest), Gemma 3.x,
      Deep Research, Nano Banana, Gemini Robotics
    Level examples:
      flash-lite, flash, pro, nano, experimental, computer-use,
      1b, 4b, 12b, 27b, other
    """
    mid = model_id.lower()

    # --- Deep Research ---
    if 'deep-research' in mid:
        level = 'pro' if 'pro' in mid else 'standard'
        return 'Deep Research', level

    # --- Nano Banana (image generation) ---
    if 'nano-banana' in mid:
        level = 'pro' if 'pro' in mid else 'standard'
        return 'Nano Banana', level

    # --- Gemma (open-weight) models ---
    if mid.startswith('gemma'):
        # gemma-3n-e2b-it → Gemma 3.x / nano-2b
        # gemma-3-27b-it  → Gemma 3.x / 27b
        ver_match = re.search(r'gemma-?(\d+)', mid)
        major = int(ver_match.group(1)) if ver_match else 0
        series = f'Gemma {major}.x' if major else 'Gemma (other)'

        if 'gemma-3n' in mid or 'gemma3n' in mid:
            # Nano variant — extract parameter count
            size_match = re.search(r'e(\d+)b', mid)
            level = f'nano-{size_match.group(1)}b' if size_match else 'nano'
        else:
            size_match = re.search(r'(\d+)b', mid)
            level = f'{size_match.group(1)}b' if size_match else 'other'
        return series, level

    # --- Gemini Robotics ---
    if 'gemini-robotics' in mid:
        ver_match = re.search(r'(\d+(?:\.\d+)?)', mid)
        version = ver_match.group(1) if ver_match else ''
        return 'Gemini Robotics', version or 'other'

    # --- Gemini experimental (gemini-exp-MMDD) ---
    if re.match(r'^gemini-exp', mid):
        # gemini-exp-1206 is an experimental Gemini 2.0 model
        return 'Gemini 2.x', 'experimental'

    # --- Gemini latest aliases (gemini-flash-latest, gemini-pro-latest) ---
    if re.match(r'^gemini-(flash-lite|flash|pro|nano)-latest$', mid):
        if 'flash-lite' in mid:
            level = 'flash-lite'
        elif 'flash' in mid:
            level = 'flash'
        elif 'pro' in mid:
            level = 'pro'
        elif 'nano' in mid:
            level = 'nano'
        else:
            level = 'other'
        return 'Gemini (latest)', level

    # --- Standard Gemini versioned models ---
    if 'gemini' in mid:
        ver_match = re.search(r'gemini[- ]?(\d+)', mid)
        if ver_match:
            major = int(ver_match.group(1))
            series = f'Gemini {major}.x'
        else:
            series = 'Gemini (other)'

        # Determine level/tier
        if 'computer-use' in mid or 'computer_use' in mid:
            level = 'computer-use'
        elif 'flash-lite' in mid or 'flash_lite' in mid:
            level = 'flash-lite'
        elif 'flash' in mid:
            if 'image' in mid:
                level = 'flash-image'
            elif 'tts' in mid:
                level = 'flash-tts'
            else:
                level = 'flash'
        elif 'pro' in mid:
            if 'image' in mid:
                level = 'pro-image'
            elif 'tts' in mid:
                level = 'pro-tts'
            else:
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

    # Classify all models into (series, level, sublevel)
    classified = []  # list of (model_id, series, level, sublevel)
    for mid in model_ids:
        result = classify_fn(mid)
        # Support both 2-tuple and 3-tuple classifiers
        if len(result) == 3:
            series, level, sublevel = result
        else:
            series, level = result
            sublevel = None
        classified.append((mid, series, level, sublevel))

    # Step 3: series selection
    series_groups = defaultdict(list)
    for mid, series, level, sublevel in classified:
        series_groups[series].append((mid, level, sublevel))

    series_options = [
        (models[0][0] if len(models) == 1 else f"{s} ({len(models)} models)", s)
        for s, models in sorted(series_groups.items())
    ]
    selected_series = keypress_select(series_options, "Select series")
    if selected_series is None:
        return None

    models_in_series = series_groups[selected_series]

    # Step 4: level/tier selection (skip if all models share one level)
    level_groups = defaultdict(list)
    for mid, level, sublevel in models_in_series:
        level_groups[level].append((mid, sublevel))

    if len(level_groups) > 1:
        level_options = [
            (items[0][0] if len(items) == 1 else f"{lvl} ({len(items)} models)", lvl)
            for lvl, items in sorted(level_groups.items())
        ]
        selected_level = keypress_select(level_options, "Select level/tier")
        if selected_level is None:
            return None
        models_after_level = level_groups[selected_level]
    else:
        # Only one level — skip the prompt
        models_after_level = list(level_groups.values())[0]

    # Step 4b: sublevel selection (e.g. 4/4o/4.1 or 5/5.1/5.2)
    sublevel_groups = defaultdict(list)
    for mid, sublevel in models_after_level:
        sublevel_groups[sublevel].append(mid)

    # Only show sublevel step if there are multiple distinct sublevels
    distinct_sublevels = [s for s in sublevel_groups if s is not None]
    if len(distinct_sublevels) > 1:
        sub_options = [
            (mids[0] if len(mids) == 1 else f"{sl} ({len(mids)} models)", sl)
            for sl, mids in sorted(sublevel_groups.items()) if sl is not None
        ]
        selected_sub = keypress_select(sub_options, "Select sub-series")
        if selected_sub is None:
            return None
        remaining = sublevel_groups[selected_sub]
    else:
        # No meaningful sublevel split — flatten all
        remaining = [mid for mid, _ in models_after_level]

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
