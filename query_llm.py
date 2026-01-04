"""
Script 1: Query LLMs about deaths and export results to CSV.
Calls APIs (OpenAI, Claude, Gemini) and saves raw results.
"""

import csv
import os
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import argparse

from openai import OpenAI
from google import genai
import anthropic


# ============================================================================
# Global Configuration
# ============================================================================

# Debug mode (set via --debug flag)
DEBUG = False

# LLM Settings
TEMPERATURE = 0

# Max tokens per model type/thinking configuration
MAX_TOKENS_OPENAI = 5
MAX_TOKENS_CLAUDE = 2
MAX_TOKENS_GEMINI_MINIMAL = 5
MAX_TOKENS_LOW_REASONING = 200  # Used for all models when --reasoning is enabled, and gemini-3-pro (always low)
MAX_TOKENS_REASONING_RETRY = 800  # Max tokens to retry with if LOW_REASONING is not enough (OpenAI)

SYSTEM_PROMPT = 'Answer only "Yes" or "No". Nothing else.'
SYSTEM_PROMPT_REASONING = 'Answer only "Yes" or "No". Nothing else. Produce minimal reasoning.'
SYSTEM_PROMPT_CLAUDE = 'Answer only "Yes" or "No". Nothing else. If unsure, answer "Yes".'
SYSTEM_PROMPT_CLAUDE_REASONING = 'Answer only "Yes" or "No". Nothing else. Produce minimal reasoning. If unsure, answer "Yes".'

# Prompt template - use .format(name=..., description=...) to fill in
PROMPT_TEMPLATE = 'Is {name}, {description}, still alive?'

# Default models
DEFAULT_MODEL_OPENAI = "gpt-5.2"
DEFAULT_MODEL_GEMINI = "gemini-3.0-flash"
DEFAULT_MODEL_CLAUDE = "claude-opus-4-5-20250929"


def debug_print(*args, **kwargs):
    """Print only if DEBUG mode is enabled."""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


def debug_print_settings(provider: str, settings: dict):
    """Print all LLM settings in debug mode."""
    if not DEBUG:
        return
    print("\n" + "=" * 60)
    print(f"[DEBUG] {provider} SETTINGS")
    print("=" * 60)
    for key, value in settings.items():
        if key == 'system_prompt':
            print(f"  {key}: \"{value}\"")
        else:
            print(f"  {key}: {value}")
    print("=" * 60 + "\n")


def print_status_line(text: str, pad: int = 120):
    """Overwrite previous status line even when the new text is shorter."""
    print(f"\r{text.ljust(pad)}", end="", flush=True)


# ============================================================================
# Helper Functions
# ============================================================================

def parse_yes_no_response(answer: str) -> bool:
    """Parse a yes/no response to 'is X alive?'. Returns True (knows death) if No, False if Yes.
    Raises ValueError if answer contains both/neither 'yes' and 'no'.
    """
    normalized = answer.strip().lower()
    has_yes = "yes" in normalized
    has_no = "no" in normalized
    
    if has_yes and has_no:
        raise ValueError(f"Ambiguous response (contains both 'yes' and 'no'): '{answer}'")
    elif has_yes:
        return False  # Thinks alive = doesn't know about death
    elif has_no:
        return True   # Thinks dead = knows about death
    else:
        raise ValueError(f"Invalid response (no 'yes' or 'no' found): '{answer}'")


def parse_yes_no_response_safe(answer: str) -> Tuple[bool, str]:
    """Parse a yes/no response safely, returning (result, error_message).
    Returns (bool, None) on success, (None, error_message) on failure.
    """
    try:
        return parse_yes_no_response(answer), None
    except ValueError as e:
        return None, str(e)


def select_top_deaths_by_pageviews(
    deaths: List[Dict],
    top_per_day: int = None,
    top_per_month: int = None,
    min_views: int = None
) -> List[Dict]:
    """
    Select deaths based on pageview criteria.
    - top_per_day: Select the top N deaths per day by pageviews
    - top_per_month: Select the top N deaths per month by pageviews
    - min_views: Filter to only deaths with at least this many pageviews
    Deaths should have a 'pageviews' field from the fetch script.
    """
    if top_per_day and top_per_month:
        raise ValueError("top_per_day and top_per_month are mutually exclusive")

    # First filter by minimum views if specified
    if min_views:
        deaths = [d for d in deaths if int(d.get('pageviews', 0)) >= min_views]
    
    if top_per_month:
        deaths_by_month = defaultdict(list)
        for death in deaths:
            month_key = death.get('death_date', '')[:7]
            deaths_by_month[month_key].append(death)
        selected = []
        for month_key, month_deaths in sorted(deaths_by_month.items()):
            month_deaths.sort(key=lambda x: int(x.get('pageviews', 0)), reverse=True)
            selected.extend(month_deaths[:top_per_month])
        return selected

    if top_per_day:
        deaths_by_date = defaultdict(list)
        for death in deaths:
            deaths_by_date[death['death_date']].append(death)
        selected_deaths = []
        for date, date_deaths in sorted(deaths_by_date.items()):
            # Sort by pageviews descending
            date_deaths.sort(key=lambda x: int(x.get('pageviews', 0)), reverse=True)
            selected_deaths.extend(date_deaths[:top_per_day])
        return selected_deaths
    
    return deaths


def print_result(index: int, total: int, name: str, death_date: str, knows_death: bool, response: str = None):
    """Print a single test result."""
    prefix = f"[{index+1}/{total}] {name} (died {death_date})..."
    if response:
        prefix += f" Answer: {response}"
    
    if knows_death is True:
        print(f"{prefix} ✓ Knows")
    elif knows_death is False:
        print(f"{prefix} ✗ Doesn't know")
    else:
        print(f"{prefix} ! Error")


# ============================================================================
# Data Loading
# ============================================================================

def load_deaths_from_csv(csv_file: str) -> List[Dict]:
    """
    Load deaths data from CSV file.
    """
    deaths = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            deaths.append(row)
    return deaths


# ============================================================================
# OpenAI API (Sequential/Flex)
# ============================================================================

def check_death_knowledge(client: OpenAI, name: str, description: str, model: str = DEFAULT_MODEL_OPENAI, max_tokens: int = MAX_TOKENS_OPENAI, reasoning_effort: str = "none", temperature: float = TEMPERATURE) -> Tuple[bool, str]:
    """
    Ask the LLM if a person has died (OpenAI).
    Returns (knows_death: bool, raw_response: str)
    If reasoning is enabled and response is empty due to length limit, retries with higher max_tokens up to MAX_TOKENS_REASONING_RETRY.
    Raises exception on error (for OpenAI flex calls, we want to exit on error).
    """
    prompt = PROMPT_TEMPLATE.format(name=name, description=description)
    current_max_tokens = max_tokens
    max_retries_tokens = MAX_TOKENS_REASONING_RETRY
    system_prompt = SYSTEM_PROMPT_REASONING if reasoning_effort != "none" else SYSTEM_PROMPT
    
    debug_print(f"Prompt: \"{prompt}\"")

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=current_max_tokens,
                temperature=temperature,
                service_tier="flex",
                reasoning_effort=reasoning_effort
            )
            
            debug_print(f"Full response object: {response}")
            
            answer = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            debug_print(f"finish_reason: {finish_reason}")
            
            # Check if we got an empty response due to length limit with reasoning enabled
            if (not answer or answer.strip() == "") and finish_reason == "length" and reasoning_effort != "none":
                if current_max_tokens < max_retries_tokens:
                    # Increase tokens and retry
                    new_max_tokens = min(current_max_tokens * 2, max_retries_tokens)
                    print(f"Empty response (finish_reason=length), retrying with max_tokens={new_max_tokens}...", end=" ")
                    current_max_tokens = new_max_tokens
                    continue
                else:
                    print(f"Empty response even with max_tokens={current_max_tokens}")
            
            if answer:
                answer = answer.strip()
            else:
                answer = ""
            print(f"Answer: '{answer}'")
            return parse_yes_no_response(answer), answer
        
        except Exception as e:
            print(f"\n  Error querying for {name}: {e}")
            raise


def test_deaths_openai(
    deaths: List[Dict], 
    model: str = DEFAULT_MODEL_OPENAI,
    delay: float = 0.1,
    top_per_day: int = None,
    top_per_month: int = None,
    min_views: int = None,
    reasoning: bool = False
) -> Tuple[List[Dict], bool]:
    """
    Test the LLM's knowledge of deaths (OpenAI Flex - sequential).
    Returns (results, had_error) tuple.
    On error, returns partial results collected so far with had_error=True.
    """
    client = OpenAI()  # Uses OPENAI_API_KEY environment variable
    deaths = select_top_deaths_by_pageviews(deaths, top_per_day, top_per_month, min_views)
    results = []
    had_error = False
    
    max_tokens = MAX_TOKENS_LOW_REASONING if reasoning else MAX_TOKENS_OPENAI
    reasoning_effort = "low" if reasoning else "none"
    temperature = 1 if reasoning else TEMPERATURE
    system_prompt = SYSTEM_PROMPT_REASONING if reasoning else SYSTEM_PROMPT
    
    debug_print_settings("OpenAI", {
        'model': model,
        'max_tokens': max_tokens,
        'reasoning_effort': reasoning_effort,
        'temperature': temperature,
        'service_tier': 'flex',
        'system_prompt': system_prompt,
        'prompt_template': PROMPT_TEMPLATE
    })
    
    total = len(deaths)
    print(f"Testing {total} deaths with model: {model} (max_tokens={max_tokens}, reasoning_effort={reasoning_effort})")
    print("=" * 50)
    
    for i, death in enumerate(deaths):
        print(f"[{i+1}/{total}] {death['name']} (died {death['death_date']})...", end=" ")
        
        try:
            knows_death, response = check_death_knowledge(
                client, death['name'], death.get('description', ''), model, max_tokens, reasoning_effort, temperature
            )
            
            results.append({
                **death,
                'llm_knows_death': knows_death,
                'llm_response': response
            })
            
            if knows_death is True:
                print("✓ Knows")
            elif knows_death is False:
                print("✗ Doesn't know")
            else:
                print("! Error")
        except Exception as e:
            print(f"\nFatal error at item {i+1}: {e}")
            had_error = True
            break
        
        if i < total - 1:
            time.sleep(delay)
    
    return results, had_error


# ============================================================================
# Claude Batch API
# ============================================================================

def process_batch_results(
    deaths: List[Dict],
    result_map: Dict[str, Tuple[bool, str]],
    total: int
) -> List[Dict]:
    """
    Process batch results and build the final results list.
    Shared by Claude and Gemini batch processing.
    """
    results = []
    for i, death in enumerate(deaths):
        knows_death, response = result_map.get(str(i), (None, "No result"))
        
        # Debug: verify we're matching the right result to the right person
        debug_print(f"Matching result {i} to {death['name']} (died {death['death_date']}): {response}")
        
        result = {
            **death,
            'llm_knows_death': knows_death,
            'llm_response': response
        }
        results.append(result)
        print_result(i, total, death['name'], death['death_date'], knows_death, response)
    
    return results


def test_deaths_batch_claude(
    deaths: List[Dict], 
    model_name: str = DEFAULT_MODEL_CLAUDE,
    top_per_day: int = None,
    top_per_month: int = None,
    min_views: int = None,
    reasoning: bool = False
) -> List[Dict]:
    """
    Test Claude's knowledge of deaths using the Batch API.
    Submits all requests at once and waits for completion.
    Always returns results (never exits on error for batch).
    """
    client = anthropic.Anthropic()
    deaths = select_top_deaths_by_pageviews(deaths, top_per_day, top_per_month, min_views)
    
    # When reasoning is enabled, use extended thinking with minimum budget (1024)
    # max_tokens includes both thinking budget AND output tokens
    # budget_tokens must be less than max_tokens
    temperature = 1 if reasoning else TEMPERATURE
    
    # Thinking config: enabled with budget_tokens when reasoning, disabled otherwise
    if reasoning:
        thinking_budget = 1024  # Minimum allowed
        max_tokens = thinking_budget + MAX_TOKENS_CLAUDE  # thinking + output (Yes/No)
        thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}
    else:
        max_tokens = MAX_TOKENS_CLAUDE
        thinking_config = {"type": "disabled"}
    
    total = len(deaths)
    print(f"Testing {total} deaths with model: {model_name} (using Batch API, max_tokens={max_tokens}, thinking={'enabled (1024)' if reasoning else 'disabled'})")
    print("=" * 50)
    
    # Build batch requests
    system_prompt = SYSTEM_PROMPT_CLAUDE_REASONING if reasoning else SYSTEM_PROMPT_CLAUDE
    
    debug_print_settings("Claude", {
        'model': model_name,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'thinking': thinking_config,
        'system_prompt': system_prompt,
        'prompt_template': PROMPT_TEMPLATE
    })
    
    batch_requests = []
    for i, death in enumerate(deaths):
        prompt = PROMPT_TEMPLATE.format(name=death['name'], description=death.get('description', ''))
        params = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": thinking_config
        }
        batch_requests.append({
            "custom_id": str(i),
            "params": params
        })
        if i == 0:
            debug_print(f"Sample request (first): {batch_requests[0]}")
    
    print(f"Submitting batch of {len(batch_requests)} requests...")
    batch = client.messages.batches.create(requests=batch_requests)
    batch_id = batch.id
    print(f"Batch created: {batch_id}")
    
    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print_status_line(
            f"Status: {status} | Succeeded: {counts.succeeded}/{total} | Processing: {counts.processing} | Errored: {counts.errored}"
        )
        
        if status == "ended":
            print()
            break
        time.sleep(5)
    
    # Retrieve results - handle errors gracefully
    print("Retrieving results...")
    result_map = {}
    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        debug_print(f"Result {custom_id}: {entry}")
        if entry.result.type == "succeeded":
            try:
                answer = entry.result.message.content[0].text.strip()
                knows_death, error = parse_yes_no_response_safe(answer)
                if error:
                    result_map[custom_id] = (None, f"{answer} (parse error: {error})")
                else:
                    result_map[custom_id] = (knows_death, answer)
            except Exception as e:
                result_map[custom_id] = (None, f"Error parsing response: {e}")
        else:
            result_map[custom_id] = (None, f"Error: {entry.result.type}")
    
    return process_batch_results(deaths, result_map, total)


# ============================================================================
# Gemini Batch API
# ============================================================================

def test_deaths_batch_gemini(
    deaths: List[Dict], 
    model_name: str = DEFAULT_MODEL_GEMINI,
    top_per_day: int = None,
    top_per_month: int = None,
    min_views: int = None,
    reasoning: bool = False
) -> List[Dict]:
    """
    Test Gemini's knowledge of deaths using the Batch API.
    Submits all requests at once and waits for completion.
    Always returns results (never exits on error for batch).
    """
    client = genai.Client(api_key=os.environ.get('GOOGLE_API_KEY'))
    deaths = select_top_deaths_by_pageviews(deaths, top_per_day, top_per_month, min_views)
    
    total = len(deaths)
    print(f"Testing {total} deaths with model: {model_name} (using Batch API)")
    print("=" * 50)
    
    # Determine thinking level based on model and reasoning flag
    # gemini-3-pro-preview doesn't support "minimal", use "low" instead
    # When --reasoning is enabled, use "low" for all models
    if reasoning or "gemini-3-pro" in model_name.lower():
        thinking_level = "low"
    else:
        thinking_level = "minimal"
    
    # Select max tokens based on thinking level
    if thinking_level == "low":
        max_tokens = MAX_TOKENS_LOW_REASONING
    else:
        max_tokens = MAX_TOKENS_GEMINI_MINIMAL
    
    temperature = 1 if reasoning else TEMPERATURE
    
    # Build inline requests for Gemini Batch API
    system_prompt = SYSTEM_PROMPT_REASONING if reasoning else SYSTEM_PROMPT
    
    debug_print_settings("Gemini", {
        'model': model_name,
        'max_output_tokens': max_tokens,
        'temperature': temperature,
        'thinking_level': thinking_level,
        'system_prompt': system_prompt,
        'prompt_template': PROMPT_TEMPLATE
    })
    
    inline_requests = []
    for i, death in enumerate(deaths):
        prompt = PROMPT_TEMPLATE.format(name=death['name'], description=death.get('description', ''))
        inline_requests.append({
            'contents': [{
                'parts': [{'text': prompt}],
                'role': 'user'
            }],
            'config': {
                'system_instruction': {'parts': [{'text': system_prompt}]},
                'temperature': float(temperature),
                'max_output_tokens': max_tokens,
                'thinking_config': {'thinking_level': thinking_level}
            }
        })
        if i == 0:
            debug_print(f"Sample request (first): {inline_requests[0]}")
    
    print(f"Submitting batch of {len(inline_requests)} requests (thinking_level={thinking_level})...")
    batch_job = client.batches.create(
        model=model_name,
        src=inline_requests,
        config={'display_name': f'cutoff-test-{model_name}'}
    )
    job_name = batch_job.name
    print(f"Batch created: {job_name}")
    
    # Poll for completion
    completed_states = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        print_status_line(f"Status: {state}")
        
        if state in completed_states:
            print()
            break
        time.sleep(5)
    
    # Handle batch job failure - still try to get partial results
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Warning: Batch job ended with state: {batch_job.state.name}")
    
    # Retrieve results - handle errors gracefully
    # WARNING: For inline requests, there's no custom_id. We assume responses are in same order as requests.
    # This may not be guaranteed by the API - consider using file-based approach with keys for reliability.
    print("Retrieving results...")
    result_map = {}
    if batch_job.dest and batch_job.dest.inlined_responses:
        num_responses = len(batch_job.dest.inlined_responses)
        num_requests = len(inline_requests)
        if num_responses != num_requests:
            print(f"WARNING: Response count ({num_responses}) != Request count ({num_requests})")
        
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            # Debug: show what we got and what we expected
            if DEBUG and i < len(deaths):
                expected_name = deaths[i]['name']
                debug_print(f"Result {i} (expected: {expected_name}): {inline_response}")
            else:
                debug_print(f"Result {i}: {inline_response}")
            
            if inline_response.response:
                try:
                    answer = inline_response.response.text.strip()
                    knows_death, error = parse_yes_no_response_safe(answer)
                    if error:
                        result_map[str(i)] = (None, f"{answer} (parse error: {error})")
                    else:
                        result_map[str(i)] = (knows_death, answer)
                except AttributeError:
                    result_map[str(i)] = (None, "Error: Could not parse response")
            elif inline_response.error:
                result_map[str(i)] = (None, f"Error: {inline_response.error}")
            else:
                result_map[str(i)] = (None, "Error: No response")
    
    return process_batch_results(deaths, result_map, total)


# ============================================================================
# Save Results
# ============================================================================

def save_results(results: List[Dict], output_file: str):
    """
    Save the test results to a CSV file.
    """
    if not results:
        print("No results to save!")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'death_date', 'description', 'pageviews', 'llm_knows_death', 'llm_response']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Query LLMs about deaths and export results to CSV.'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='deaths_data.csv',
        help='Input CSV file with deaths data (default: deaths_data.csv)'
    )
    parser.add_argument(
        '--start', 
        type=str, 
        default=None,
        help='Start date filter in YYYY-MM-DD format (default: no filter)'
    )
    parser.add_argument(
        '--end', 
        type=str, 
        default=None,
        help='End date filter in YYYY-MM-DD format (default: no filter)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=DEFAULT_MODEL_OPENAI,
        help=f'Model to test (default: {DEFAULT_MODEL_OPENAI}). Use "claude-*" for Claude, "gemini-*" for Gemini.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output CSV file for results (default: auto-generated from model and dates)'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--top-per-day', 
        type=int, 
        default=5,
        help='Select top N deaths per day by pageviews (default: 5, use 0 for all)'
    )
    group.add_argument(
        '--top-per-month', 
        type=int, 
        default=None,
        help='Select top N deaths per month by pageviews (use 0 for all)'
    )
    parser.add_argument(
        '--min-views', 
        type=int, 
        default=None,
        help='Minimum pageviews required to include a death (default: no minimum)'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=0.05,
        help='Delay between API calls in seconds (default: 0.05 for Flex tier, OpenAI only)'
    )
    parser.add_argument(
        '--reasoning', 
        action='store_true',
        help='Enable reasoning mode (higher token limits, temperature=1)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode (print all settings, prompts, and full responses)'
    )
    
    args = parser.parse_args()
    
    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    
    # Check for API key based on model
    is_gemini = "gemini" in args.model.lower()
    is_claude = "claude" in args.model.lower()
    
    if is_gemini:
        if not os.environ.get('GOOGLE_API_KEY'):
            print("Error: GOOGLE_API_KEY environment variable not set")
            print("Set it with: $env:GOOGLE_API_KEY = 'your-key-here'")
            return
    elif is_claude:
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            print("Set it with: $env:ANTHROPIC_API_KEY = 'your-key-here'")
            return
    else:
        if not os.environ.get('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY environment variable not set")
            print("Set it with: $env:OPENAI_API_KEY = 'your-key-here'")
            return
    
    # Load deaths data
    print(f"Loading deaths from {args.input}")
    deaths = load_deaths_from_csv(args.input)
    print(f"Loaded {len(deaths)} death records")
    
    # Filter by date range if specified
    if args.start or args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d') if args.start else datetime.min
        end_date = datetime.strptime(args.end, '%Y-%m-%d') if args.end else datetime.max
        
        filtered_deaths = []
        for death in deaths:
            death_date = datetime.strptime(death['death_date'], '%Y-%m-%d')
            if start_date <= death_date <= end_date:
                filtered_deaths.append(death)
        
        print(f"Filtered to {len(filtered_deaths)} deaths in range {args.start or 'beginning'} to {args.end or 'end'}")
        deaths = filtered_deaths
    
    if not deaths:
        print("No deaths to test!")
        return
    
    # Generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        # Build filename from model and date range
        model_short = args.model.replace('/', '-').replace(':', '-')
        date_part = ""
        if args.start:
            date_part += f"_{args.start}"
        if args.end:
            date_part += f"_to_{args.end}" if args.start else f"_until_{args.end}"
        output_file = f"results_{model_short}{date_part}.csv"
    
    # Test the model
    top_per_day = args.top_per_day if args.top_per_day and args.top_per_day > 0 else None
    top_per_month = args.top_per_month if args.top_per_month and args.top_per_month > 0 else None
    min_views = args.min_views
    reasoning = args.reasoning
    
    results = None
    had_error = False
    
    try:
        if is_gemini:
            results = test_deaths_batch_gemini(
                deaths, 
                model_name=args.model,
                top_per_day=top_per_day,
                top_per_month=top_per_month,
                min_views=min_views,
                reasoning=reasoning
            )
        elif is_claude:
            results = test_deaths_batch_claude(
                deaths, 
                model_name=args.model,
                top_per_day=top_per_day,
                top_per_month=top_per_month,
                min_views=min_views,
                reasoning=reasoning
            )
        else:
            # OpenAI flex - returns partial results on error
            results, had_error = test_deaths_openai(
                deaths, 
                model=args.model,
                delay=args.delay,
                top_per_day=top_per_day,
                top_per_month=top_per_month,
                min_views=min_views,
                reasoning=reasoning
            )
    except Exception as e:
        print(f"\nFatal error: {e}")
    
    # Always save results if we have any
    if results:
        save_results(results, output_file)
        
        # Print quick summary
        total_correct = sum(1 for r in results if r['llm_knows_death'] is True)
        total_incorrect = sum(1 for r in results if r['llm_knows_death'] is False)
        total_errors = sum(1 for r in results if r['llm_knows_death'] is None)
        
        print("\n" + "=" * 50)
        print("QUICK SUMMARY")
        print("=" * 50)
        print(f"Total tested: {len(results)}")
        print(f"  Knows death: {total_correct} ({total_correct/len(results)*100:.1f}%)")
        print(f"  Doesn't know: {total_incorrect} ({total_incorrect/len(results)*100:.1f}%)")
        if total_errors > 0:
            print(f"  Errors: {total_errors} ({total_errors/len(results)*100:.1f}%)")
        print(f"\nRun 'python process_results.py --input {output_file}' for detailed analysis.")
    
    # Exit with error code if there was an error
    if had_error:
        print("\nExiting due to error (partial results saved).")
        exit(1)


if __name__ == "__main__":
    main()
