# LLM Cutoff Date Testing — Developer Documentation

## Project Overview

This project tests LLM knowledge cutoff dates by querying whether deceased people are still alive. Deaths are sourced from Wikipedia's "Deaths in [Month] [Year]" pages along with pageview counts. Multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini) are queried via their APIs, and the results are analyzed to determine the knowledge cutoff point of each model.

## Core User Requirements

- Only record birth year for people with a clear birth year, so month and day must also be known
- Fetch death records from Wikipedia with pageview data
- Query multiple LLM providers (OpenAI, Claude, Gemini) about whether the person is still alive
- Use structured outputs (JSON with boolean `answer` field) to force deterministic Yes/No answers (native or fallback)
- Support batch APIs for Claude and Gemini, sequential/flex for OpenAI
- **Support 2026 Models & Reasoning:**
  - **OpenAI:** Support `gpt-5`, `o1`, `o3`. Control `reasoning_effort` (`minimal`, `low`, etc.).
  - **Claude:** Support `claude-3.7`, `claude-4.6`. Control extended `thinking` (`budget_tokens`, `adaptive`).
  - **Gemini:** Support `gemini-3`. Control `thinking_config` (`thinking_level`, `thinking_budget`).
- Filter deaths by pageviews (top per day/month, minimum views)
- Export results to CSV for analysis
- Generate accuracy-over-time plots and statistics

## Data Pipeline

```
Wikipedia "Deaths in X" pages
        │
        ▼
fetch_deaths_wikipedia.py  →  deaths_data.csv
        │
        ▼
query_llm.py  →  results/<model>_<dates>_<selection>.csv
        │
        ▼
process_results.py / process_results_monthly.py  →  plots + statistics
```

## File Structure

### Core Scripts (tracked in git)

| File | Lines | Purpose |
|------|-------|---------|
| `fetch_deaths_wikipedia.py` | ~800 | Fetches deaths from Wikipedia API, parses wikitext, gets pageviews (with redirect resolution and page-creation-date fallback for 404s) using OAuth 2 authentication, exports to CSV |
| `query_llm.py` | ~150 | Entry point to query LLMs. Selects provider and executes tests. |
| `config.py` | ~60 | Global configuration constants (API keys, prompts, tokens, reasoning levels). |
| `llm_providers.py` | ~350 | LLM Provider implementations (OpenAI, Claude, Gemini) with 2026 model support. |
| `process_results.py` | ~454 | Generates accuracy statistics by date/month/pageviews, produces plots |
| `process_results_monthly.py` | ~270 | Monthly accuracy analysis with trend lines, cleaner axis labels |
| `select_model.py` | ~270 | Interactive model selector. |

### Data Files

| File | Purpose |
|------|---------|
| `deaths_data.csv` | All deaths with columns: `name`, `article_title`, `death_date`, `birth_date`, `description`, `pageviews` |
| `results/` | Output directory for results CSV files and plot images |

### Debug/Utility Scripts (gitignored)

| File | Purpose |
|------|---------|
| `analyze_batch.py` | Checks for misalignment bugs in batch results |
| `check_around.py` | Inspects results around a specific index |
| `check_liam.py` | Debug script for Liam Payne result mismatch |
| `check_new.py` | Quick check of new result files |
| `test_batch_sizes.py` | Tests if batch size affects Gemini responses |
| `verify_batch_order.py` | Verifies Gemini Batch API returns results in order |
| `reproduce_batch_order_bug.py` | Repro script for Gemini batch ordering bug |
| `top_pageviews.py` | Shows top 10 people by pageviews in a result file |

## Configuration

### Environment Variables

| Variable | Provider | Required for |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | `gpt-*`, `o*` models |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-*` models |
| `GOOGLE_API_KEY` | Google | `gemini-*` models |

### Key Constants in `config.py`

| Constant | Value | Description |
|----------|-------|-------------|
| `TEMPERATURE` | 0 | Deterministic output (1 when reasoning is on) |
| `MAX_TOKENS_OPENAI` | 5 | Max output tokens for OpenAI |
| `MAX_TOKENS_CLAUDE` | 2 | Max output tokens for Claude |
| `MAX_TOKENS_GEMINI_MINIMAL` | 5 | Max output tokens for Gemini (minimal thinking) |
| `REASONING_EFFORT_MINIMAL` | "minimal" | For OpenAI GPT-5 series |
| `REASONING_EFFORT_LOW` | "low" | For OpenAI o-series |
| `GEMINI_THINKING_LEVEL_MINIMAL` | "MINIMAL" | For Gemini 3 series |

### CLI Usage (`query_llm.py`)

```bash
python query_llm.py --model <model> --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    --top-per-month <N> [--min-views <N>] [--reasoning] [--debug]
```

**Required** (mutually exclusive): `--top-per-day <N>` or `--top-per-month <N>` (use 0 for all).

## Provider-Specific Details (2026 Updates)

### OpenAI (Sequential/Flex)
- **Models:** `gpt-5`, `gpt-5.2`, `o1`, `o3`.
- **Reasoning Controls:**
  - **Default:** `reasoning_effort="minimal"` (GPT-5) or `"low"` (o-series).
  - **With `--reasoning`:** `reasoning_effort="medium"`.

### Claude (Batch API)
- **Models:** `claude-3.7`, `claude-4.6`.
- **Reasoning Controls:**
  - **Default:** Extended thinking disabled (parameter omitted).
  - **With `--reasoning`:**
    - Claude 4.6: `thinking={"type": "adaptive", "budget_tokens": 4096}`.
    - Claude 3.7: `thinking={"type": "enabled", "budget_tokens": 1024}`.

### Gemini (Batch API)
- **Models:** `gemini-3`, `gemini-2.5`.
- **Reasoning Controls:**
  - **Default:**
    - Gemini 3: `thinking_config={"thinking_level": "MINIMAL"}`.
    - Gemini 2.5: `thinking_config={"thinking_budget": 0}`.
  - **With `--reasoning`:**
    - Gemini 3: `thinking_config={"thinking_level": "HIGH", "include_thoughts": True}`.
    - Gemini 2.5: `thinking_config={"thinking_budget": 1024, "include_thoughts": True}`.

## Performance Optimizations
- **High Concurrency:** The script uses a `ThreadPoolExecutor` with 30 workers for pageview fetching.
- **Batching:** Wikidata requests (QIDs and Claims) are processed in parallel batches of 50.
- **Minimal Latency:** Artificial delays have been removed.

## Results CSV Format

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | Person's name |
| `article_title` | string | Wikipedia article title |
| `death_date` | string | `YYYY-MM-DD` |
| `birth_date` | string | `YYYY-MM-DD` |
| `description` | string | Brief description from Wikipedia |
| `pageviews` | int | Wikipedia pageviews (60 days after death) |
| `llm_knows_death` | bool | `True` = model knows the person died |
| `llm_response` | string | Raw LLM response text |
