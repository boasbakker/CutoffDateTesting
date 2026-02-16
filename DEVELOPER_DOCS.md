# LLM Cutoff Date Testing — Developer Documentation

## Project Overview

This project tests LLM knowledge cutoff dates by querying whether deceased people are still alive. Deaths are sourced from Wikipedia's "Deaths in [Month] [Year]" pages along with pageview counts. Multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini) are queried via their APIs, and the results are analyzed to determine the knowledge cutoff point of each model.

## Core User Requirements

- Only record birth year for people with a clear birth year, so month and day must also be known
- Fetch death records from Wikipedia with pageview data
- Query multiple LLM providers (OpenAI, Claude, Gemini) about whether the person is still alive
- Use structured outputs (JSON with boolean `answer` field) to force deterministic Yes/No answers (native or fallback)
- Support batch APIs for Claude and Gemini, sequential/flex for OpenAI
- Support reasoning/thinking modes for all providers (simplified prompt strategy)
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
| `config.py` | ~50 | Global configuration constants (API keys, prompts, tokens). |
| `llm_providers.py` | ~300 | LLM Provider implementations (OpenAI, Claude, Gemini). |
| `process_results.py` | ~454 | Generates accuracy statistics by date/month/pageviews, produces plots |
| `process_results_monthly.py` | ~270 | Monthly accuracy analysis with trend lines, cleaner axis labels |
| `select_model.py` | ~270 | Interactive model selector: provider → series → level → version via single-keypress input. Importable (`from select_model import select_model`) or standalone. |

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
| `OPENAI_API_KEY` | OpenAI | `gpt-*` models |
| `ANTHROPIC_API_KEY` | Anthropic | `claude-*` models |
| `GOOGLE_API_KEY` | Google | `gemini-*` models |

### Key Constants in `config.py`

| Constant | Value | Description |
|----------|-------|-------------|
| `TEMPERATURE` | 0 | Deterministic output (1 when reasoning is on) |
| `MAX_TOKENS_OPENAI` | 5 | Max output tokens for OpenAI |
| `MAX_TOKENS_CLAUDE` | 2 | Max output tokens for Claude |
| `MAX_TOKENS_GEMINI_MINIMAL` | 5 | Max output tokens for Gemini (minimal thinking) |
| `MAX_TOKENS_LOW_REASONING` | 200 | Max tokens when reasoning is enabled |
| `SYSTEM_PROMPT` | `"Answer 'Yes' or 'No'..."` | Standard system prompt (reasoning variant exists) |

### CLI Usage (`query_llm.py`)

```bash
python query_llm.py --model <model> --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    --top-per-month <N> [--min-views <N>] [--reasoning] [--debug]
```

**Required** (mutually exclusive): `--top-per-day <N>` or `--top-per-month <N>` (use 0 for all).

### CLI Usage (`process_results_monthly.py`)

```bash
python process_results_monthly.py --input results/<file>.csv [--min-samples <N>]
```

## Provider-Specific Details

### OpenAI (Sequential/Flex)
- Uses `client.chat.completions.create()` with `service_tier="flex"`
- Sequential calls with configurable delay between requests
- Structured output via `response_format` with `json_schema` and `strict: true`
- Reasoning via `reasoning_effort` parameter; retries with higher `max_tokens` on empty responses

### Claude (Batch API)
- Uses `client.messages.batches.create()` for batch processing
- Structured output via `output_config.format` with `json_schema` (models ≥ 4.5)
- **Legacy Fallback**: For older 3.x models, uses response prefilling with `{"answer":` + stop sequence `}`
- Extended thinking via `thinking` config with `budget_tokens`
- Uses simplified `SYSTEM_PROMPT` or `SYSTEM_PROMPT_REASONING` based on mode

### Gemini (Batch API with File-Based Input)
- Uses JSONL file upload → batch job → file download for results
- **Always uses structured outputs** via `responseMimeType: "application/json"` + `responseJsonSchema`
- **MIME Type Workaround**: Uses `mime_type: "text/plain"` for file uploads to avoid a known `400 INVALID_ARGUMENT` bug with `application/jsonl`.
- No longer uses probing; assumes model supports requested features
- Results matched by `key` field in JSONL for reliable ordering

## Performance Optimizations
- **High Concurrency**: The script uses a `ThreadPoolExecutor` with 30 workers for pageview fetching and parallelized chunking for Wikidata requests.
- **Batching**: Wikidata requests (QIDs and Claims) are processed in parallel batches of 50.
- **Minimal Latency**: Artificial delays have been removed, relying on OAuth 2's robust rate limits and standard retry logic.

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
