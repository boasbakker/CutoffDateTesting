"""
Script 1: Query LLMs about deaths and export results to CSV.
Calls APIs (OpenAI, Claude, Gemini) and saves raw results.
"""

import csv
import argparse
import time
from typing import List, Dict

import config
from llm_providers import OpenAIProvider, ClaudeBatchProvider, GeminiBatchProvider

# ============================================================================
# Helper Functions
# ============================================================================

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
    from collections import defaultdict
    
    if top_per_day is not None and top_per_month is not None:
        raise ValueError("top_per_day and top_per_month are mutually exclusive")

    # First filter by minimum views if specified
    if min_views:
        deaths = [d for d in deaths if int(d.get('pageviews', 0)) >= min_views]
    
    if top_per_month is not None:
        deaths_by_month = defaultdict(list)
        for death in deaths:
            month_key = death.get('death_date', '')[:7]
            deaths_by_month[month_key].append(death)
        selected = []
        for month_key, month_deaths in sorted(deaths_by_month.items()):
            month_deaths.sort(key=lambda x: int(x.get('pageviews', 0)), reverse=True)
            selected.extend(month_deaths[:top_per_month])
        return selected

    if top_per_day is not None:
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
        fieldnames = [
            'name', 'article_title', 'death_date', 'birth_date', 
            'description', 'pageviews', 'llm_knows_death', 'llm_response'
        ]
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
        default='gpt-5.2',
        help='Model to use (default: gpt-5.2)'
    )
    parser.add_argument(
        '--top-per-day', 
        type=int, 
        default=None,
        help='Select top N deaths per day by pageviews'
    )
    parser.add_argument(
        '--top-per-month', 
        type=int, 
        default=None,
        help='Select top N deaths per month by pageviews'
    )
    parser.add_argument(
        '--min-views', 
        type=int, 
        default=None,
        help='Minimum pageviews to include'
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=0.1,
        help='Delay between requests in seconds (OpenAI only, default: 0.1)'
    )
    parser.add_argument(
        '--reasoning',
        action='store_true',
        help='Enable reasoning/thinking mode for supported models'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Set global debug flag in config is not really possible directly as it's a constant module, 
    # but we pass it to the provider
    
    # Load data
    print(f"Loading deaths from {args.input}...")
    deaths = load_deaths_from_csv(args.input)
    print(f"Loaded {len(deaths)} deaths.")
    
    # Filter by date if specified
    if args.start:
        deaths = [d for d in deaths if d['death_date'] >= args.start]
    if args.end:
        deaths = [d for d in deaths if d['death_date'] <= args.end]
        
    print(f"After date filtering: {len(deaths)} deaths.")
    
    if len(deaths) == 0:
        print("No deaths found in the specified range.")
        return

    # Select provider based on model name
    model = args.model.lower()
    
    if 'gpt' in model or 'o1' in model or 'o3' in model:
        provider = OpenAIProvider(model=args.model, debug=args.debug)
    elif 'claude' in model:
        provider = ClaudeBatchProvider(model=args.model, debug=args.debug)
    elif 'gemini' in model:
        provider = GeminiBatchProvider(model=args.model, debug=args.debug)
    else:
        print(f"Unknown model provider for {args.model}. Defaulting to OpenAIProvider.")
        provider = OpenAIProvider(model=args.model, debug=args.debug)
        
    # Run tests
    # Helper to selection is done inside test_deaths for Providers? 
    # No, I should do selection here before passing to provider to keep provider simple
    
    selected_deaths = select_top_deaths_by_pageviews(
        deaths, 
        top_per_day=args.top_per_day, 
        top_per_month=args.top_per_month, 
        min_views=args.min_views
    )
    
    print(f"Selected {len(selected_deaths)} deaths for testing.")
    
    if len(selected_deaths) == 0:
        print("No deaths selected.")
        return

    try:
        results = provider.test_deaths(
            selected_deaths, 
            delay=args.delay, 
            reasoning=args.reasoning
        )
        
        # Handle tuple return for OpenAI (results, had_error)
        if isinstance(results, tuple):
            results, had_error = results
            if had_error:
                print("Note: Some errors occurred during processing.")
    
        # Generate output filename
        date_range = "all"
        if args.start:
            date_range = args.start
            if args.end:
                date_range += f"_to_{args.end}"
        
        selection_str = ""
        if args.top_per_day:
            selection_str = f"_top{args.top_per_day}day"
        elif args.top_per_month:
            selection_str = f"_top{args.top_per_month}month"
            
        if args.reasoning:
            selection_str += "_reasoning"
            
        output_file = f"results/{args.model.replace('/', '-')}_{date_range}{selection_str}.csv"
        
        # Ensure results directory exists
        import os
        os.makedirs('results', exist_ok=True)
        
        save_results(results, output_file)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
