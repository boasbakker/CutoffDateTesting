"""Script 1: Fetch deaths from Wikipedia and save to CSV with pageview counts.
Uses Wikipedia API to get deaths from "Deaths in [Month] [Year]" pages.
Exports ALL deaths with their pageview counts (60 days after death).
Filtering/selection is done by the LLM testing script.
"""

import requests
import re
import csv
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global headers for Wikipedia API requests
WIKI_HEADERS = {
    "User-Agent": "CutoffDateTesting/1.0 (Educational research project)"
}


def get_pageviews_sum(article_title: str, death_date: datetime) -> int:
    """
    Get total pageviews for an article from 1 day before death up to 60 days after.
    Uses Wikimedia Pageviews REST API.
    """
    # Pageviews API expects underscores instead of spaces
    safe_title = article_title.replace(' ', '_')
    start = (death_date - timedelta(days=1)).strftime('%Y%m%d')
    # End is inclusive; use +58 to cover 60 days total (1 day before + 59 days after)
    end = (death_date + timedelta(days=58)).strftime('%Y%m%d')
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/user/{safe_title}/daily/{start}/{end}"
    )
    resp = requests.get(url, headers=WIKI_HEADERS, timeout=30)
    if resp.status_code == 404:
        return 0  # Page not found or no views
    resp.raise_for_status()
    data = resp.json()
    return sum(item.get('views', 0) for item in data.get('items', []))


def get_pageviews_for_articles(article_entries: List[Dict], max_workers: int = 20) -> Dict[str, int]:
    """
    Compute pageviews for multiple articles using parallel requests.
    article_entries: list of dicts with 'article_title' and 'death_date' (datetime).
    Returns mapping article_title -> pageviews_sum.
    """
    if not article_entries:
        return {}
    
    results: Dict[str, int] = {}
    total = len(article_entries)
    completed = 0
    
    def fetch_one(entry: Dict) -> tuple:
        title = entry['article_title']
        death_dt = entry['death_date']
        return title, get_pageviews_sum(title, death_dt)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, entry): entry for entry in article_entries}
        for future in as_completed(futures):
            title, views = future.result()
            results[title] = views
            completed += 1
            print(f"    Pageviews: {completed}/{total}", end="\r")
    
    print()  # finish line
    return results


def get_wikipedia_page_content(page_title: str) -> Optional[str]:
    """
    Fetch the wikitext content of a Wikipedia page using the API.
    Uses a single API call to get the full page content.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2"
    }
    
    response = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    pages = data.get("query", {}).get("pages", [])
    if pages and "revisions" in pages[0]:
        return pages[0]["revisions"][0]["slots"]["main"]["content"]
    return None


def parse_deaths_from_wikitext(wikitext: str, year: int, month: int) -> List[Dict]:
    """
    Parse the wikitext to extract deaths with their dates.
    Wikipedia "Deaths in [Month] [Year]" pages have a consistent format.
    Returns list of dicts with name, article_title, death_date, description.
    """
    deaths = []
    current_day = None
    
    lines = wikitext.split('\n')
    
    for line in lines:
        # Match day headers like "==1==" or "== 1 ==" or "===1==="
        day_match = re.match(r'^==+\s*(\d{1,2})\s*==+', line)
        if day_match:
            current_day = int(day_match.group(1))
            continue
        
        # Skip if we haven't found a day yet
        if current_day is None:
            continue
        
        # Match death entries - they typically start with * and contain [[Name]]
        # Format is usually: * [[Name]], age, description
        if line.startswith('*') and '[[' in line:
            # Extract the first linked name (usually the person who died)
            name_match = re.search(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', line)
            if name_match:
                # Article title is always the link target (group 1)
                article_title = name_match.group(1).strip()
                
                # Display name is either the piped text (group 2) or the link target
                name = name_match.group(2) if name_match.group(2) else article_title
                name = name.strip()
                
                # Skip if it looks like a category or file link
                if ':' in article_title:
                    continue
                
                # Extract description (everything after the name link)
                # First, get everything after the first ]] 
                after_name = line.split(']]', 1)[1] if ']]' in line else ''
                
                # Remove wiki markup: [[link|text]] -> text, [[link]] -> link
                description = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', after_name)
                # Remove HTML tags and refs
                description = re.sub(r'<[^>]+>', '', description)
                description = re.sub(r'\{\{[^}]+\}\}', '', description)  # Remove templates
                # Clean up punctuation and whitespace
                description = description.strip(' ,;')
                # Remove leading age if present (e.g., "73, American politician" or "94–95, British actor")
                # Handle both single ages and age ranges with various dash types
                description = re.sub(r'^\d{1,3}(?:[–—-]\d{1,3})?\s*,\s*', '', description)
                description = description.strip()
                # Remove date ranges like (1994-2001) or (1994–2001)
                description = re.sub(r'\(\d{4}[–—-]\d{4}\)', '', description)
                description = description.strip()
                # Cut off at first comma or period that's NOT inside parentheses
                # First, find commas/periods outside parentheses
                paren_depth = 0
                cutoff_pos = None
                for i, char in enumerate(description):
                    if char == '(':
                        paren_depth += 1
                    elif char == ')':
                        paren_depth = max(0, paren_depth - 1)
                    elif char in ',.' and paren_depth == 0:
                        cutoff_pos = i
                        break
                if cutoff_pos is not None:
                    description = description[:cutoff_pos].strip()
                
                try:
                    death_date = datetime(year, month, current_day)
                    deaths.append({
                        'name': name,
                        'article_title': article_title,
                        'death_date': death_date.strftime('%Y-%m-%d'),
                        'description': description if description else ""
                    })
                except ValueError:
                    # Invalid date (e.g., Feb 30)
                    pass
    
    return deaths


def fetch_deaths_for_month(year: int, month: int) -> List[Dict]:
    """
    Fetch all deaths for a given month and year.
    """
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    month_name = month_names[month - 1]
    page_title = f"Deaths in {month_name} {year}"
    
    print(f"Fetching: {page_title}")
    
    wikitext = get_wikipedia_page_content(page_title)
    if wikitext is None:
        print(f"  Could not fetch page: {page_title}")
        return []
    
    deaths = parse_deaths_from_wikitext(wikitext, year, month)
    print(f"  Found {len(deaths)} deaths")
    
    return deaths


def fetch_deaths_for_date_range(start_date: datetime, end_date: datetime) -> List[Dict]:
    """
    Fetch deaths for a range of dates by fetching monthly pages.
    Exports ALL deaths with their pageview counts (60 days after death).
    
    Args:
        start_date: Start of date range
        end_date: End of date range  
    """
    all_deaths = []
    
    # Get unique year-month combinations in the range
    months_to_fetch = set()
    current = start_date.replace(day=1)
    while current <= end_date:
        months_to_fetch.add((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    # Sort chronologically
    months_to_fetch = sorted(months_to_fetch)
    
    for i, (year, month) in enumerate(months_to_fetch):
        deaths = fetch_deaths_for_month(year, month)
        
        # Group deaths by day
        deaths_by_day = {}
        for death in deaths:
            death_date = datetime.strptime(death['death_date'], '%Y-%m-%d')
            if start_date <= death_date <= end_date:
                day_key = death['death_date']
                if day_key not in deaths_by_day:
                    deaths_by_day[day_key] = []
                deaths_by_day[day_key].append(death)

        # Get pageviews for all deaths in this month
        if deaths_by_day:
            # Collect all article titles + death dates for this month
            all_article_entries = []
            for day_deaths in deaths_by_day.values():
                for death in day_deaths:
                    all_article_entries.append({
                        'article_title': death['article_title'],
                        'death_date': datetime.strptime(death['death_date'], '%Y-%m-%d')
                    })

            # Get pageview counts for all articles in this month (parallel requests)
            print(f"  Fetching pageview counts for {len(all_article_entries)} articles...")
            pageview_counts = get_pageviews_for_articles(all_article_entries)

            # Add pageview count to each death record and collect all
            for day_key in sorted(deaths_by_day.keys()):
                for death in deaths_by_day[day_key]:
                    death['pageviews'] = pageview_counts.get(death['article_title'], 0)
                    death.pop('article_title', None)  # Remove helper field
                    all_deaths.append(death)
        
        # Polite delay between requests (Wikipedia asks for no more than 1 req/sec)
        if i < len(months_to_fetch) - 1:
            time.sleep(1.5)
    
    return all_deaths


def save_to_csv(deaths: List[Dict], output_file: str):
    """
    Save the deaths data to a CSV file.
    """
    if not deaths:
        print("No deaths to save!")
        return
    
    # Sort by date, then by pageviews (descending) within each day
    deaths.sort(key=lambda x: (x['death_date'], -x.get('pageviews', 0)))
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'death_date', 'description', 'pageviews']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(deaths)
    
    print(f"\nSaved {len(deaths)} deaths to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch notable deaths from Wikipedia for a date range.'
    )
    parser.add_argument(
        '--start', 
        type=str, 
        default='2024-01-01',
        help='Start date in YYYY-MM-DD format (default: 2024-01-01)'
    )
    parser.add_argument(
        '--end', 
        type=str, 
        default='2025-12-31',
        help='End date in YYYY-MM-DD format (default: 2025-12-31)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='deaths_data.csv',
        help='Output CSV file (default: deaths_data.csv)'
    )
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        return
    
    if start_date > end_date:
        print("Error: Start date must be before end date")
        return
    
    print(f"Fetching ALL deaths from {args.start} to {args.end}")
    print("Pageviews will be fetched for each person (60 days after death)")
    print("=" * 50)
    
    deaths = fetch_deaths_for_date_range(start_date, end_date)
    save_to_csv(deaths, args.output)
    
    # Print summary by month
    print("\nSummary by month:")
    from collections import Counter
    month_counts = Counter(d['death_date'][:7] for d in deaths)
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count} deaths")


if __name__ == "__main__":
    main()
