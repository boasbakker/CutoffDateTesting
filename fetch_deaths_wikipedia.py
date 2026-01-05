"""Script 1: Fetch deaths from Wikipedia and save to CSV with pageview counts.
Uses Wikipedia API to get deaths from "Deaths in [Month] [Year]" pages.
Exports ALL deaths with their pageview counts (60 days after death).
Filtering/selection is done by the LLM testing script.
"""

import requests
import re
import csv
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Script version - increment this when making changes to force new output file
SCRIPT_VERSION = "1.0"

# Global headers for Wikipedia API requests
WIKI_HEADERS = {
    "User-Agent": "CutoffDateTesting/1.0 (Educational research project)"
}


def get_pageviews_sum(article_title: str, death_date: datetime) -> int:
    """
    Get total pageviews for an article from 1 day before death up to 60 days after.
    Uses Wikimedia Pageviews REST API.
    Returns -1 if the page doesn't exist (404).
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
        return -1  # Page not found - article doesn't exist
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


def parse_death_entry(line: str, year: int, month: int, current_day: int, line_num: int, parent_item: Optional[str] = None) -> Optional[Dict]:
    """
    Parse a single death entry line and return a dict or None if invalid.
    Expected format: * [[Name]], age, description  OR  * [[Name]], description (if age unknown)
    
    Requirements:
    - Name must be a wiki link [[Name]] or [[Name|Display Name]]
    - Age is optional (may be unknown)
    - Description must be present and at least 2 words
    
    Args:
        parent_item: If this is a subitem, the text of the parent bullet point
    """
    # Helper to format error/warning prefix with optional parent context
    def msg_prefix():
        if parent_item:
            return f"  {{}} (line {line_num}, under: {parent_item[:60]}...): "
        return f"  {{}} (line {line_num}): "
    
    def extract_nationality_from_parent(parent: str) -> Optional[str]:
        """
        Try to extract nationality/description from parent item.
        Patterns like:
        - Notable Brazilians who died in the...
        - Chinese marathon runners killed in the...
        - Notable Americans killed in the...
        - Chinese marathon runner killed in the... (singular - use full description)
        - Israeli people killed in the 7 October attacks...
        """
        if not parent:
            return None
        
        # Pattern for "Nationality people killed in" (e.g., "Israeli people killed in the 7 October attacks")
        people_match = re.search(
            r'([A-Z][a-z]+)\s+people\s+(?:killed|who\s+died)\s+in',
            parent,
            re.IGNORECASE
        )
        if people_match:
            return people_match.group(1).strip()
        
        # First, try to match singular form (full description like "Chinese marathon runner")
        # Pattern: (Nationality + role in singular) (killed|who died) in the
        singular_match = re.search(
            r'(?:Notable\s+)?([A-Z][a-z]+(?:[-\s][a-z]+)+)\s+(?:killed|who\s+died)\s+in\s+the',
            parent,
            re.IGNORECASE
        )
        if singular_match:
            description = singular_match.group(1).strip()
            # Check it's not a plural nationality (ends with 's' followed by nothing or space before killed/died)
            # If it doesn't end with 's', it's likely a full description like "Chinese marathon runner"
            if not description.endswith('s'):
                return description
        
        # Pattern: (Notable)? (Nationality ending in 's' OR Nationality + role) (killed|who died) in the
        # Examples: "Notable Brazilians who died", "Chinese marathon runners killed", "Notable Americans killed"
        match = re.search(
            r'(?:Notable\s+)?([A-Z][a-z]+(?:[-\s][A-Z]?[a-z]+)*s)(?:\s+[a-z]+(?:\s+[a-z]+)*)?\s+(?:killed|who\s+died)\s+in\s+the',
            parent,
            re.IGNORECASE
        )
        if match:
            # Extract the nationality (e.g., "Brazilians" -> "Brazilian")
            nationality_plural = match.group(1)
            # Convert plural nationality to singular adjective
            if nationality_plural.lower().endswith('s'):
                nationality = nationality_plural[:-1]  # Remove trailing 's'
            else:
                nationality = nationality_plural
            return nationality
        return None
    
    def handle_error(error_msg: str, parsed_name: Optional[str] = None, parsed_description: Optional[str] = None) -> Optional[Dict]:
        """Handle an error by prompting user for input."""
        print(msg_prefix().format('ERROR') + error_msg)
        print(f"  Full line: {line}")
        if parent_item:
            print(f"  Parent: {parent_item}")
        
        skip = input("  Skip this person? (y/n, Enter=n): ").strip().lower()
        if skip == 'y':
            return None
        
        # Get name from user (suggest parsed name if available)
        if parsed_name:
            user_name = input(f"  Enter name (press Enter to keep '{parsed_name}'): ").strip()
            if not user_name:
                user_name = parsed_name
        else:
            user_name = input("  Enter name: ").strip()
        
        if not user_name:
            print("  No name provided, skipping.")
            return None
        
        # Try to build description from parent item for subitems with one-word descriptions
        suggested_description = parsed_description
        if parsed_description and len(parsed_description.split()) == 1 and parent_item:
            extracted = extract_nationality_from_parent(parent_item)
            if extracted:
                # Check if extracted is a full description (multiple words) or just nationality
                if len(extracted.split()) > 1:
                    # Full description like "Chinese marathon runner"
                    suggested_description = extracted
                else:
                    # Just nationality, combine with parsed description
                    suggested_description = f"{extracted} {parsed_description}"
                confirm = input(f"  Detected description: '{suggested_description}'. Correct? (y/n, Enter=y): ").strip().lower()
                if confirm == '' or confirm == 'y':
                    user_description = suggested_description
                else:
                    user_description = input("  Enter description: ").strip()
            else:
                # No nationality detected, ask for description
                if parsed_description:
                    user_description = input(f"  Enter description (press Enter to keep '{parsed_description}'): ").strip()
                    if not user_description:
                        user_description = parsed_description
                else:
                    user_description = input("  Enter description: ").strip()
        elif parsed_description:
            user_description = input(f"  Enter description (press Enter to keep '{parsed_description}'): ").strip()
            if not user_description:
                user_description = parsed_description
        else:
            user_description = input("  Enter description: ").strip()
        
        if not user_description:
            print("  No description provided, skipping.")
            return None
        
        try:
            death_date = datetime(year, month, current_day)
            return {
                'name': user_name,
                'article_title': user_name,  # Use name as article title for manual entries
                'death_date': death_date.strftime('%Y-%m-%d'),
                'description': user_description
            }
        except ValueError:
            print(f"  Invalid date {year}-{month}-{current_day}, skipping.")
            return None

    # Strip the leading * or ** 
    entry_text = re.sub(r'^\*+\s*', '', line)
    
    # Remove HTML comments like <!--D-->
    entry_text = re.sub(r'<!--[^>]*-->', '', entry_text)
    
    # Remove external links in format [https://...] or [http://...]
    entry_text = re.sub(r'\[https?://[^\]]*\]', '', entry_text)
    
    # Check for {{ill|Name|lang}} template (interlanguage link - no English article)
    ill_match = re.match(r'\{\{ill\|([^|\}]+)\|([^|\}]+)', entry_text)
    if ill_match:
        name = ill_match.group(1).strip()
        lang = ill_match.group(2).strip()
        print(msg_prefix().format('WARNING') + f"Skipping '{name}' - no English article (only {lang} Wikipedia)")
        return None
    
    # Remove {{circa}} and similar templates that appear before/with ages
    entry_text = re.sub(r'\{\{circa\}\}\s*', 'c. ', entry_text, flags=re.IGNORECASE)
    entry_text = re.sub(r'\{\{c\.\}\}\s*', 'c. ', entry_text, flags=re.IGNORECASE)
    
    # The entry MUST start with a wiki link (the person's name)
    if not entry_text.startswith('[['):
        return handle_error(f"Entry does not start with a wiki link: {line[:80]}...")
    
    # Extract the first linked name (the person who died)
    name_match = re.match(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', entry_text)
    if not name_match:
        return handle_error(f"Could not parse name link: {line[:80]}...")
    
    # Article title is always the link target (group 1)
    article_title = name_match.group(1).strip()
    
    # Display name is either the piped text (group 2) or the link target
    name = name_match.group(2) if name_match.group(2) else article_title
    name = name.strip()
    
    # Get everything after the name link
    after_name = entry_text[name_match.end():]
    
    # Parse description first so it's available for error handling
    description = None
    if after_name.strip() and after_name.strip() != ',':
        # Try to match age - format: ", 73," or ", 73–74," (age range) or ", 60s," (decade)
        # Age is optional, so we handle both cases
        age_match = re.match(r'\s*,\s*(\d{1,3}(?:[–—-]\d{1,3})?s?)\s*,', after_name)
        if age_match:
            # Age found - extract description after the age
            description_start = after_name[age_match.end():]
        else:
            # No age - description starts right after the comma following the name
            description_start = re.sub(r'^\s*,\s*', '', after_name)
        
        # Remove wiki markup: [[link|text]] -> text, [[link]] -> link
        description = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', description_start)
        # Remove HTML tags and refs
        description = re.sub(r'<[^>]+>', '', description)
        description = re.sub(r'\{\{[^}]+\}\}', '', description)  # Remove templates
        # Clean up punctuation and whitespace
        description = description.strip(' ,;')
        # Remove date ranges like (1994-2001) or (1994–2001)
        description = re.sub(r'\(\d{4}[–—-]\d{4}\)', '', description)
        description = description.strip()
        
        # Cut off at first comma or period that's NOT inside parentheses
        # But don't cut off at "c." (circa) which is used for approximate ages
        paren_depth = 0
        cutoff_pos = None
        for i, char in enumerate(description):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth = max(0, paren_depth - 1)
            elif char == ',' and paren_depth == 0:
                cutoff_pos = i
                break
            elif char == '.' and paren_depth == 0:
                # Check if this is "c." (circa) - skip it if so
                if i > 0 and description[i-1].lower() == 'c' and (i == 1 or not description[i-2].isalpha()):
                    continue
                cutoff_pos = i
                break
        if cutoff_pos is not None:
            description = description[:cutoff_pos].strip()
    
    # Now do name validation checks (with description available for defaults)
    
    # Skip if it looks like a category, file link, or other special page
    if ':' in article_title:
        return handle_error(f"Article title contains colon (special page): {article_title}", name, description)
    
    # Sanity check: name should look like a person's name (not too short, not a generic term)
    if len(name) < 2:
        return handle_error(f"Name too short: '{name}'", name, description)
    
    # Two letter names are allowed if they are a capital letter followed by a lowercase letter
    if len(name) == 2:
        if re.match(r'^[A-Z][a-z]$', name):
            print(msg_prefix().format('WARNING') + f"Name is only 2 characters: '{name}'")
        else:
            return handle_error(f"Invalid 2-character name (must be capital + lowercase): '{name}'", name, description)
    
    # Warn if name is just one word (most people have first and last name)
    if len(name.split()) == 1:
        print(msg_prefix().format('WARNING') + f"Name is only one word: '{name}'")
    
    # Now validate description
    
    # Must have content after the name (age and/or description)
    if not after_name.strip() or after_name.strip() == ',':
        return handle_error(f"No content after name: {name}", name, description)
    
    # Sanity check: description must be longer than 3 characters
    if not description or len(description) <= 3:
        return handle_error(f"Description too short for '{name}': '{description}'", name, description)
    
    # Description must have at least 2 words
    words = description.split()
    if len(words) == 1:
        return handle_error(f"Description is only one word for '{name}': '{description}'", name, description)
    
    # Description should not be just punctuation or special characters
    if re.match(r'^[\s\-:;,\.]+$', description):
        return handle_error(f"Description is just punctuation for '{name}': '{description}'", name, description)
    
    try:
        death_date = datetime(year, month, current_day)
        return {
            'name': name,
            'article_title': article_title,
            'death_date': death_date.strftime('%Y-%m-%d'),
            'description': description
        }
    except ValueError:
        print(msg_prefix().format('ERROR') + f"Invalid date {year}-{month}-{current_day} for '{name}'")
        return None


def parse_deaths_from_wikitext(wikitext: str, year: int, month: int) -> List[Dict]:
    """
    Parse the wikitext to extract deaths with their dates.
    Wikipedia "Deaths in [Month] [Year]" pages have a consistent format.
    Returns list of dicts with name, article_title, death_date, description.
    
    Handles:
    - Regular entries: * [[Name]], age, description
    - Group entries with subitems: parent bullet is ignored, subitems are processed
    - Stops at ==References== section
    """
    deaths = []
    current_day = None
    errors_count = 0
    
    lines = wikitext.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Stop at References section
        if re.match(r'^==+\s*References\s*==+', line, re.IGNORECASE):
            print(f"  Stopping at References section (line {i+1})")
            break
        
        # Also stop at other end sections like "See also"
        if re.match(r'^==+\s*(See also|External links|Notes|Further reading)\s*==+', line, re.IGNORECASE):
            print(f"  Stopping at '{line.strip()}' section (line {i+1})")
            break
        
        # Match day headers like "==1==" or "== 1 ==" or "===1==="
        day_match = re.match(r'^==+\s*(\d{1,2})\s*==+', line)
        if day_match:
            current_day = int(day_match.group(1))
            i += 1
            continue
        
        # Skip if we haven't found a day yet
        if current_day is None:
            i += 1
            continue
        
        # Check if this is a top-level bullet point (* but not **)
        if re.match(r'^\*[^\*]', line):
            # Look ahead to see if there are subitems (lines starting with **)
            has_subitems = False
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                # If next line is a subitem, this entry has subitems
                if next_line.startswith('**'):
                    has_subitems = True
                    break
                # If next line is a new top-level bullet or section header, no subitems
                elif next_line.startswith('*') and not next_line.startswith('**'):
                    break
                elif next_line.startswith('=='):
                    break
                elif next_line.strip() == '':
                    # Skip empty lines when looking ahead
                    j += 1
                    continue
                else:
                    # Some other content, stop looking
                    break
                j += 1
            
            if has_subitems:
                # Skip this parent entry (it's a group header, not a person)
                # Process the subitems instead
                parent_text = line.strip()
                i += 1
                while i < len(lines) and lines[i].startswith('**'):
                    subitem_line = lines[i]
                    # Only process subitems that have wiki links (actual people)
                    if '[[' in subitem_line:
                        death = parse_death_entry(subitem_line, year, month, current_day, i + 1, parent_item=parent_text)
                        if death:
                            deaths.append(death)
                        else:
                            errors_count += 1
                    i += 1
                continue
            else:
                # Regular entry without subitems - only process if it has a wiki link
                if '[[' in line:
                    death = parse_death_entry(line, year, month, current_day, i + 1)
                    if death:
                        deaths.append(death)
                    else:
                        errors_count += 1
                i += 1
                continue
        
        # Handle standalone subitems (** entries that aren't part of a group we already processed)
        elif line.startswith('**') and '[[' in line:
            death = parse_death_entry(line, year, month, current_day, i + 1)
            if death:
                deaths.append(death)
            else:
                errors_count += 1
            i += 1
            continue
        
        i += 1
    
    if errors_count > 0:
        print(f"  Encountered {errors_count} entries with format errors (skipped)")
    
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


def fetch_deaths_for_date_range(start_date: datetime, end_date: datetime, output_file: str) -> List[Dict]:
    """
    Fetch deaths for a range of dates by fetching monthly pages.
    Exports ALL deaths with their pageview counts (60 days after death).
    Writes to CSV live and supports resuming from existing file.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        output_file: Path to output CSV file
    """
    all_deaths = []
    completed_months: Set[Tuple[int, int]] = set()
    
    # Check if output file exists and load completed months
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}")
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_deaths.append(row)
                # Track which months are complete
                death_date = datetime.strptime(row['death_date'], '%Y-%m-%d')
                completed_months.add((death_date.year, death_date.month))
        print(f"  Loaded {len(all_deaths)} existing entries")
        print(f"  Completed months: {sorted(completed_months)}")
    
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
    
    # Filter out already completed months
    months_to_process = [m for m in months_to_fetch if m not in completed_months]
    
    if not months_to_process:
        print("All months already completed!")
        return all_deaths
    
    print(f"Months to process: {len(months_to_process)} (skipping {len(months_to_fetch) - len(months_to_process)} completed)")
    
    # Open file in append mode if exists, otherwise write mode with header
    file_exists = os.path.exists(output_file)
    
    for i, (year, month) in enumerate(months_to_process):
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
        month_deaths = []
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
            # Skip entries where article doesn't exist (pageviews == -1)
            skipped_no_article = 0
            for day_key in sorted(deaths_by_day.keys()):
                for death in deaths_by_day[day_key]:
                    views = pageview_counts.get(death['article_title'], 0)
                    if views == -1:
                        # Article doesn't exist in English Wikipedia
                        print(f"  WARNING: Skipping '{death['name']}' - no English Wikipedia article")
                        skipped_no_article += 1
                        continue
                    death['pageviews'] = views
                    death.pop('article_title', None)  # Remove helper field
                    month_deaths.append(death)
                    all_deaths.append(death)
            
            if skipped_no_article > 0:
                print(f"  Skipped {skipped_no_article} entries with no English Wikipedia article")
        
        # Write this month's deaths to CSV immediately
        if month_deaths:
            # Sort by date, then by pageviews (descending) within each day
            month_deaths.sort(key=lambda x: (x['death_date'], -x.get('pageviews', 0)))
            
            with open(output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['name', 'death_date', 'description', 'pageviews']
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(month_deaths)
            
            print(f"  Wrote {len(month_deaths)} deaths to {output_file}")
        
        # Polite delay between requests (Wikipedia asks for no more than 1 req/sec)
        if i < len(months_to_process) - 1:
            time.sleep(1.5)
    
    return all_deaths


def get_versioned_output_filename(base_output: str) -> str:
    """
    Generate output filename with version number.
    E.g., 'deaths_data.csv' -> 'deaths_data_v1.0.csv'
    """
    base, ext = os.path.splitext(base_output)
    return f"{base}_v{SCRIPT_VERSION}{ext}"


def save_to_csv(deaths: List[Dict], output_file: str):
    """
    Save the deaths data to a CSV file (final save, sorts all data).
    """
    if not deaths:
        print("No deaths to save!")
        return
    
    # Sort by date, then by pageviews (descending) within each day
    deaths.sort(key=lambda x: (x['death_date'], -int(x.get('pageviews', 0))))
    
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
        help='Output CSV file base name (default: deaths_data.csv). Version will be appended.'
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
    
    # Generate versioned output filename
    output_file = get_versioned_output_filename(args.output)
    
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Output file: {output_file}")
    print(f"Fetching ALL deaths from {args.start} to {args.end}")
    print("Pageviews will be fetched for each person (60 days after death)")
    print("=" * 50)
    
    deaths = fetch_deaths_for_date_range(start_date, end_date, output_file)
    
    # Final sort and save (to ensure proper ordering after resume)
    if deaths:
        save_to_csv(deaths, output_file)
    
    # Print summary by month
    print("\nSummary by month:")
    from collections import Counter
    month_counts = Counter(d['death_date'][:7] for d in deaths)
    for month, count in sorted(month_counts.items()):
        print(f"  {month}: {count} deaths")


if __name__ == "__main__":
    main()
