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
from typing import List, Dict, Optional, Set, Tuple, Union
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Script version - increment this when making changes to force new output file
SCRIPT_VERSION = "1.2"

# Global headers for Wikipedia API requests
WIKI_HEADERS = {
    "User-Agent": "CutoffDateTesting/1.0 (boasbakker123@gmail.com)"
}

# Bot credentials
WIKI_USERNAME = "Boasbakker@CutoffDateTesting"
WIKI_PASSWORD = "io6csmi2bvebdeb18e7dd2aojpvrlaim"

# Global session object
SESSION: Optional[requests.Session] = None


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter using Additive Increase / Multiplicative Decrease (AIMD).
    Ensures gaps between requests to avoid bursts.
    """
    def __init__(self, initial_rate=10, max_rate=15, min_rate=1, scale_up_interval=20):
        self.rate = initial_rate
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.scale_up_interval = scale_up_interval
        
        self.success_count = 0
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait(self):
        """Calculates wait time and sleeps outside the lock to avoid blocking other threads."""
        with self.lock:
            now = time.time()
            # Ensure we don't start requests more often than 1/rate
            target_time = self.last_request_time + (1.0 / self.rate)
            sleep_time = target_time - now
            
            if sleep_time <= 0:
                self.last_request_time = now
                return
            
            # Reserve this time slot
            self.last_request_time = target_time
            
        time.sleep(sleep_time)

    def report_success(self):
        """Report a successful request (not 429)."""
        with self.lock:
            self.success_count += 1
            if self.success_count >= self.scale_up_interval:
                old_rate = self.rate
                self.rate = min(self.max_rate, self.rate + 1)
                if self.rate > old_rate:
                    print(f"  [RateLimiter] Scaling up: {old_rate:.1f} -> {self.rate:.1f} req/s")
                self.success_count = 0

    def report_429(self):
        """Report a rate limit error (429)."""
        with self.lock:
            old_rate = self.rate
            self.rate = max(self.min_rate, self.rate * 0.5)
            self.success_count = 0
            print(f"  [RateLimiter] 429 detected! Scaling down: {old_rate:.1f} -> {self.rate:.1f} req/s")


def get_session() -> requests.Session:
    """
    Get or create the global requests Session.
    Logs in if credentials are provided and session is new.
    """
    global SESSION
    if SESSION is None:
        SESSION = requests.Session()
        SESSION.headers.update(WIKI_HEADERS)
        
        if WIKI_USERNAME and WIKI_PASSWORD:
            login_to_wikipedia(SESSION)
            
    return SESSION


def login_to_wikipedia(session: requests.Session):
    """
    Log in to Wikipedia using action=login and the provided bot credentials.
    """
    try:
        print(f"Logging in as {WIKI_USERNAME}...")
        api_url = "https://en.wikipedia.org/w/api.php"
        
        # 1. Get login token
        params_token = {
            "action": "query",
            "meta": "tokens",
            "type": "login",
            "format": "json"
        }
        resp = session.get(api_url, params=params_token, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        login_token = data.get("query", {}).get("tokens", {}).get("logintoken")
        
        if not login_token:
            print("  Failed to obtain login token.")
            return

        # 2. Log in
        login_data = {
            "action": "login",
            "lgname": WIKI_USERNAME,
            "lgpassword": WIKI_PASSWORD,
            "lgtoken": login_token,
            "format": "json"
        }
        resp = session.post(api_url, data=login_data, timeout=30)
        resp.raise_for_status()
        login_result = resp.json().get("login", {})
        
        if login_result.get("result") == "Success":
            print(f"  Login successful!")
        else:
            print(f"  Login failed: {login_result.get('reason', 'Unknown reason')}")
            
    except Exception as e:
        print(f"  Login error: {e}")


def resolve_error_interactive(line: str, error_msg: str, year: int, month: int, day: int, 
                            parsed_name: Optional[str] = None, parsed_description: Optional[str] = None, 
                            parent_item: Optional[str] = None) -> Optional[Dict]:
    """
    Interactively resolve a parsing error with the user.
    """
    print(f"  ERROR: {error_msg}")
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
    
    # Get description
    if parsed_description:
        user_description = input(f"  Enter description (press Enter to keep '{parsed_description}'): ").strip()
        if not user_description:
            user_description = parsed_description
    else:
        user_description = input("  Enter description: ").strip()
    
    if not user_description:
        print("  No description provided, skipping.")
        return None
    
    try:
        death_date = datetime(year, month, day)
        return {
            'name': user_name,
            'article_title': user_name,  # Use name as article title for manual entries
            'death_date': death_date.strftime('%Y-%m-%d'),
            'description': user_description
        }
    except ValueError:
        print(f"  Invalid date {year}-{month}-{day}, skipping.")
        return None


def get_page_creation_date(article_title: str, limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[datetime]:
    """
    Get the creation date of a Wikipedia article by fetching its first revision.
    Returns None if the page doesn't exist.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": article_title,
        "prop": "revisions",
        "rvprop": "timestamp",
        "rvlimit": "1",
        "rvdir": "newer",
        "format": "json"
    }
    for attempt in range(3):
        try:
            if limiter: limiter.wait()
            session = get_session()
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                if limiter: limiter.report_429()
                time.sleep(2 * (attempt + 1))
                continue
            if limiter: limiter.report_success()
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if "missing" in page:
                    return None
                revisions = page.get("revisions", [])
                if revisions:
                    ts = revisions[0]["timestamp"]  # e.g. "2020-01-01T12:00:00Z"
                    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            return None
        except requests.exceptions.RequestException:
            if attempt == 2:
                return None
            time.sleep(1)
    return None


def _fetch_pageviews_raw(safe_title: str, start_dt: datetime, end_dt: datetime, limiter: Optional[AdaptiveRateLimiter] = None) -> int:
    """
    Low-level pageviews fetch for a single title and date range.
    Returns view count, 0 if no data, or -1 if 404.
    """
    start = start_dt.strftime('%Y%m%d')
    end = end_dt.strftime('%Y%m%d')
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/user/{safe_title}/daily/{start}/{end}"
    )
    resp = None
    for attempt in range(3):
        try:
            if limiter: limiter.wait()
            session = get_session()
            resp = session.get(url, timeout=30)
            if resp.status_code == 429:
                if limiter: limiter.report_429()
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 404:
                if limiter: limiter.report_success()
                return -1
            resp.raise_for_status()
            if limiter: limiter.report_success()
            break
        except requests.exceptions.RequestException:
            if attempt == 2:
                raise
            time.sleep(1)
    if not resp:
        return 0
    data = resp.json()
    return sum(item.get('views', 0) for item in data.get('items', []))


def get_pageviews_sum(article_title: str, death_date: datetime, mode: str = 'after', limiter: Optional[AdaptiveRateLimiter] = None) -> int:
    """
    Get total pageviews for an article.
    mode='after': 1 day before death up to 60 days after (death incl. margin).
    mode='before': 60 days ending 6 days before death (death excl. margin).

    On 404 from the pageviews API, checks the page creation date:
    - Page doesn't exist at all -> returns -1 (error).
    - Page created after the counting period -> returns 0.
    - Page created during the counting period -> re-fetches from creation date.
    """
    safe_title = article_title.replace(' ', '_')

    if mode == 'after':
        start_dt = death_date - timedelta(days=1)
        end_dt = start_dt + timedelta(days=60)
    else:  # mode == 'before'
        end_dt = death_date - timedelta(days=6)
        start_dt = end_dt - timedelta(days=60)

    views = _fetch_pageviews_raw(safe_title, start_dt, end_dt, limiter=limiter)
    if views != -1:
        return views

    # 404 received — check if the page actually exists and when it was created
    creation_date = get_page_creation_date(article_title, limiter=limiter)
    if creation_date is None:
        return -1  # Page truly doesn't exist

    if creation_date > start_dt:
        # Page was created after the counting period starts -> strict ignore
        return 0

    # Page existed before the period but API still returned 404 (unusual)
    return 0


def get_pageviews_for_articles(article_entries: List[Dict], mode: str = 'after', max_workers: int = 25) -> Dict[str, int]:
    """
    Compute pageviews for multiple articles using parallel requests.
    article_entries: list of dicts with 'article_title' and 'death_date' (datetime).
    mode: 'before' or 'after'
    Returns mapping article_title -> pageviews_sum.
    """
    if not article_entries:
        return {}
    
    results: Dict[str, int] = {}
    total = len(article_entries)
    completed = 0
    
    limiter = AdaptiveRateLimiter(initial_rate=10, max_rate=15, min_rate=1)
    
    def fetch_one(entry: Dict) -> tuple:
        title = entry['article_title']
        death_dt = entry['death_date']
        # Rely on the internal adaptive limiter for timing
        return title, get_pageviews_sum(title, death_dt, mode=mode, limiter=limiter)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, entry): entry for entry in article_entries}
        for future in as_completed(futures):
            title, views = future.result()
            results[title] = views
            completed += 1
            if completed % 5 == 0 or completed == total:
                print(f"    Pageviews: {completed}/{total} (Current rate: {limiter.rate:.1f} req/s)   ", end="\r")
    
    print()  # finish line
    return results


def _parse_birth_date_from_infobox(article_title: str, session: requests.Session, limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[str]:
    """
    Parse a full birth date (YYYY-MM-DD) from a Wikipedia article's infobox wikitext.
    Looks for {{birth date|YYYY|M|D}} or {{Birth date and age|YYYY|M|D|...}} templates.
    Returns None if no full date is found.
    """
    try:
        if limiter: limiter.wait()
        resp = session.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query",
            "titles": article_title,
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2"
        }, timeout=30)
        if resp.status_code == 429:
            if limiter: limiter.report_429()
        else:
            if limiter: limiter.report_success()
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", [])
        if not pages or "missing" in pages[0]:
            return None
        content = pages[0].get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
    except Exception:
        return None

    # Match {{birth date|YYYY|M|D...}} or {{Birth date and age|YYYY|M|D...}}
    pattern = r'\{\{[Bb]irth[\s_]date(?:[\s_]and[\s_]age)?\|(\d{4})\|(\d{1,2})\|(\d{1,2})'
    match = re.search(pattern, content)
    if match:
        year, month, day = match.group(1), match.group(2).zfill(2), match.group(3).zfill(2)
        return f"{year}-{month}-{day}"
    return None


def get_birth_dates_for_articles(articles: List[str], max_workers: int = 10, limiter: Optional[AdaptiveRateLimiter] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Fetch strict birth dates (YYYY-MM-DD) for a list of articles using Wikidata.
    Two-step process:
    1. Wikipedia API: Get Wikidata QID for each article (handles redirects).
    2. Wikidata API: Get 'Date of Birth' (P569) for each QID.
       - If precision < 11, fall back to parsing the Wikipedia infobox.
    3. Return results and per-article skip reasons for failures.
    
    Returns tuple: (birth_dates, skip_reasons)
    - birth_dates: article_title -> birth_date_string (YYYY-MM-DD)
    - skip_reasons: article_title -> reason string (for articles without a birth date)
    """
    if not articles:
        return {}, {}
        
    print(f"  Fetching birth dates for {len(articles)} articles from Wikidata...")

    # Session for connection pooling
    session = get_session()
    # session.headers.update(WIKI_HEADERS) # Already set in get_session
    
    # Step 1: Get QIDs in chunks of 50
    article_to_qid = {}
    chunk_size = 50
    
    for i in range(0, len(articles), chunk_size):
        chunk = articles[i:i+chunk_size]
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(chunk),
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "redirects": "1"
        }
        
        success = False
        for attempt in range(3):
            try:
                if limiter: limiter.wait()
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    if limiter: limiter.report_429()
                    time.sleep(2 * (attempt + 1))
                    continue
                if limiter: limiter.report_success()
                resp.raise_for_status()
                data = resp.json()
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    Error fetching QIDs for chunk: {e}")
                time.sleep(1)
        
        if not success:
            continue
            
        # Parse data...
        pages = data.get("query", {}).get("pages", {})
        
        # Create a map of normalized/redirected titles to QIDs
        redirect_map = {}
        if "query" in data:
            if "redirects" in data["query"]:
                for r in data["query"]["redirects"]:
                    redirect_map[r["to"]] = r["from"]
            if "normalized" in data["query"]:
                for n in data["query"]["normalized"]:
                    redirect_map[n["to"]] = n["from"]
        
        for page_id, page in pages.items():
            if "missing" in page or "pageprops" not in page:
                continue
                
            title = page.get("title")
            qid = page.get("pageprops", {}).get("wikibase_item")
            
            if not qid:
                continue
            
            # 1. Direct match
            if title in chunk:
                article_to_qid[title] = qid
            # 2. Mapped match (redirect target -> original)
            elif title in redirect_map:
                original = redirect_map[title]
                if original in chunk:
                        article_to_qid[original] = qid
        
        # Polite delay between chunks
        time.sleep(0.5)
            
    # Step 2: Get Birth Dates and P31 (instance-of) from Wikidata
    results = {}
    skip_reasons = {}
    qids = list(set(article_to_qid.values()))
    qid_to_birthdate = {}
    qid_to_low_precision = {}  # QIDs with P569 but precision < 11
    disambig_qids = set()  # QIDs that are disambiguation pages
    DISAMBIG_TYPES = {"Q4167410", "Q22808320"}  # Wikimedia disambiguation page types
    
    for i in range(0, len(qids), chunk_size):
        chunk_qids = qids[i:i+chunk_size]
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(chunk_qids),
            "props": "claims",
            "languages": "en" 
        }
        
        success = False
        for attempt in range(3):
            try:
                if limiter: limiter.wait()
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    if limiter: limiter.report_429()
                    time.sleep(2 * (attempt + 1))
                    continue
                if limiter: limiter.report_success()
                resp.raise_for_status()
                data = resp.json()
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    print(f"    Error fetching Wikidata claims for chunk: {e}")
                time.sleep(1)

        if not success:
            continue
            
        entities = data.get("entities", {})
        for qid, entity in entities.items():
            if "claims" not in entity:
                continue
            
            # Check if this is a disambiguation page (P31 includes Q4167410 or Q22808320)
            p31_claims = entity["claims"].get("P31", [])
            for p31 in p31_claims:
                p31_val = p31.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(p31_val, dict) and p31_val.get("id") in DISAMBIG_TYPES:
                    disambig_qids.add(qid)
                    break
            
            if qid in disambig_qids:
                continue  # Skip birth date extraction for disambiguation pages
            
            # P569: Date of Birth
            birth_claims = entity["claims"].get("P569", [])
            if not birth_claims:
                continue
            
            for claim in birth_claims:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                if not datavalue: continue
                
                value = datavalue.get("value", {})
                if not isinstance(value, dict): continue
                
                # Precision: 11 = day
                precision = value.get("precision", 0)
                time_str = value.get("time", "")
                
                if precision >= 11 and time_str:
                    # Format is typically "+1946-06-14T00:00:00Z"
                    if time_str.startswith('+'):
                        clean_time = time_str[1:]
                    else:
                        clean_time = time_str
                        
                    # Extract first 10 chars: YYYY-MM-DD
                    if len(clean_time) >= 10:
                        birth_date = clean_time[:10]
                        qid_to_birthdate[qid] = birth_date
                        break
                elif time_str:
                    # Low precision - record for potential infobox fallback
                    qid_to_low_precision[qid] = precision
                    break
    
    # Step 2b: Infobox fallback for low-precision Wikidata entries
    low_prec_articles = [
        a for a, q in article_to_qid.items() 
        if q not in qid_to_birthdate and q in qid_to_low_precision
    ]
    if low_prec_articles:
        print(f"  Trying infobox fallback for {len(low_prec_articles)} low-precision entries...")
        
        def fetch_infobox(art):
            return art, _parse_birth_date_from_infobox(art, session, limiter=limiter)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_infobox, article) for article in low_prec_articles]
            for future in as_completed(futures):
                article, infobox_date = future.result()
                if infobox_date:
                    results[article] = infobox_date
                    # print(f"    Parsed birth date from infobox for '{article}': {infobox_date}")
                    
    # Map back: Article -> QID -> Birth Date
    count = 0
    for article, qid in article_to_qid.items():
        if article in results:
            count += 1  # Already found via infobox fallback
            continue
        if qid in qid_to_birthdate:
            results[article] = qid_to_birthdate[qid]
            count += 1
        else:
            # Determine specific skip reason
            if qid in disambig_qids:
                skip_reasons[article] = "Disambiguation page — use --add-titles to set correct article_title"
            elif qid in qid_to_low_precision:
                skip_reasons[article] = f"Birth date precision too low (Wikidata precision={qid_to_low_precision[qid]}, need day-level)"
            else:
                skip_reasons[article] = "No birth date found on Wikidata"
    
    # Also record skip reasons for articles that didn't get a QID at all
    for article in articles:
        if article not in results and article not in skip_reasons:
            skip_reasons[article] = "No Wikidata entry found"
            
    print(f"  Found full birth dates for {count}/{len(articles)} articles")
    return results, skip_reasons


def get_wikipedia_page_content(page_title: str, limiter: Optional[AdaptiveRateLimiter] = None) -> Optional[str]:
    """
    Fetch the wikitext content of a Wikipedia page using the API.
    Uses a single API call to get the full page content.
    Includes retry logic for rate limits.
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
    
    for attempt in range(5):
        try:
            if limiter: limiter.wait()
            session = get_session()
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 429:
                if limiter: limiter.report_429()
                wait = 2 * (attempt + 1)
                print(f"  Rate limit (429) hitting for main page. Waiting {wait}s...")
                time.sleep(wait)
                continue
            if limiter: limiter.report_success()
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if pages and "revisions" in pages[0]:
                return pages[0]["revisions"][0]["slots"]["main"]["content"]
            if pages and "missing" in pages[0]:
                print(f"  Page '{page_title}' does not exist (missing in API response).")
                return None
            print(f"  Unexpected API response format for '{page_title}'.")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"  Request error for '{page_title}': {e}")
            if attempt == 4:
                print(f"  Failed to fetch '{page_title}' after 5 attempts.")
                return None
            time.sleep(1)
            
    return None


def parse_death_entry(line: str, year: int, month: int, current_day: int, line_num: int, mode: str = 'after', parent_item: Optional[str] = None, silent: bool = False, defer_errors: bool = False) -> Union[Optional[Dict], Dict]:
    """
    Parse a single death entry line and return a dict or None if invalid.
    Expected format: * [[Name]], age, description  OR  * [[Name]], description (if age unknown)
    
    Requirements:
    - Name must be a wiki link [[Name]] or [[Name|Display Name]]
    - Age is optional (may be unknown)
    - Description must be present and at least 2 words
    
    Args:
        parent_item: If this is a subitem, the text of the parent bullet point
        defer_errors: If True, returns a dict with 'is_error': True instead of prompting user.
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
    
    def handle_error(error_msg: str, parsed_name: Optional[str] = None, parsed_description: Optional[str] = None) -> Union[Optional[Dict], Dict]:
        """Handle an error by prompting user for input, or deferring it."""
        if silent:
            return None
            
        # Optimization: Pre-check if page exists/is valid for our period before asking user
        if parsed_name:
            try:
                death_dt = datetime(year, month, current_day)
                if mode == 'after':
                    start_dt = death_dt - timedelta(days=1)
                else:  # mode == 'before'
                    end_dt = death_dt - timedelta(days=6)
                    start_dt = end_dt - timedelta(days=60)
                
                creation_date = get_page_creation_date(parsed_name)
                
                if creation_date is None:
                    # print(msg_prefix().format('SKIP') + f"Page '{parsed_name}' does not exist (skipping manual entry)")
                    return None
                
                if creation_date > start_dt:
                     # print(msg_prefix().format('SKIP') + f"Page '{parsed_name}' created too late ({creation_date.date()} > {start_dt.date()})")
                     return None
                     
            except ValueError:
                pass # Invalid date, let user try to fix

        if defer_errors:
            # Return error state for later resolution
            return {
                'is_error': True,
                'raw_line': line,
                'error_msg': error_msg,
                'parsed_name': parsed_name,
                'parsed_description': parsed_description,
                'year': year,
                'month': month,
                'day': current_day,
                'parent_item': parent_item,
                'article_title': parsed_name if parsed_name else None
            }

        return resolve_error_interactive(line, error_msg, year, month, current_day, parsed_name, parsed_description, parent_item)

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
        if silent: return None
        return handle_error(f"Name too short: '{name}'", name, description)
    
    # Two letter names are allowed if they are a capital letter followed by a lowercase letter
    if len(name) == 2:
        if re.match(r'^[A-Z][a-z]$', name):
            if not silent:
                 print(msg_prefix().format('WARNING') + f"Name is only 2 characters: '{name}'")
        else:
            if silent: return None
            return handle_error(f"Invalid 2-character name (must be capital + lowercase): '{name}'", name, description)
    
    # Warn if name is just one word (most people have first and last name)
    if len(name.split()) == 1:
        if not silent:
             print(msg_prefix().format('WARNING') + f"Name is only one word: '{name}'")
    
    # Now validate description
    
    # Must have content after the name (age and/or description)
    if not after_name.strip() or after_name.strip() == ',':
        if not silent: # In silent mode, accept it if we have a name
             return handle_error(f"No content after name: {name}", name, description)
    
    # Sanity check: description must be longer than 3 characters
    if not description or len(description) <= 3:
        if not silent:
            return handle_error(f"Description too short for '{name}': '{description}'", name, description)
    
    # Description must have at least 2 words
    words = description.split() if description else []
    if len(words) == 1:
        if not silent:
            return handle_error(f"Description is only one word for '{name}': '{description}'", name, description)
    
    # Description should not be just punctuation or special characters
    if description and re.match(r'^[\s\-:;,\.]+$', description):
        if not silent:
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


def parse_deaths_from_wikitext(wikitext: str, year: int, month: int, mode: str = 'after', silent: bool = False, defer_errors: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse the wikitext to extract deaths with their dates.
    Wikipedia "Deaths in [Month] [Year]" pages have a consistent format.
    Returns tuple: (list of valid deaths, list of pending error dicts).
    
    Handles:
    - Regular entries: * [[Name]], age, description
    - Group entries with subitems: parent bullet is ignored, subitems are processed
    - Stops at ==References== section
    """
    deaths = []
    pending_errors = []
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
                        death = parse_death_entry(subitem_line, year, month, current_day, i + 1, mode=mode, parent_item=parent_text, silent=silent, defer_errors=defer_errors)
                        if death:
                            if death.get('is_error'):
                                pending_errors.append(death)
                            else:
                                deaths.append(death)
                        else:
                            errors_count += 1
                    i += 1
                continue
            else:
                # Regular entry without subitems - only process if it has a wiki link
                if '[[' in line:
                    death = parse_death_entry(line, year, month, current_day, i + 1, mode=mode, silent=silent, defer_errors=defer_errors)
                    if death:
                        if death.get('is_error'):
                            pending_errors.append(death)
                        else:
                            deaths.append(death)
                    else:
                        errors_count += 1
                i += 1
                continue
        
        # Handle standalone subitems (** entries that aren't part of a group we already processed)
        elif line.startswith('**') and '[[' in line:
            death = parse_death_entry(line, year, month, current_day, i + 1, mode=mode, defer_errors=defer_errors)
            if death:
                if death.get('is_error'):
                    pending_errors.append(death)
                else:
                    deaths.append(death)
            else:
                errors_count += 1
            i += 1
            continue
        
        i += 1
    
    if errors_count > 0:
        print(f"  Encountered {errors_count} entries with format errors (skipped)")
    
    if defer_errors:
        print(f"  Found {len(deaths)} valid deaths and {len(pending_errors)} items needing review")
    
    return deaths, pending_errors


def fetch_deaths_for_month(year: int, month: int, mode: str = 'after', defer_errors: bool = False, limiter: Optional[AdaptiveRateLimiter] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch all deaths for a given month and year.
    Returns: (valid_deaths, pending_errors)
    """
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    month_name = month_names[month - 1]
    page_title = f"Deaths in {month_name} {year}"
    
    print(f"Fetching: {page_title}")
    
    wikitext = get_wikipedia_page_content(page_title, limiter=limiter)
    if wikitext is None:
        print(f"  Could not fetch page: {page_title}")
        return [], []
    
    return parse_deaths_from_wikitext(wikitext, year, month, mode=mode, defer_errors=defer_errors)


def fetch_deaths_for_date_range(start_date: datetime, end_date: datetime, output_file: str, mode: str = 'after') -> List[Dict]:
    """
    Fetch deaths for a range of dates by fetching monthly pages.
    Process month-by-month to allow incremental saving and deferred prompting.
    
    Flow per month:
    1. Fetch wikitext -> get valid deaths + pending errors.
    2. Fetch pageviews for ALL candidates (valid + pending).
    3. Filter out 0-view items.
    4. Interactively resolve pending errors if they have views.
    5. Fetch birth dates.
    6. Save to CSV.
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
                try:
                    death_date = datetime.strptime(row['death_date'], '%Y-%m-%d')
                    completed_months.add((death_date.year, death_date.month))
                except ValueError:
                    continue
        print(f"  Loaded {len(all_deaths)} existing entries")
        print(f"  Completed months: {sorted(list(completed_months))}")

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
        # 0. Initialize Limiter for the whole run if not already present
        # (Though we mostly care about it within each month's pageview chunk)
        limiter = AdaptiveRateLimiter(initial_rate=10, max_rate=15, min_rate=1)

        # 1. Fetch & Parse
        valid_deaths, pending_errors = fetch_deaths_for_month(year, month, mode=mode, defer_errors=True, limiter=limiter)
        
        if not valid_deaths and not pending_errors:
            continue
            
        combined_items = valid_deaths + pending_errors
        
        # 2. Prepare for Pageviews
        all_article_entries = []
        for item in combined_items:
            if item.get('article_title'):
                try:
                    d_date = datetime.strptime(item.get('death_date', ''), '%Y-%m-%d')
                except ValueError:
                    d_date = datetime(year, month, 1)

                all_article_entries.append({
                    'article_title': item['article_title'],
                    'death_date': d_date
                })

        # 3. Fetch Pageviews (Parallel)
        # Lowered max_workers to 10 to prevent 429 errors
        print(f"  Fetching pageview counts ({mode} mode) for {len(all_article_entries)} articles...")
        pageview_counts = get_pageviews_for_articles(all_article_entries, mode=mode, max_workers=10)
        
        # 4. Filter & Resolve
        month_final_deaths = []
        skipped_no_views = 0
        skipped_user = 0
        resolved_count = 0
        
        # Process Valid
        for item in valid_deaths:
            views = pageview_counts.get(item['article_title'], 0)
            if views <= 0:
                skipped_no_views += 1
                continue
            item['pageviews'] = views
            month_final_deaths.append(item)
            
        # Process Pending
        if pending_errors:
            print(f"  Checking {len(pending_errors)} pending items for view counts...")
            for item in pending_errors:
                title = item.get('article_title')
                views = pageview_counts.get(title, 0) if title else 0
                
                if views <= 0:
                    skipped_no_views += 1
                    continue
                
                # Has views -> Prompt
                print(f"    Pending item '{item.get('parsed_name')}' has {views} views. Resolving...")
                resolved = resolve_error_interactive(
                    item['raw_line'], 
                    item['error_msg'], 
                    item['year'], 
                    item['month'], 
                    item['day'], 
                    item['parsed_name'], 
                    item['parsed_description'], 
                    item['parent_item']
                )
                
                if resolved:
                    resolved['pageviews'] = views
                    month_final_deaths.append(resolved)
                    resolved_count += 1
                else:
                    skipped_user += 1
        
        if not month_final_deaths:
            print(f"  No valid deaths found for {year}-{month} after filtering.")
            continue

        # 5. Get Birth Dates
        titles_to_check = [d['article_title'] for d in month_final_deaths]
        birth_dates_map, birth_skip_reasons = get_birth_dates_for_articles(titles_to_check, limiter=limiter)
        
        deaths_to_save = []
        skipped_bad_date = 0
        
        for d in month_final_deaths:
            bdate = birth_dates_map.get(d['article_title'])
            if bdate:
                d['birth_date'] = bdate
                # Cleanup
                d.pop('is_error', None)
                d.pop('raw_line', None)
                d.pop('error_msg', None)
                d.pop('parsed_name', None)
                d.pop('parsed_description', None)
                d.pop('year', None)
                d.pop('month', None)
                d.pop('day', None)
                d.pop('parent_item', None)
                
                # Check date range (should be valid by month, but good practice)
                d_dt = datetime.strptime(d['death_date'], '%Y-%m-%d')
                if start_date <= d_dt <= end_date:
                    deaths_to_save.append(d)
                    all_deaths.append(d)
            else:
                skipped_bad_date += 1
        
        if skipped_bad_date > 0:
             print(f"  Skipped {skipped_bad_date} entries missing birth dates.")

        # 6. Save
        if deaths_to_save:
            deaths_to_save.sort(key=lambda x: (x['death_date'], -x.get('pageviews', 0)))
            
            with open(output_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['name', 'death_date', 'birth_date', 'description', 'pageviews']
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(deaths_to_save)
            
            print(f"  Saved {len(deaths_to_save)} deaths for {year}-{month}.")
        
        # Delay
        if i < len(months_to_process) - 1:
            time.sleep(1.0)
            
    return all_deaths


def get_versioned_output_filename(base_output: str, mode: str) -> str:
    """
    Generate output filename with version number and mode.
    E.g., 'deaths_data.csv' -> 'deaths_data_after_v1.2.csv'
    """
    base, ext = os.path.splitext(base_output)
    return f"{base}_{mode}_v{SCRIPT_VERSION}{ext}"





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
        fieldnames = ['name', 'death_date', 'birth_date', 'description', 'pageviews']
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
        default='2020-01-01',
        help='Start date in YYYY-MM-DD format (default: 2020-01-01)'
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
        help='Output CSV file base name (default: deaths_data.csv). Mode and version will be appended.'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['before', 'after'],
        default='before',
        help='Pageview counting mode: "after" (60 days starting 1 day before death) or "before" (default, 60 days ending 6 days before death)'
    )
    args = parser.parse_args()
    
    print(f"Script version: {SCRIPT_VERSION}")
    
    # Initialize session and log in if configured
    get_session()
    
    # REGULAR MODE: Fetch from Wikipedia
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
    output_file = get_versioned_output_filename(args.output, args.mode)
    
    print(f"Output file: {output_file}")
    print(f"Fetching ALL deaths from {args.start} to {args.end}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'after':
        print("Pageviews: 60 days starting 1 day before death (includes death spike)")
    else:
        print("Pageviews: 60 days ending 6 days before death (baseline fame)")
    print("=" * 50)
    
    deaths = fetch_deaths_for_date_range(start_date, end_date, output_file, mode=args.mode)
    
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
