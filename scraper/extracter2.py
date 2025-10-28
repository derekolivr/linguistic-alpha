#!/usr/bin/env python3
"""
SEC Quarterly 10-Q Direct Data Extractor (Full Pipeline)
*** CRITICAL FIX: Removed overly strict content size check in extraction function ***
"""
import os
import sys
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
from ixbrlparse import IXBRL
from io import StringIO
from typing import Dict, List, Any

# --------------------------------------------------------------------
# USER SETTINGS (UPDATE THESE!)
# --------------------------------------------------------------------
# !!! CRITICAL: REPLACE 'your.email@example.com' with your actual email address
USER_AGENT = "MyPersonalScraper/1.1 (sasti.saravanan@gmail.com)" # <-- CHANGE THIS
REQUESTS_PER_SEC = 1.5 # Conservative rate limit
YEARS_BACK = 4
FILING_TYPE = "10-Q"

# Initialize Session with REQUIRED User-Agent
session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def _safe_get(url, params=None, max_retries=3, backoff=2.0):
    """Handles rate limiting and retries for SEC requests."""
    # Enforce time delay before request
    time.sleep(1.0 / REQUESTS_PER_SEC) 
    
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            if r.status_code == 403:
                print(f"Error 403 Forbidden: Request denied by SEC. CHECK YOUR USER-AGENT and wait a few minutes.")
                raise RuntimeError(f"Failed to fetch {url} due to 403.") 
            
            wait = backoff * (2 ** attempt)
            print(f"Error {e} on {url} (Status: {r.status_code}). Retrying in {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            wait = backoff * (2 ** attempt)
            print(f"General Error {e} on {url}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
            
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def get_cik_for_ticker(ticker):
    """Uses SEC's official ticker → CIK mapping."""
    ticker = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    r = _safe_get(url)
    data = r.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None

def list_recent_filings(cik, years_back=4):
    """Lists recent filings of the specified type from the SEC API."""
    base_url = "https://data.sec.gov/submissions/CIK" + cik + ".json"
    r = _safe_get(base_url)
    data = r.json()

    cutoff_date = datetime.now() - timedelta(days=years_back * 365)
    filings = []
    
    if "recent" in data.get("filings", {}):
        for form, date, accession in zip(
            data["filings"]["recent"]["form"],
            data["filings"]["recent"]["filingDate"],
            data["filings"]["recent"]["accessionNumber"],
        ):
            if form == FILING_TYPE:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if date_obj >= cutoff_date:
                    accession_nodash = accession.replace("-", "")
                    link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_nodash}/{accession}-index.html"
                    filings.append({"filing_date": date, "accession": accession_nodash, "link": link})
    return filings

def find_ixbrl_report_url(index_url: str) -> str | None:
    """Scrapes the index page to find the direct link to the primary iXBRL report."""
    r = _safe_get(index_url)
    soup = BeautifulSoup(r.text, "lxml")
    
    table = soup.find("table", class_="tableFile") or soup.find("table", class_="tableFile2")
    if not table:
        return None

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) > 2:
            link_tag = cols[2].find("a")
            doc_type = cols[3].text.strip() if len(cols) > 3 else ""
            
            if link_tag and ('10-Q' in doc_type or doc_type in ['EX-101.INS', '']) and link_tag.text.lower().endswith(('.htm', '.html', '.xhtml')):
                href = link_tag.get("href")
                return urljoin("https://www.sec.gov", href)
    
    return None

def extract_remote_ixbrl_facts(ixbrl_url: str, filing_name: str) -> List[Dict[str, Any]]:
    """
    Downloads iXBRL content from a URL, extracts all facts, 
    using the successful download-and-parse method without the strict size check.
    """
    print(f"   Downloading iXBRL report from: {ixbrl_url}")
    try:
        # Step 1: Download the file content using the session and _safe_get
        response = _safe_get(ixbrl_url)
        content = response.text
        
        # --- FIX APPLIED HERE: REMOVED the content size check (content_size_bytes < 10240) ---
        
        print(f"   Successfully downloaded {len(content.encode('utf-8'))} bytes. Proceeding to parse.")
        
        # Step 2: Parse the iXBRL content using StringIO
        file_content = StringIO(content)
        ixbrl_doc = IXBRL(file_content)
        facts_list = ixbrl_doc.to_table(fields='all') 

        # Step 3: Add source metadata
        for fact in facts_list:
            fact['SourceFiling'] = filing_name
            fact['SourceURL'] = ixbrl_url
            
        return facts_list

    except RuntimeError as e:
        print(f"   ❌ Network error during download: {e}")
        return []
    except Exception as e:
        print(f"   ❌ Parsing error: {e}")
        return []

# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------

def run_direct_extraction_pipeline(ticker, max_filings=12):
    """Executes the full direct extraction pipeline."""
    # Check for placeholder email before starting
    if "your.real.email@provider.com" in USER_AGENT:
        sys.exit("!!! CRITICAL: Please update the USER_AGENT with your actual email address before running. !!!")

    print(f"--- Starting SEC Direct Extraction for {ticker} (Type: {FILING_TYPE}) ---")
    
    # 1. Get CIK
    cik = get_cik_for_ticker(ticker)
    if not cik:
        print(f"❌ Could not find CIK for ticker {ticker}.")
        return pd.DataFrame()
    print(f"✅ Found CIK: {cik}")

    # 2. List Filings
    filings = list_recent_filings(cik, years_back=YEARS_BACK)
    filings = filings[:max_filings]
    if not filings:
        print(f"No recent {FILING_TYPE} filings found in the last {YEARS_BACK} years.")
        return pd.DataFrame()

    all_extracted_facts = []
    
    for i, f in enumerate(filings, 1):
        filing_name = f"{ticker}_{f['filing_date']}_{f['accession']}"
        print(f"\n--- Processing Filing [{i}/{len(filings)}]: {filing_name} ---")
        
        # 3. Find Direct iXBRL Link
        try:
            ixbrl_url = find_ixbrl_report_url(f['link'])
        except RuntimeError:
            print(f"   Skipping filing due to prior network error on index page: {f['link']}")
            continue # Move to the next filing
        
        if ixbrl_url:
            print(f"   Found primary iXBRL link. Proceeding to extract data...")
            
            # 4. Extract Data Directly
            facts = extract_remote_ixbrl_facts(ixbrl_url, filing_name)
            
            if facts:
                all_extracted_facts.extend(facts)
                print(f"   ✅ SUCCESS: Extracted {len(facts)} facts.")
            else:
                print("   ⚠️ No facts extracted from this filing.")
        else:
            print(f"   ❌ Could not find the primary iXBRL report link on index page: {f['link']}")

    if not all_extracted_facts:
        print("\n❌ Final result: No facts were extracted from any filing.")
        return pd.DataFrame()

    # 5. Final DataFrame Consolidation and Formatting
    df_combined = pd.DataFrame(all_extracted_facts)
    df_combined.rename(columns={'name': 'ConceptName', 
                                 'startdate': 'StartDate', 
                                 'enddate': 'EndDate'}, inplace=True)
    
    print("\n--- ✅ Extraction Pipeline Complete ---")
    print(f"Total rows in final DataFrame: {len(df_combined)}")
    
    return df_combined

if __name__ == "__main__":
    
    # Check for placeholder email before starting
    if "your.real.email@provider.com" in USER_AGENT:
        sys.exit("!!! CRITICAL: Please update the USER_AGENT with your actual email address before running. !!!")

    ticker_input = input(f"Enter ticker symbol (e.g., AAPL) for {FILING_TYPE} filings: ").strip().upper()
    
    # Execute the pipeline
    final_data_frame = run_direct_extraction_pipeline(ticker_input, max_filings=12)
    
    if not final_data_frame.empty:
        # Save the final data to a CSV file
        output_file = f"xbrl_facts_{ticker_input}_10Q_direct.csv"
        final_data_frame.to_csv(output_file, index=False)
        print(f"\nSaved final consolidated data to {output_file}")