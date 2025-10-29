#!/usr/bin/env python3
"""
SEC Quarterly 10-Q Filing Downloader (last 4 years only, fixed for ticker lookup)

Usage:
    python sec_quarterly_scraper.py AAPL
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta

# --------------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------------
USER_AGENT = "QuarterlyScraper/1.0 (your.email@example.com)"  # <-- change this
REQUESTS_PER_SEC = 2
YEARS_BACK = 4
FILING_TYPE = "10-Q"

session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------
def _safe_get(url, params=None, max_retries=3, backoff=1.0):
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=30)
            r.raise_for_status()
            time.sleep(1.0 / REQUESTS_PER_SEC)
            return r
        except Exception as e:
            wait = backoff * (2 ** attempt)
            print(f"Error {e} on {url}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}")

def get_cik_for_ticker(ticker):
    """Use SEC's official ticker → CIK JSON mapping"""
    ticker = ticker.upper().strip()
    url = "https://www.sec.gov/files/company_tickers.json"
    r = _safe_get(url)
    data = r.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            cik_str = str(entry["cik_str"]).zfill(10)  # SEC expects 10 digits
            return cik_str
    return None

def list_recent_10q_filings(cik, count=40, years_back=4):
    """List 10-Q filings in the last N years"""
    base_url = "https://data.sec.gov/submissions/CIK" + cik + ".json"
    r = _safe_get(base_url)
    data = r.json()

    cutoff_date = datetime.now() - timedelta(days=years_back * 365)
    filings = []
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
                filings.append({"filing_date": date, "link": link})
    return filings

def download_filing_documents(filing_detail_url, out_dir="downloads"):
    """Download .htm, .xml, .xsd, .zip files from the filing"""
    os.makedirs(out_dir, exist_ok=True)
    r = _safe_get(filing_detail_url)
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", class_="tableFile") or soup.find("table", class_="tableFile2")
    saved = []
    if not table:
        return saved

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if not cols:
            continue
        link_tag = cols[2].find("a") if len(cols) > 2 else row.find("a")
        if not link_tag:
            continue
        href = link_tag.get("href")
        filename = link_tag.text.strip()
        full_url = urljoin("https://www.sec.gov", href)
        if any(filename.lower().endswith(ext) for ext in (".htm", ".html", ".xhtml", ".xml", ".zip", ".xsd")):
            safe_name = re.sub(r"[^\w\-_\. ]", "_", filename)
            outpath = os.path.join(out_dir, safe_name)
            try:
                rr = _safe_get(full_url)
                with open(outpath, "wb") as f:
                    f.write(rr.content)
                print(f"Saved {outpath}")
                saved.append(outpath)
            except Exception as e:
                print(f"Failed {filename}: {e}")
    return saved

def download_company_quarterly_filings(ticker, max_filings=10):
    print(f"Resolving CIK for {ticker}...")
    cik = get_cik_for_ticker(ticker)
    if not cik:
        print(f"❌ Could not find CIK for ticker {ticker}. Check ticker symbol.")
        return []
    print(f"✅ Found CIK: {cik}")

    filings = list_recent_10q_filings(cik, count=max_filings, years_back=YEARS_BACK)
    if not filings:
        print("No 10-Q filings in the last 4 years.")
        return []

    all_saved = []
    for i, f in enumerate(filings, 1):
        print(f"[{i}] {f['filing_date']} -> {f['link']}")
        folder = os.path.join("downloads", ticker, f"filing_{i}")
        os.makedirs(folder, exist_ok=True)
        saved = download_filing_documents(f["link"], folder)
        all_saved.extend(saved)
    return all_saved

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    files = download_company_quarterly_filings(ticker, max_filings=12)
    print("\nDownloaded files:")
    for f in files:
        print("  ", f)

