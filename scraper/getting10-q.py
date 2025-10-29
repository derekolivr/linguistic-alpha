import requests
import time
import json
import os

# --- Configuration and Constants ---
# The SEC requires a proper User-Agent header for API and web requests.
# REPLACE 'your.email@example.com' with your actual email address.
HEADERS = {"User-Agent": "Mass10QURLCollector your.email@example.com"} 
BASE_SEC_URL = "https://www.sec.gov"
SUBMISSIONS_API_BASE = "https://data.sec.gov/submissions/CIK"
COMPANY_CIK_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
MAX_CIKS_TO_PROCESS = 50 
TARGET_FORMS = ['10-Q', '10-Q/A'] # Only collect these forms

# --- Function 1: Get CIKs (Padded) ---
def get_top_ciks(max_count):
    """Fetches the list of all CIKs from the SEC and returns the top N."""
    print(f"1. Fetching all company CIKs and Tickers from the SEC...")
    time.sleep(0.1) 
    
    try:
        response = requests.get(COMPANY_CIK_TICKERS_URL, headers=HEADERS)
        response.raise_for_status()
        all_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch company tickers: {e}")
        return {}

    ciks = {}
    for i in range(max_count):
        key = str(i)
        if key in all_data:
            entry = all_data[key]
            # CIKs are PADDED and stored as 10-digit strings
            cik_padded = str(entry['cik_str']).zfill(10)
            ciks[cik_padded] = entry['ticker']
    
    print(f"Successfully collected the top {len(ciks)} CIKs.")
    return ciks

# --- Function 2: Fetch Submissions and Extract ONLY 10-Q URLs ---
def get_10q_urls_from_submissions(cik_padded, ticker):
    """
    Requests the Submissions API and extracts metadata ONLY for 10-Q reports 
    to construct raw .txt URLs.
    """
    # Use the PADDED CIK for the Submissions API request URL
    url = f"{SUBMISSIONS_API_BASE}{cik_padded}.json"
    
    print(f"   Processing {ticker} (CIK: {cik_padded})...")
    time.sleep(0.5) # CRITICAL: Respect SEC rate limits

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"      Failed to fetch submissions for {ticker}. Error: {e}")
        return []

    filing_list = []
    
    if 'filings' in data and 'recent' in data['filings'] and 'accessionNumber' in data['filings']['recent']:
        recent_filings = data['filings']['recent']
        
        for i in range(len(recent_filings['accessionNumber'])):
            acc_num = recent_filings['accessionNumber'][i]
            form = recent_filings['form'][i]
            
            # Filter: Only proceed if the form is a 10-Q or 10-Q/A
            if form not in TARGET_FORMS:
                continue
                
            # For constructing the /edgar/data/ archive URL, we use the UNPADDED CIK
            cik_unpadded = cik_padded.lstrip('0') 
            base_path = acc_num.replace('-', '')
            
            # Construct the raw text URL 
            raw_txt_url = (
                f"{BASE_SEC_URL}/Archives/edgar/data/{cik_unpadded}/"
                f"{base_path}/{acc_num}.txt"
            )
            
            filing_list.append({
                "ticker": ticker,
                "cik": cik_padded,
                "accession_number": acc_num,
                "date": recent_filings['filingDate'][i],
                "form": form,
                "url": raw_txt_url
            })
            
    print(f"      Found {len(filing_list)} 10-Q reports.")
    return filing_list

# --- Function 3: Main Execution and Saving ---
def main():
    
    # Step 1: Get the list of CIKs
    ciks_to_process = get_top_ciks(MAX_CIKS_TO_PROCESS)
    
    if not ciks_to_process:
        return

    all_urls = []
    print("\n2. Starting bulk 10-Q URL collection...")
    
    # Step 2: Iterate and collect ONLY 10-Q filing URLs
    for cik, ticker in ciks_to_process.items():
        filings = get_10q_urls_from_submissions(cik, ticker) 
        all_urls.extend(filings)

    # Step 3: Save the URLs to a text file
    filename = "top_50_cik_10q_raw_text_urls.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            # Write the header row (comma-separated)
            f.write("Ticker,CIK,Form Type,Filing Date,Accession Number,URL\n")
            
            # Sort the output by Filing Date (most recent first)
            all_urls.sort(key=lambda x: x['date'], reverse=True)
            
            for entry in all_urls:
                # Write the data row (comma-separated)
                line = (
                    f"{entry['ticker']},{entry['cik']},{entry['form']},"
                    f"{entry['date']},{entry['accession_number']},{entry['url']}"
                )
                f.write(line + '\n')

        print(f"\nBulk Extraction Complete!")
        print(f"A total of {len(all_urls)} 10-Q URLs saved to {filename}.")

    except IOError as e:
        print(f"Failed to write to file: {e}")

if __name__ == "__main__":
    main()