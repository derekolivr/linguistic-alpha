import requests
import time
import re
import os
import csv
import pandas as pd

# --- Configuration and Constants ---
INPUT_FILENAME = "top_50_cik_10q_raw_text_urls.txt" 
OUTPUT_FILENAME = "final_html_urls.txt" # Output in comma-separated TXT format
HEADERS = {"User-Agent": "FinalURLExtractor your.email@example.com"} 
MAX_FILES_TO_PROCESS = 10 # <-- LIMIT SET TO 10 FILES

# Regex pattern to find the value inside the <FILENAME> tag
FILENAME_PATTERN = re.compile(r'<FILENAME>\s*(\S+)', re.IGNORECASE)

# --- Function 1: Download and Extract Filename ---
def extract_filename_tag_value(url):
    """Downloads the raw filing text and extracts the primary filename."""
    time.sleep(0.5) 

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        raw_text = response.text
    except Exception as e:
        return {"Status": "Download Failed", "Extracted_Value": str(e)}

    match = FILENAME_PATTERN.search(raw_text)

    if match:
        extracted_value = match.group(1).strip()
        return {"Status": "Success", "Extracted_Value": extracted_value}
    else:
        return {"Status": "Warning", "Extracted_Value": "Tag not found"}

# --- Function 2: Main Execution ---
def main():
    
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ Error: Input file '{INPUT_FILENAME}' not found. Cannot proceed.")
        return

    print(f"1. Reading URLs from {INPUT_FILENAME} and building final links (Limit: {MAX_FILES_TO_PROCESS})...")
    
    all_filing_records = []
    files_processed = 0
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Read and store the original header
            
            for row in reader:
                if files_processed >= MAX_FILES_TO_PROCESS:
                    print(f"\n🛑 Reached limit of {MAX_FILES_TO_PROCESS} files. Stopping processing.")
                    break
                    
                if len(row) < 6: continue
                
                ticker, cik, form_type, filing_date, acc_num, raw_txt_url = row
                
                # A. Extract the filename
                result = extract_filename_tag_value(raw_txt_url)
                extracted_filename = result["Extracted_Value"]
                
                # B. Extract the Base Path (directory)
                base_path = raw_txt_url.rsplit('/', 1)[0] + '/'
                
                final_output_url = None
                
                if result["Status"] == "Success":
                    # C. Construct the Final HTML URL
                    final_output_url = os.path.join(base_path, extracted_filename).replace('\\', '/')
                else:
                    # If download or extraction failed, use the error message as the URL field value
                    final_output_url = f"[ERROR: {extracted_filename}]" 

                # D. Store the record
                all_filing_records.append([
                    ticker,
                    cik,
                    form_type,
                    filing_date,
                    acc_num,
                    final_output_url # The new URL
                ])
                
                print(f"   -> Processed file {files_processed + 1}/{MAX_FILES_TO_PROCESS}: {ticker} ({result['Status']})")
                files_processed += 1 # Increment the counter

    except Exception as e:
        print(f"\n❌ A critical error occurred during processing: {e}")
        return

    # 2. Save the results to the new TXT file (comma-separated)
    if not all_filing_records:
        print("\n🛑 No records were successfully processed.")
        return
        
    print(f"\n2. Saving {len(all_filing_records)} records to {OUTPUT_FILENAME}...")
    
    try:
        # Use the csv writer to ensure proper comma separation in the TXT file
        with open(OUTPUT_FILENAME, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write the header
            writer.writerow(header)
            
            # Write all the processed rows
            writer.writerows(all_filing_records)
        
        print("\n🎉 Success! The final URLs have been generated.")
        print(f"   The new file, **{OUTPUT_FILENAME}** (comma-separated), is ready.")

        # Print the first row of the output for confirmation
        print("\n--- Example Output Row ---")
        print(','.join(all_filing_records[0]))
        
    except IOError as e:
        print(f"\n❌ Error: Failed to write to file. Reason: {e}")

    return

if __name__ == "__main__":
    main()