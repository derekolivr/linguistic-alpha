import requests
from bs4 import BeautifulSoup
import os
import re

def scrape_edgar_htm(url, output_dir="output"):
    """
    Scrapes the prose from a given SEC Edgar HTM link and saves it to a file.
    This version is designed to handle Inline XBRL by stripping out financial data tags.

    Args:
        url (str): The URL of the SEC Edgar filing.
        output_dir (str): The directory to save the output file.
    """
    try:
        # Per SEC developer guidelines, a custom User-Agent is required for programmatic access.
        headers = {
            "User-Agent": "LinguisticAlpha/1.0 (contact@example.com)"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  

        soup = BeautifulSoup(response.content, "html.parser")
        body = soup.find('body')
        if not body:
            print("No <body> tag found in the document.")
            return
        for tag in body.find_all(['table', 'script', 'style']):
            tag.decompose()
        for tag in body.find_all(lambda t: t.name and t.name.startswith('ix:')):
            tag.decompose()
        text = body.get_text(separator=' ', strip=True)

        text = re.sub(r'\s+', ' ', text).strip()
        # 2. Add newlines before "Item" headings to restore some document structure for readability.
        text = re.sub(r'(?i)(Item\s+\d+\w*\.)', r'\n\n\1', text)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the extracted text to a new file to show the improvement
        filename_base = url.split('/')[-1].replace('.htm', '')
        output_path = os.path.join(output_dir, f"{filename_base}_clean.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Successfully scraped and saved cleaned text to {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # The URL from the user prompt.
    filing_url = "https://www.sec.gov/Archives/edgar/data/1413329/000162828025046179/pm-20250930.htm"
    scrape_edgar_htm(filing_url)
