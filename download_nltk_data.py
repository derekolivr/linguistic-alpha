# download_nltk_data.py
import nltk
import os
import ssl

def download_nltk_packages():
    """
    Downloads required NLTK data packages to a local 'nltk_data' folder.
    This makes the project self-contained.
    """
    # --- Create a local directory for NLTK data ---
    DOWNLOAD_DIR = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # --- Tell NLTK to use this local path ---
    if DOWNLOAD_DIR not in nltk.data.path:
        nltk.data.path.append(DOWNLOAD_DIR)
    
    print(f"--- NLTK Downloader ---")
    print(f"Target directory: {DOWNLOAD_DIR}")

    # --- This is a workaround for a common SSL certificate error ---
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # --- List of required packages ---
    packages = ['punkt', 'vader_lexicon', 'averaged_perceptron_tagger']
    
    for package in packages:
        try:
            print(f"\nChecking for package: '{package}'...")
            # A more reliable way to check if data is available
            nltk.data.find(f'tokenizers/{package}.zip')
            print(f" -> '{package}' is already downloaded and available.")
        except LookupError:
            print(f" -> Package '{package}' not found. Downloading...")
            nltk.download(package, download_dir=DOWNLOAD_DIR)

    print("\n[SUCCESS] All required NLTK data packages are ready in the local directory.")

if __name__ == "__main__":
    download_nltk_packages()