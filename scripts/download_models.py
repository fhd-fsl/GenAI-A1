import os
import urllib.request
import zipfile
import sys

# Replace this with the actual direct download link to your models (e.g., Google Drive / Dropbox direct link)
MODELS_URL = "https://drive.google.com/file/d/1DDcpX0xdP4t2WyXMxvs6g-3tt-Y7kcAJ/view?usp=sharing"
DOWNLOAD_DEST = "models.zip"
EXTRACT_DIR = "checkpoints/onnx"

import subprocess

def download_file(url, dest):
    print(f"Downloading models from {url}...")
    try:
        # We use gdown because standard urllib fails on large Google Drive files due to the virus scan warning
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        gdown.download(url, dest, quiet=False)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download models: {e}")
        print("Please ensure the MODELS_URL in scripts/download_models.py is set to a valid direct download link.")
        sys.exit(1)

def extract_zip(file_path, extract_to):
    print(f"Extracting to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

def main():
    if MODELS_URL == "https://example.com/path/to/your/models.zip":
        print("WARNING: MODELS_URL is set to a placeholder.")
        print("Please edit scripts/download_models.py and provide the actual URL to your ONNX models zip file.")
        sys.exit(1)

    # Download
    download_file(MODELS_URL, DOWNLOAD_DEST)
    
    # Extract
    extract_zip(DOWNLOAD_DEST, EXTRACT_DIR)
    
    # Clean up zip
    if os.path.exists(DOWNLOAD_DEST):
        os.remove(DOWNLOAD_DEST)
        print("Cleaned up zip file.")

    print(f"Models successfully downloaded and placed in {EXTRACT_DIR}")

if __name__ == "__main__":
    main()
