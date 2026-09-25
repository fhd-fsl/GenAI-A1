import os
import argparse
import urllib.request
import tarfile
import tqdm

class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(url: str, extract_to: str):
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)
    
    print(f"Extracting {filename}...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")

def download_oxford_pet(data_dir: str):
    """
    Downloads and extracts the Oxford-IIIT Pet dataset images and annotations.
    """
    print(f"Setting up Oxford-IIIT Pet dataset in {data_dir}...")
    
    # Official URLs
    images_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    annotations_url = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
    
    pet_dir = os.path.join(data_dir, "oxford-iiit-pet")
    
    download_and_extract(images_url, pet_dir)
    download_and_extract(annotations_url, pet_dir)
    
    print("Oxford-IIIT Pet dataset setup complete.")

def download_fs2k(data_dir: str):
    """
    Stub for FS2K dataset download. 
    Task 4 is deferred, so we will implement this custom download logic later.
    """
    print("FS2K dataset download is deferred until Task 4.")
    # Note: FS2K is not in torchvision. It typically requires downloading
    # from a provided Google Drive or GitHub link.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for GenAI Assignment 1")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--dataset", type=str, choices=["all", "pet", "fs2k"], default="all")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.dataset in ["all", "pet"]:
        download_oxford_pet(args.data_dir)
        
    if args.dataset in ["all", "fs2k"]:
        download_fs2k(args.data_dir)
