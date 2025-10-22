import requests
import zipfile
import os
import argparse

def download_and_unzip_dataset(file_id: str, output_zip: str, extract_dir: str):
    """
    Download a public dataset from Figshare by file_id and unzip it.


    Args:
        file_id (str): The Figshare file ID (e.g., 57518986).
        output_zip (str): Path to save the downloaded zip file.
        extract_dir (str): Directory to extract the contents.
    """
    url = f"https://figshare.com/ndownloader/files/{file_id}"
    
    print(f"Downloading dataset from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    
    with open(output_zip, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded: {output_zip}")
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    # Unzip contents
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to: {extract_dir}")

if __name__ == "__main__":
    file_id = "57518986"  # <-- Figshare file ID for your dataset
    output_zip = "EEG_Freewill_Reaching_Grasping.zip"
    extract_dir = "EEG_Dataset"

    parser = argparse.ArgumentParser(description="Download and unzip EEG dataset from Figshare")
    parser.add_argument("--file_id", type=str, default=file_id, help="Figshare file ID")
    parser.add_argument("--output_zip", type=str, default=output_zip, help="Output zip file path")
    parser.add_argument("--extract_dir", type=str, default=extract_dir, help="Directory to extract contents")

    args = parser.parse_args()

    download_and_unzip_dataset(args.file_id, args.output_zip, args.extract_dir)