import os, json, subprocess, shutil, requests
from tqdm import tqdm
import py7zr


json_path = os.path.join(os.path.dirname(__file__), "data.json")

with open(file=json_path, mode='r') as f:
    datasets = json.load(f)
    
def download_file(url, folder):
    os.makedirs(name=folder, exist_ok=True)
    filename = url.split('/')[-1]
    file_path = os.path.join(folder, filename)
    
    if os.path.exists(file_path):
        print(f"Files already exists {filename}")
        return file_path
    
    print(f"Downloading {filename} --> {folder}")
    
    try:
        response = requests.get(url=url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with open(file_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"Downloaded: {filename}")
        return file_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None
    
def extract_7z(file_path, folder):
    print(f"Extracting {file_path}")
    try:
        with py7zr.SevenZipFile(file_path, mode="r") as archive:
            files = archive.getnames()
            totals = len(files) 
            
        with py7zr.SevenZipFile(file_path, mode='r') as archive, tqdm(
            total=totals, desc="Extracting", unit="file"
        ) as bar:
            archive.extractall(path=folder)
            bar.update(totals)

        print(f"Extraction completed: {file_path}")
            
    except Exception as e:
        print(f"Extraction failed {e}")

for dataset in datasets:
    dataset_name = dataset['name']
    dataset_path = dataset['path']
    links = dataset['links']
    
    if not links:
        print(f"No download links provided for {dataset_name}. Skipping...") 
        continue
    
    for url in links:
        file_path = download_file(url=url, folder=dataset_path)
        if file_path and file_path.endswith('.7z'):
            extract_7z(file_path=file_path, folder=dataset_path)
print("\nAll datasets have been downloaded !")