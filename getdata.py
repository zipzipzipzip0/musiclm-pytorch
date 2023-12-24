import os
import requests

def download_data(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length'))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'wb') as file:
        chunk_size = 1024
        downloaded = 0
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            downloaded += len(data)
            progress = (downloaded / total_size) * 100 if total_size else 0
            print(f"Download Progress: {progress:.2f}% ({downloaded}/{total_size} bytes)", end='\r', flush=True)

    print("\nDownload complete!")

fma_small_url = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
fma_small_path = './audio/fma_small.zip'
download_data(fma_small_url, fma_small_path)

hubert1_url = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt'
hubert1_path = './hubert/hubert_base_ls960.pt'
#download_data(hubert1_url, hubert1_path)

hubert2_url = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin'
hubert2_path = './hubert/hubert_base_ls960_L9_km500.bin'
#download_data(hubert2_url, hubert2_path)
