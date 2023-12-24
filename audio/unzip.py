import zipfile
import os
import shutil
import random

def extract_mp3_files(zip_file_path, output_directory, max_files_to_extract):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        extracted_count = 0

        # Get a list of all MP3 files in the zip file
        mp3_files = [file for file in zip_ref.namelist() if file.lower().endswith('.mp3')]
        
        # Shuffle the MP3 files randomly
        random.shuffle(mp3_files)

        for mp3_file in mp3_files:
            # Extract the MP3 file directly to the output directory without subdirectories
            zip_ref.extract(mp3_file, output_directory)
            
            # Get the base filename without subdirectory information
            mp3_base_name = os.path.basename(mp3_file)
            extracted_path = os.path.join(output_directory, mp3_base_name)

            # Rename the extracted file if a file with the same name already exists
            if os.path.exists(extracted_path):
                file_name, file_extension = os.path.splitext(mp3_base_name)
                mp3_base_name = f"{file_name}_{random.randint(1, 10000)}{file_extension}"
                extracted_path = os.path.join(output_directory, mp3_base_name)

            os.rename(os.path.join(output_directory, mp3_file), extracted_path)
            extracted_count += 1

            # Print the name of the extracted file
            print(f"Extracted: {mp3_base_name}")

            if extracted_count >= max_files_to_extract:
                return

# Usage:
zip_file_path = './fma_small.zip'
output_directory = './fma_small'
max_files_to_extract = 1000  # Set the number of mp3 files to extract

extract_mp3_files(zip_file_path, output_directory, max_files_to_extract)
