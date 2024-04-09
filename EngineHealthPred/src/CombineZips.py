from zipfile import ZipFile
import os

# Replace these with your actual file paths
zip_files = ['C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_test_FD001_all.zip', 'C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_test_FD002_all.zip','C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_test_FD003_all.zip','C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_test_FD004_all.zip']
# zip_files = ['C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_FD001_3.zip','C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_FD003_3.zip']
# Temporary directory to extract files
extracted_dir = 'C:\\Users\\ssen\\Documents\\FlexKI\\processed4'

# Create the directory if it doesn't exist
os.makedirs(extracted_dir, exist_ok=True)

# Extract all files from the zip files
for zip_file in zip_files:
    with ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)

# Path for the final combined zip file
final_zip_path = 'C:\\Users\\ssen\\Documents\\RUL\\RULprediction\\processed_data_final_test_all_1.zip'

# Create a new zip file and add all extracted files
with ZipFile(final_zip_path, 'w') as zipf:
    for folder, subfolders, files in os.walk(extracted_dir):
        for file in files:
            file_path = os.path.join(folder, file)
            zipf.write(file_path, os.path.relpath(file_path, extracted_dir))

# Clean up the extracted files (optional)
os.rmdir(extracted_dir)
