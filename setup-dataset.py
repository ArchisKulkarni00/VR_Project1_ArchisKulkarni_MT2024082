import os
import subprocess
import shutil

# GitHub Dataset Details
GITHUB_REPO = "https://github.com/chandrikadeb7/Face-Mask-Detection.git"
GITHUB_DATASET_PATH = "dataset"  # The folder we need
LOCAL_GITHUB_PATH = "FaceMaskDataset"

# Google Drive Dataset Details
GOOGLE_DRIVE_FILE_ID = "1KycQj4dik91RuBGvbhDJou7YDQEKAH2Z"  # Replace with actual ID
LOCAL_GDRIVE_FILE = os.path.join(LOCAL_GITHUB_PATH, "MaskedFaceSegmentation.zip")

# Step 1: Clone only the dataset folder from GitHub
def download_github_dataset():
    if not os.path.exists(LOCAL_GITHUB_PATH):
        print("[INFO] Downloading GitHub dataset...")
        subprocess.run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", GITHUB_REPO, LOCAL_GITHUB_PATH
        ])
        subprocess.run(["git", "-C", LOCAL_GITHUB_PATH, "sparse-checkout", "set", GITHUB_DATASET_PATH])
        print("[INFO] GitHub dataset downloaded successfully.")
    else:
        print("[INFO] GitHub dataset already exists. Skipping download.")

# Step 3: Download the Google Drive dataset ZIP inside the dataset folder
def download_google_drive_dataset():
    if not os.path.exists(LOCAL_GDRIVE_FILE):
        print("[INFO] Downloading Google Drive dataset into the dataset folder...")
        try:
            import gdown
            gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", LOCAL_GDRIVE_FILE, quiet=False)
            print("[INFO] Google Drive dataset downloaded successfully.")
        except ImportError:
            print("[ERROR] gdown is not installed. Install it using `pip install gdown`.")
    else:
        print("[INFO] Google Drive dataset already exists. Skipping download.")

# User menu for dataset selection
print("Select an option:")
print("1. Download only GitHub dataset")
print("2. Download only Google Drive dataset")
print("3. Download both datasets")
choice = input("Enter your choice (1/2/3): ")

if choice == "1":
    download_github_dataset()
elif choice == "2":
    download_google_drive_dataset()
elif choice == "3":
    download_github_dataset()
    download_google_drive_dataset()
else:
    print("[ERROR] Invalid choice. Exiting.")

print("[INFO] Dataset setup complete. Ready to proceed!")
print("[INFO] Check the /FaceMaskDataset folder. dataset folder contains first dataset and zip file contains second dataset.")
