import cv2
import os
import numpy as np
import requests
import py7zr
import shutil
import time
from typing import List, Tuple  # Retained for clarity, though maybe not strictly needed now

# --- Import your FaceDetector ---
try:
    from face import FaceDetector

    print("Successfully imported FaceDetector from face_detector.py")
except ImportError:
    print("ERROR: Could not import FaceDetector from face_detector.py.")
    print(
        "Please ensure 'face_detector.py' containing the FaceDetector class exists in the same directory or Python path.")
    exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during import: {e}")
    exit(1)

# --- Configuration (Simplified - No LBF Model Paths) ---

# 1. Dataset Download & Extraction Configuration
DATASET_URL_DROPBOX = "https://www.dropbox.com/s/hgixfsj2ea8qwrq/helenstar_release.7z?dl=1"

DATA_DIR = "./dataset"  # Main directory for all data
ARCHIVE_FILENAME = "helenstar_release.7z"
ARCHIVE_PATH = os.path.join(DATA_DIR, ARCHIVE_FILENAME)

# Assuming .7z extracts into a 'helenstar_release' folder inside DATA_DIR
EXTRACTED_DATA_BASE_DIR = DATA_DIR
MIDDLE_DIR_NAME = 'helenstar_release'
MIDDLE_DIR_PATH = os.path.join(EXTRACTED_DATA_BASE_DIR, MIDDLE_DIR_NAME)

# 2. Processing Configuration
input_label_dir = os.path.join(MIDDLE_DIR_PATH, 'train')
input_image_dir = os.path.join(MIDDLE_DIR_PATH, 'train')
output_binary_mask_dir = os.path.join(EXTRACTED_DATA_BASE_DIR, 'threshold_train_single_face')

# 3. Label value for facial skin
SKIN_LABEL_VALUE = 1

# 4. Other settings
CLEANUP_ARCHIVE = True


# --- End Configuration ---


# --- Helper Functions (Download, Extract - Unchanged) ---
def download_file(url, save_path, file_description="file", chunk_size=8192):
    """Downloads a file from a URL, showing progress."""
    try:
        print(f"Attempting to download {file_description} from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving to: {save_path}")
        with open(save_path, 'wb') as f:
            downloaded_size = 0;
            start_time = time.time()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk);
                    downloaded_size += len(chunk)
                    percent = (downloaded_size / total_size * 100) if total_size > 0 else 0
                    elapsed_time = time.time() - start_time
                    speed = (downloaded_size / elapsed_time / 1024 ** 2) if elapsed_time > 0 else 0
                    print(
                        f"\rDownloaded {downloaded_size / 1024 ** 2:.2f} / {total_size / 1024 ** 2:.2f} MB ({percent:.1f}%) at {speed:.2f} MB/s",
                        end="")
        print(f"\n{file_description} download complete.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {file_description}: {e}")
        if os.path.exists(save_path): os.remove(save_path)
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during {file_description} download: {e}")
        if os.path.exists(save_path): os.remove(save_path)
        return False


def extract_7z_archive(archive_path, extract_path):
    """Extracts a .7z archive using py7zr."""
    print(f"Attempting to extract archive: {archive_path}")
    if not os.path.exists(archive_path): print(f"Error: Archive file not found at {archive_path}"); return False
    try:
        os.makedirs(extract_path, exist_ok=True)
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            print(f"Extracting contents to: {extract_path}")
            z.extractall(path=extract_path)
        print("Archive extraction complete.")
        if not os.path.isdir(os.path.join(extract_path, MIDDLE_DIR_NAME)):
            print(
                f"Warning: Expected directory '{MIDDLE_DIR_NAME}' not found directly in {extract_path} after extraction.")
            print(f"Contents of {extract_path}: {os.listdir(extract_path)}")
        return True
    except py7zr.exceptions.Bad7zFile:
        print(f"Error: {archive_path} is not a valid .7z file or is corrupted."); return False
    except Exception as e:
        print(f"An error occurred during extraction: {e}"); return False


# --- Simplified Processing Function (Assumes FaceDetector is ready) ---
def create_binary_skin_masks_single_face(label_dir, image_dir, output_dir, skin_value):
    """
    Reads labels, detects faces using the pre-configured FaceDetector, creates binary
    skin masks ONLY for images with exactly one face, and saves them.
    """
    print("-" * 20)
    print(f"Starting single-face mask processing...")

    # --- Initialize Imported Face Detector (Assuming it finds its resources) ---
    try:
        detector = FaceDetector()
        print("Imported Face detector initialized successfully (assuming resources are accessible).")
    except cv2.error as cv_err:
        print(f"ERROR: OpenCV error during FaceDetector initialization: {cv_err}")
        print("This likely means 'lbfmodel.yaml' was not found where FaceDetector expects it,")
        print("or 'opencv-contrib-python' is not installed correctly.")
        return False
    except FileNotFoundError as fnf_err:
        print(f"ERROR: File not found during FaceDetector initialization: {fnf_err}")
        print("Ensure 'haarcascade_frontalface_default.xml' and 'lbfmodel.yaml' are accessible.")
        return False
    except Exception as e:
        print(f"Error initializing imported Face Detector: {e}")
        return False

        # --- Check Directories ---
    if not os.path.isdir(label_dir): print(f"Error: Input label directory not found: {label_dir}"); return False
    if not os.path.isdir(image_dir): print(f"Error: Input image directory not found: {image_dir}"); return False

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing labels from: {label_dir}")
    print(f"Looking for images in: {image_dir}")
    print(f"Saving filtered binary masks to: {output_dir}")

    # --- Process Files ---
    count_processed = 0
    count_skipped_no_image = 0
    count_skipped_wrong_faces = 0
    count_error_processing = 0

    all_label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('_label.png')]
    total_labels = len(all_label_files)
    print(f"Found {total_labels} potential label files.")

    for i, label_filename in enumerate(all_label_files):
        image_filename = label_filename.replace('_label.png', '_image.jpg')
        label_path = os.path.join(label_dir, label_filename)
        image_path = os.path.join(image_dir, image_filename)
        output_filename = label_filename
        output_path = os.path.join(output_dir, output_filename)

        if (i + 1) % 100 == 0: print(f"Checking file {i + 1}/{total_labels}: {label_filename}")

        if not os.path.exists(image_path):
            count_skipped_no_image += 1
            continue

        try:
            # Use the detect method from your imported class
            _, face_rectangles, _, _ = detector.detect(image_path)
            num_faces = len(face_rectangles)
        except Exception as e:
            print("sex")
            print(f"Error detecting faces in {image_filename} using imported detector: {e}. Skipping.")
            count_error_processing += 1
            continue

        if num_faces == 1:
            try:
                label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label_image is None:
                    count_error_processing += 1
                    continue

                binary_mask = cv2.inRange(label_image, skin_value, skin_value)
                save_success = cv2.imwrite(output_path, binary_mask)

                if not save_success:
                    count_error_processing += 1
                    continue

                count_processed += 1

            except Exception as e:
                print(f"Error processing label/saving mask for {label_filename}: {e}")
                count_error_processing += 1
        else:
            count_skipped_wrong_faces += 1

    # --- Final Summary ---
    print(f"\n--- Processing Summary ---")
    print(f"Total label files found: {total_labels}")
    print(f"Successfully processed (1 face detected): {count_processed}")
    print(f"Skipped (corresponding image not found): {count_skipped_no_image}")
    print(f"Skipped (0 or >1 faces detected): {count_skipped_wrong_faces}")
    print(f"Skipped due to processing/detection errors: {count_error_processing}")
    print(f"Filtered binary skin masks saved in: {output_dir}")
    print("-" * 20)

    return True


# --- Main Execution ---
if __name__ == "__main__":
    print("--- HELEN* Automation: Download, Extract, Filter & Process (Using Pre-configured FaceDetector) ---")

    # --- Dataset Setup ---
    if os.path.isdir(output_binary_mask_dir) and len(os.listdir(output_binary_mask_dir)) > 0:
        print(f"Processed single-face masks already found in: {output_binary_mask_dir}")
        print("Skipping all steps.")
        print("--- Script Finished (Already Done) ---")
        exit(0)

    elif os.path.isdir(input_label_dir):
        print(f"Extracted dataset found in: {MIDDLE_DIR_PATH}")
        print("Skipping download and extraction.")
        if not create_binary_skin_masks_single_face(  # No longer pass model path
                input_label_dir, input_image_dir, output_binary_mask_dir, SKIN_LABEL_VALUE):
            print("\nProcessing failed.")
            exit(1)
        else:
            print("--- Script Finished Successfully ---")
            exit(0)

    else:
        print(f"Extracted data not found in {MIDDLE_DIR_PATH}.")

        # --- Download Step ---
        download_success = False
        if not os.path.exists(ARCHIVE_PATH):
            if download_file(DATASET_URL_DROPBOX, ARCHIVE_PATH, "Dataset Archive"):
                download_success = True
            else:
                print("Dataset download failed.")
        else:
            print(f"Dataset archive file already exists: {ARCHIVE_PATH}. Skipping download.")
            download_success = True

        # --- Extraction Step ---
        extraction_success = False
        if download_success:
            if extract_7z_archive(ARCHIVE_PATH, EXTRACTED_DATA_BASE_DIR):
                if os.path.isdir(MIDDLE_DIR_PATH):
                    extraction_success = True
                    if CLEANUP_ARCHIVE and os.path.exists(ARCHIVE_PATH):
                        try:
                            print(f"Cleaning up downloaded archive: {ARCHIVE_PATH}")
                            os.remove(ARCHIVE_PATH)
                        except OSError as e:
                            print(f"Warning: Could not remove archive file: {e}")
                else:
                    print(
                        f"Error: Expected directory '{MIDDLE_DIR_NAME}' not found in {EXTRACTED_DATA_BASE_DIR} after extraction.")
                    extraction_success = False
            else:
                print("Extraction failed.")
        else:
            print("Download failed or skipped, cannot proceed to extraction.")

        # --- Processing Step ---
        if extraction_success:
            if not create_binary_skin_masks_single_face(  # No longer pass model path
                    input_label_dir, input_image_dir, output_binary_mask_dir, SKIN_LABEL_VALUE):
                print("\nProcessing failed after successful extraction.")
                exit(1)
        else:
            print("\nExtraction failed or skipped, cannot proceed to processing.")
            exit(1)

    print("--- Script Finished Successfully ---")