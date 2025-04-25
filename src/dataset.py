import cv2
import os
import requests
import py7zr
import time
from src.face import FaceDetector

DATASET_URL_DROPBOX = "https://www.dropbox.com/s/hgixfsj2ea8qwrq/helenstar_release.7z?dl=1"
DATA_DIR = "./dataset"
ARCHIVE_FILENAME = "helenstar_release.7z"
ARCHIVE_PATH = os.path.join(DATA_DIR, ARCHIVE_FILENAME)
EXTRACTED_DATA_BASE_DIR = DATA_DIR
MIDDLE_DIR_NAME = 'helenstar_release'
MIDDLE_DIR_PATH = os.path.join(EXTRACTED_DATA_BASE_DIR, MIDDLE_DIR_NAME)

input_label_dir = os.path.join(MIDDLE_DIR_PATH, 'train')
input_image_dir = os.path.join(MIDDLE_DIR_PATH, 'train')
output_binary_mask_dir = os.path.join(EXTRACTED_DATA_BASE_DIR, 'threshold_training')

SKIN_LABEL_VALUE = 1
NOSE_LABEL_VALUE = 6
CLEANUP_ARCHIVE = True


def download_file(url, save_path, file_description="file", chunk_size=8192):
    print(f"Downloading {file_description} from: {url}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as f:
        downloaded_size = 0
        start_time = time.time()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                percent = (downloaded_size / total_size * 100) if total_size > 0 else 0
                elapsed_time = time.time() - start_time
                speed = (downloaded_size / elapsed_time / 1024 ** 2) if elapsed_time > 0 else 0
                print(
                    f"\rdownload {downloaded_size / 1024 ** 2:.2f} / {total_size / 1024 ** 2:.2f} MB ({percent:.1f}%) at {speed:.2f} MB/s",
                    end="")

    print(f"\n{file_description} download complete.")
    return True


def extract_7z_archive(archive_path, extract_path):
    print(f"extracting archive: {archive_path}")
    os.makedirs(extract_path, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=extract_path)

    print("successfully extracted the archive")
    return True


def create_combined_skin_nose_mask_single_face(label_dir, image_dir, output_dir, skin_val, nose_val):
    """
    Creates binary masks where pixels are white (255) if they belong to
    either skin or nose in the original label, and black (0) otherwise.
    Only processes images with exactly one face.
    """
    print("-" * 20)
    print(f"creating the masks...")

    detector = FaceDetector()
    os.makedirs(output_dir, exist_ok=True)

    count_processed = 0
    count_skipped_no_image = 0
    count_skipped_wrong_faces = 0
    count_error_processing = 0

    all_label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('_label.png')]
    total_labels = len(all_label_files)
    print(f"found {total_labels} potential label files.")

    for i, label_filename in enumerate(all_label_files):
        image_filename = label_filename.replace('_label.png', '_image.jpg')
        label_path = os.path.join(label_dir, label_filename)
        image_path = os.path.join(image_dir, image_filename)
        output_path = os.path.join(output_dir, label_filename)

        if (i + 1) % 100 == 0:
            print(f"processing {i + 1}/{total_labels}: {label_filename}")

        if not os.path.exists(image_path):
            count_skipped_no_image += 1
            continue

        try:
            _, face_rectangles, _, _ = detector.detect(image_path)

            if len(face_rectangles) == 1:
                label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label_image is None:
                    count_error_processing += 1
                    continue

                skin_mask = cv2.inRange(label_image, skin_val, skin_val)
                nose_mask = cv2.inRange(label_image, nose_val, nose_val)
                combined_mask = cv2.bitwise_or(skin_mask, nose_mask)

                cv2.imwrite(output_path, combined_mask)
                count_processed += 1
            else:
                count_skipped_wrong_faces += 1

        except Exception as e:
            print(f"error processing {label_filename}: {e}")
            count_error_processing += 1

    print(f"\ndone processing the file")
    print(f"total label files found: {total_labels}")
    print(f"successfully processed: {count_processed}")
    print(f"skipped (image not found): {count_skipped_no_image}")
    print(f"skipped (0 or >1 faces): {count_skipped_wrong_faces}")
    print(f"skipped due to errors: {count_error_processing}")
    print("-" * 20)

    return True


if __name__ == "__main__":
    print("--- fetching and extracting the dataset ---")

    if os.path.isdir(output_binary_mask_dir) and len(os.listdir(output_binary_mask_dir)) > 0:
        print(f"processed masks found in: {output_binary_mask_dir}")
        print("skipping all steps - already done.")
        exit(0)

    elif os.path.isdir(input_label_dir):
        print(f"dataset found in: {MIDDLE_DIR_PATH}")
        create_combined_skin_nose_mask_single_face(
            input_label_dir, input_image_dir,
            output_binary_mask_dir,
            SKIN_LABEL_VALUE, NOSE_LABEL_VALUE
        )

    else:
        print(f"dataset not found. downloading and extracting...")

        if not os.path.exists(ARCHIVE_PATH):
            download_file(DATASET_URL_DROPBOX, ARCHIVE_PATH, "Dataset Archive")
        else:
            print(f"archive already exists: {ARCHIVE_PATH}. skipping download.")

        extract_7z_archive(ARCHIVE_PATH, EXTRACTED_DATA_BASE_DIR)

        if CLEANUP_ARCHIVE and os.path.exists(ARCHIVE_PATH):
            os.remove(ARCHIVE_PATH)
            print(f"cleaned up downloaded archive")

        create_combined_skin_nose_mask_single_face(
            input_label_dir, input_image_dir,
            output_binary_mask_dir,
            SKIN_LABEL_VALUE, NOSE_LABEL_VALUE
        )

    print("--- completed ---")