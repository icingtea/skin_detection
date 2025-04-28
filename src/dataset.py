import cv2
import os
import requests
import py7zr
from pathlib import Path
import time
import random
from face import FaceDetector
from typing import Optional


class Dataset:
    DATASET_URL_DROPBOX = (
        "https://www.dropbox.com/s/hgixfsj2ea8qwrq/helenstar_release.7z?dl=1"
    )
    ARCHIVE_FILENAME = "helenstar_release.7z"
    MIDDLE_DIR_NAME = "helenstar_release"
    SKIN_LABEL_VALUE = 1
    NOSE_LABEL_VALUE = 6

    def __init__(self, data_dir="./dataset", cleanup_archive=True):
        self.detector = FaceDetector(Path("assets/lbfmodel.yaml"))
        self.data_dir = data_dir
        self.cleanup_archive = cleanup_archive
        self.middle_dir_path = os.path.join(self.data_dir, self.MIDDLE_DIR_NAME)
        self.archive_path = os.path.join(self.data_dir, self.ARCHIVE_FILENAME)
        self.input_label_dir = os.path.join(self.middle_dir_path, "train")
        self.input_image_dir = os.path.join(self.middle_dir_path, "train")
        self.output_binary_mask_dir = os.path.join(self.data_dir, "threshold_training")

        self.run()

    def download_file(self, url, save_path, file_description="file", chunk_size=8192):
        print(f"Downloading {file_description} from: {url}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(save_path, "wb") as f:
            downloaded_size = 0
            start_time = time.time()
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    percent = (
                        (downloaded_size / total_size * 100) if total_size > 0 else 0
                    )
                    elapsed_time = time.time() - start_time
                    speed = (
                        (downloaded_size / elapsed_time / 1024**2)
                        if elapsed_time > 0
                        else 0
                    )
                    print(
                        f"\rdownload {downloaded_size / 1024 ** 2:.2f} / {total_size / 1024 ** 2:.2f} MB ({percent:.1f}%) at {speed:.2f} MB/s",
                        end="",
                    )

        print(f"\n{file_description} download complete.")
        return True

    def extract_7z_archive(self, archive_path, extract_path):
        print(f"Extracting archive: {archive_path}")
        os.makedirs(extract_path, exist_ok=True)

        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=extract_path)

        print("Successfully extracted the archive.")
        return True

    def create_combined_skin_nose_mask_single_face(self):
        print("-" * 20)
        print(f"Creating the masks...")

        os.makedirs(self.output_binary_mask_dir, exist_ok=True)

        count_processed = 0
        count_skipped_no_image = 0
        count_skipped_wrong_faces = 0
        count_error_processing = 0

        all_label_files = [
            f
            for f in os.listdir(self.input_label_dir)
            if f.lower().endswith("_label.png")
        ]
        total_labels = len(all_label_files)
        print(f"Found {total_labels} potential label files.")

        for i, label_filename in enumerate(all_label_files):
            image_filename = label_filename.replace("_label.png", "_image.jpg")
            label_path = os.path.join(self.input_label_dir, label_filename)
            image_path = os.path.join(self.input_image_dir, image_filename)
            output_path = os.path.join(self.output_binary_mask_dir, label_filename)

            if (i + 1) % 100 == 0:
                print(f"Processing {i + 1}/{total_labels}: {label_filename}")

            if not os.path.exists(image_path):
                count_skipped_no_image += 1
                continue

            try:
                _, face_rectangles, _, _ = self.detector.detect(image_path)

                if len(face_rectangles) == 1:
                    label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    if label_image is None:
                        count_error_processing += 1
                        continue

                    skin_mask = cv2.inRange(
                        label_image, self.SKIN_LABEL_VALUE, self.SKIN_LABEL_VALUE
                    )
                    nose_mask = cv2.inRange(
                        label_image, self.NOSE_LABEL_VALUE, self.NOSE_LABEL_VALUE
                    )
                    combined_mask = cv2.bitwise_or(skin_mask, nose_mask)

                    cv2.imwrite(output_path, combined_mask)
                    count_processed += 1
                else:
                    count_skipped_wrong_faces += 1

            except Exception as e:
                print(f"Error processing {label_filename}: {e}")
                count_error_processing += 1

        print(f"\nDone processing the file")
        print(f"Total label files found: {total_labels}")
        print(f"Successfully processed: {count_processed}")
        print(f"Skipped (image not found): {count_skipped_no_image}")
        print(f"Skipped (0 or >1 faces): {count_skipped_wrong_faces}")
        print(f"Skipped due to errors: {count_error_processing}")
        print("-" * 20)

        return True

    def run(self):
        print("--- Fetching and extracting the dataset ---")

        if (
            os.path.isdir(self.output_binary_mask_dir)
            and len(os.listdir(self.output_binary_mask_dir)) > 0
        ):
            print(f"Processed masks already exist at: {self.output_binary_mask_dir}")
            print("Skipping all processing.")
            return True

        if os.path.isdir(self.middle_dir_path):
            print(f"Dataset already extracted at: {self.middle_dir_path}")
        else:
            if not os.path.exists(self.archive_path):
                print(f"Archive not found at {self.archive_path}. Downloading...")
                self.download_file(
                    self.DATASET_URL_DROPBOX, self.archive_path, "Dataset Archive"
                )
            else:
                print(
                    f"Archive already exists at: {self.archive_path}. Skipping download."
                )

            self.extract_7z_archive(self.archive_path, self.data_dir)

            if self.cleanup_archive and os.path.exists(self.archive_path):
                os.remove(self.archive_path)
                print(f"Cleaned up archive at {self.archive_path}")

        self.create_combined_skin_nose_mask_single_face()

        print("--- Completed ---")
        return True

    def return_img_pair(self, name: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
       input_image_dir = Path(self.input_image_dir)
       output_binary_mask_dir = Path(self.output_binary_mask_dir)   

       image_files = [
           f for f in input_image_dir.iterdir() if f.suffix.lower() == "_image.jpg"
       ]    

       if name:
           img = name + "_image.jpg"
           mask = name + "_label.png"   
           if Path(img) in image_files:
               image_path = input_image_dir / img
               mask_path = output_binary_mask_dir / mask
           else:
               print("Image not found")
               return None, None
       else:
           random_img = random.choice(image_files)
           random_mask = random_img.stem.replace("_image", "_label") + random_img.suffix
           image_path = input_image_dir / random_img
           mask_path = output_binary_mask_dir / random_mask 

       if image_path.exists() and mask_path.exists():
           return image_path, mask_path
       else:
           if not image_path.exists():
               print(f"Image '{image_path}' not found")
           if not mask_path.exists():
               print(f"Mask '{mask_path}' not found")
           return None, None