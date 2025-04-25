import os.path
from pathlib import Path

import cv2
import numpy as np

import optuna
import tqdm

from dataset import NOSE_LABEL_VALUE
from face import FaceDetector
from mask import MaskHandler

data_dir = Path("../dataset")
extracted_data_base_dir = data_dir
middle_dir_name = 'helenstar_release'
middle_dir_path = extracted_data_base_dir / middle_dir_name

input_image_dir = middle_dir_path / 'train'
input_label_dir = middle_dir_path / 'train'

ground_truth_mask_dir = data_dir / 'threshold_training'

SKIN_LABEL_VALUE = 1
NOSE_LABEL_VALUE = 6

study_name = "ts_pmo_gng_🥀🥀🥀"
n_trials = 100

def calculate_iou(pred_mask, gt_mask):
    pred_mask_bool = pred_mask > 128
    gt_mask_bool = gt_mask > 128

    intersection = np.logical_and(pred_mask_bool, gt_mask_bool)
    union = np.logical_or(pred_mask_bool, gt_mask_bool)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    if union_sum == 0:
        return 1.0 if intersection_sum == 0 else 0.0

    iou_score = intersection_sum / union_sum
    return iou_score

if __name__ == '__main__':
    fd = FaceDetector()
    mh = MaskHandler()
    print("loaded facehandler and maskhandler")

    precalculated_data = {}
    image_basenames = []

    for f in ground_truth_mask_dir.iterdir():
        if f.name.endswith('_label.png') and f.is_file():
            image_basenames.append(f.name.replace('_label.png', ''))

    print(f"found {len(image_basenames)} potential ground truth masks")



    print("pre calculating data for optimisation")
    for basename in tqdm.tqdm(image_basenames):
        image_path = input_image_dir / f"{basename}_image.jpg"
        label_path = input_label_dir / f"{basename}_label.png"
        gt_mask_path = ground_truth_mask_dir / f"{basename}_label.png"

        image_path_str = str(image_path)
        gt_mask_path_str = str(gt_mask_path)

        try:
            gt_mask = cv2.imread(gt_mask_path_str, cv2.IMREAD_GRAYSCALE)

            _, _, _, landmarks = fd.detect(image_path_str)
            _, selected_pts = mh.get_mask_points(image_path_str, landmarks)
            masks, _ = mh.build_masks(image_path_str, selected_pts)
            intensity_histograms = mh.get_intensity_histograms(image_path_str, masks)
            prior_face_masks, _, _ = mh.get_prior_face_masks(image_path_str, selected_pts)

            precalculated_data[basename] = {
                "image_path": image_path,
                "intensity_histogram": intensity_histograms[0],
                "prior_mask": prior_face_masks[0],
                "gt_mask": gt_mask
            }

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"error precalculating data for {basename}: {e}. skipping...")


    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.1, 0.9)
        lambda_ = trial.suggest_float("lambda_", 1e-4, 1, log=True)

        total_iou = 0.0
        processed_count = 0

        for basename, data in precalculated_data.items():
            try:
                skin_pixel_maps_list = mh.get_skin_pixel_maps(
                    data["image_path"],
                    [data["intensity_histogram"]],
                    [data["prior_mask"]],
                    alpha,
                    lambda_
                )

                if isinstance(skin_pixel_maps_list, list) and skin_pixel_maps_list:
                    predicted_mask = skin_pixel_maps_list[0]
                elif isinstance(skin_pixel_maps_list, np.ndarray):
                    predicted_mask = skin_pixel_maps_list
                else:
                    continue

                iou = calculate_iou(predicted_mask, data["gt_mask"])
                total_iou += iou
                processed_count += 1

            except Exception as e:
                continue

        average_iou = total_iou / processed_count
        return average_iou

    print(f"\nstarting: {study_name}")
    study = optuna.create_study(study_name=study_name, direction="maximize", load_if_exists=True)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, timeout=None)

    print("\noptimisation finished")

    print(f"number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print(f"best trial no.: {best_trial.number}")
        print(f"best trial value (average iou): {best_trial.value:.6f}")
        print("best hyperparams:")
        for key, value in best_trial.params.items():
            print(f"\t{key}: {value:.8f}")
    except ValueError:
        print("error: no successful trials completed")





