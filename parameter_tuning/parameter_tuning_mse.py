import cv2
from pathlib import Path
import numpy as np
import time
import functools
import optuna
import tqdm

from src.face import FaceDetector
from src.mask import MaskHandler

DATA_DIR = Path("../dataset")
MIDDLE_DIR_NAME = 'helenstar_release'
MIDDLE_DIR_PATH = DATA_DIR / MIDDLE_DIR_NAME
INPUT_IMAGE_DIR = MIDDLE_DIR_PATH / 'train'
GROUND_TRUTH_MASK_DIR = DATA_DIR / 'threshold_training'
LOG_SUBDIR = Path("../optuna_logs")
LOG_FILENAME_BASE_MSE = "optuna_log_mse"

STUDY_NAME = "mse_tuning_study"
N_TRIALS = 100000
TRIALS_PER_LOG_FILE = 1000
LOG_INTERVAL = 50


def calculate_mse(pred_mask, gt_mask):
    if pred_mask is None or gt_mask is None:
        return 1.0

    if pred_mask.shape != gt_mask.shape:
        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    pred_norm = pred_mask.astype(np.float32) / 255.0
    gt_norm = gt_mask.astype(np.float32) / 255.0
    return np.mean((pred_norm - gt_norm) ** 2)


def log_progress_callback(study, frozen_trial, log_file_base):
    trial_number = frozen_trial.number
    if trial_number == 0 or (trial_number + 1) % LOG_INTERVAL == 0:
        file_index = trial_number // TRIALS_PER_LOG_FILE
        log_file_path = LOG_SUBDIR / f"{log_file_base}_{file_index:04d}.txt"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        value = frozen_trial.value
        params = frozen_trial.params

        try:
            best_value_str = f"{study.best_value:.6f}"
        except ValueError:
            best_value_str = "N/A"

        log_entry = (
            f"{timestamp} - trial {trial_number + 1} finished. "
            f"value: {value:.6f}. params: {params}. "
            f"best value so far: {best_value_str}\n"
        )

        LOG_SUBDIR.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, 'a') as f:
            f.write(log_entry)


def preprocess_data():
    fd = FaceDetector()
    mh = MaskHandler()

    image_basenames = []
    for f in GROUND_TRUTH_MASK_DIR.iterdir():
        if f.name.endswith('_label.png') and f.is_file():
            image_basenames.append(f.name.replace('_label.png', ''))

    print(f"found {len(image_basenames)} GT masks.")
    if not image_basenames:
        raise ValueError("no ground truth masks found")

    precalculated_data = {}
    successful_precalc = 0

    for basename in tqdm.tqdm(image_basenames):
        image_path = INPUT_IMAGE_DIR / f"{basename}_image.jpg"
        gt_mask_path = GROUND_TRUTH_MASK_DIR / f"{basename}_label.png"

        if not image_path.exists():
            continue

        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue

        _, _, _, landmarks = fd.detect(str(image_path))
        if not landmarks:
            continue

        _, selected_pts = mh.get_mask_points(str(image_path), landmarks)
        if not selected_pts or not selected_pts[0]:
            continue

        masks, _ = mh.build_masks(str(image_path), selected_pts)
        if not masks:
            continue

        intensity_histograms = mh.get_intensity_histograms(str(image_path), masks)
        if not intensity_histograms:
            continue

        prior_face_masks, _, _ = mh.get_prior_face_masks(str(image_path), selected_pts)
        if not prior_face_masks:
            continue

        precalculated_data[basename] = {
            "image_path": str(image_path),
            "intensity_histogram": intensity_histograms[0],
            "prior_mask": prior_face_masks[0],
            "gt_mask": gt_mask
        }
        successful_precalc += 1

    print(f"pre-calc done. usable images: {successful_precalc} / {len(image_basenames)}")
    return precalculated_data


def objective_mse(trial, precalculated_data, mask_handler):
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    lambda_ = trial.suggest_float("lambda_", 1e-6, 1e1, log=True)

    total_mse = 0.0
    processed_count = 0

    for basename, data in precalculated_data.items():
        try:
            skin_pixel_maps_list = mask_handler.get_skin_pixel_maps(
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

            mse = calculate_mse(predicted_mask, data["gt_mask"])
            total_mse += mse
            processed_count += 1
        except Exception:
            continue

    return 1.0 if processed_count == 0 else total_mse / processed_count


def main():
    print("--- hyper parameter tuning ---")

    LOG_SUBDIR.mkdir(parents=True, exist_ok=True)
    print(f"ensured log directory exists: {LOG_SUBDIR}")

    try:
        precalculated_data = preprocess_data()
        if not precalculated_data:
            print("error: No data pre-calculated.")
            return

        mask_handler = MaskHandler()

        objective = functools.partial(objective_mse,
                                      precalculated_data=precalculated_data,
                                      mask_handler=mask_handler)

        print(f"\nstarting: {STUDY_NAME} with {N_TRIALS} trials...")
        study = optuna.create_study(study_name=STUDY_NAME,
                                    direction="minimize",
                                    load_if_exists=True)

        mse_log_callback = functools.partial(log_progress_callback,
                                             log_file_base=LOG_FILENAME_BASE_MSE)

        study.optimize(objective,
                       n_trials=N_TRIALS,
                       n_jobs=1,
                       callbacks=[mse_log_callback])

        print("\noptimization finished")
        print(f"number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        print(f"best trial no.: {best_trial.number}")
        print(f"best trial value (average MSE): {best_trial.value:.8f}")
        print("best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"\t{key}: {value:.8f}")

    except Exception as e:
        print(f"error during optimization: {e}")


if __name__ == '__main__':
    main()