import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from matplotlib import pyplot as plt
from src.projectUtils import Utils


class MaskHandler:
    def get_mask_points(
        self, img_path: Path, landmarks: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Dict[str, List[Tuple[int, int]]]]]:
        """
        selects specific facial landmarks and highlights them on the image

        in:
            img_path `str`: path to the input image file
            landmarks `List[np.ndarray]`: list of 68-point landmarks for each detected face

        out:
            img_rgb `np.ndarray`: rgb image with selected landmark points highlighted
            selected_pts `List[Dict[str, List[Tuple[int, int]]]]`: list of selected (x, y) points per face, organized by region
        """

        img_read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_GRAY2RGB)

        idxs = {
            "left half": [17, 21, 30, 31, 48],
            "right half": [30, 22, 26, 54, 35],
            "nose": [21, 22, 30],
            "left eye": [36, 37, 38, 39, 40, 41],
            "right eye": [42, 43, 44, 45, 46, 47],
            "upper lip": [48, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64],
        }

        selected_pts = []

        for lm in landmarks:
            face_pts = {}
            for key, group in idxs.items():
                group_coordinates = [(int(lm[i][0]), int(lm[i][1])) for i in group]
                for pt in group_coordinates:
                    cv2.circle(img_rgb, pt, 3, (0, 255, 255), -1)
                face_pts[key] = group_coordinates
            selected_pts.append(face_pts)

        return img_rgb, selected_pts

    def build_masks(
        self, img_path: Path, mask_pts: List[Dict[str, List[Tuple[int, int]]]]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        builds masks for each face by filling selected facial regions (left half, right half, and upper lip)
        and excluding eye regions using convex hulls

        in:
            img_path `str`: path to the original grayscale image
            mask_pts `List[Dict[str, List[Tuple[int, int]]]]`: list of selected (x, y) coordinates per face to form convex hulls

        out:
            masks `List[np.ndarray]`: grayscale masks with selected regions filled (255) and omitted regions (0)
            masked_imgs `List[np.ndarray]`: grayscale images with the masks applied, keeping only the selected regions
        """

        img_read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        masks = []
        masked_imgs = []

        for face in mask_pts:
            mask = np.zeros(img_read.shape, dtype=np.uint8)
            for key, group in face.items():
                polygon = np.array(group, dtype=np.int32)
                if key in ["left half", "right half", "upper lip", "nose"]:
                    cv2.fillPoly(mask, [polygon], color=255)
                else:
                    hull = cv2.convexHull(polygon)
                    cv2.fillPoly(mask, [hull], color=0)

            masked_img = cv2.bitwise_and(img_read, mask)
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_GRAY2RGB)

            masks.append(mask)
            masked_imgs.append(masked_img)

        return masks, masked_imgs

    def get_intensity_histograms(
        self, img_path: Path, masks: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Get intensity histograms for each face mask

        in:
            img_path `str`: path to input image file
            mask_pts `List[np.ndarray]`: list of masks, one for each face.

        out:
            prob_maps `List[np.ndarray]`: Pixel intensity histograms per face
        """
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        prob_maps = []

        for mask in masks:

            face_pixels = img_gray[mask == 255]
            if len(face_pixels) == 0:
                prob_maps.append(np.zeros(256))
                continue

            hist = cv2.calcHist([face_pixels], [0], None, [256], [0, 256])
            prob_map = hist / hist.sum()
            prob_maps.append(prob_map.flatten())

        return prob_maps

    def get_prior_face_masks(
        self,
        img_path: Path,
        selected_pts: List[Dict[str, List[Tuple[int, int]]]],
        sigma: float = 35.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get distance based probability masks for all given faces.

        in:
            img_path: `Path`: path to input image
            selected_pts: `List[Dict[str, List[Tuple[int, int]]]]`: List of selected (x, y) points per face
            sigma: `float` [DEFAULT `35.0`]

        out:
            prior_face_masks: `List[np.ndarray]`: List of prior face masks
            heatmaps: `List[np.ndarrays]`: Visualization of prior_face_masks
            heatmaps_blended: `List[np.ndarrays]`: heatmaps but blended with original image
        """
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        height, width = img_gray.shape[:2]
        rows, cols = np.indices((height, width), dtype=np.float32)

        prob_maps = []
        heatmaps = []
        heatmaps_blended = []

        TWO_SIGMA_SQUARE = 2 * (sigma**2)

        for region in selected_pts:
            # each region corresponds to a face
            landmarks_single_face: List[Tuple[int, int]] = []
            for key in region.keys():
                points: List[Tuple[int, int]] = region.get(key)
                landmarks_single_face += points

            prob_map = np.zeros(img_gray.shape, dtype=np.float32)

            if len(landmarks_single_face) > 0:
                for landmark in landmarks_single_face:
                    lm_x, lm_y = landmark[0], landmark[1]

                    dist_sq = (rows - lm_y) ** 2 + (cols - lm_x) ** 2
                    gaussian_contrib = np.exp(-dist_sq / TWO_SIGMA_SQUARE)

                    prob_map += gaussian_contrib

            # scaling down all values
            max_val = np.max(prob_map)

            if max_val > 1.0 + Utils.EPSILON:
                prob_map /= max_val

            prob_maps.append(prob_map)

            colormap = plt.get_cmap("plasma")
            heatmap_rgba = colormap(prob_map)
            heatmap = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

            alpha = 0.6
            blended_img_bgr = cv2.addWeighted(
                cv2.cvtColor(
                    heatmap, cv2.COLOR_RGB2BGR
                ),  # Convert heatmap to BGR for cv2
                alpha,
                img_rgb,
                1 - alpha,
                0,
            )
            heatmap_blended = cv2.cvtColor(
                blended_img_bgr, cv2.COLOR_BGR2RGB
            )  # Convert back to RGB if needed

            heatmaps.append(heatmap)
            heatmaps_blended.append(heatmap_blended)

        return prob_maps, heatmaps, heatmaps_blended

    def get_skin_pixel_maps(
        self,
        img_path: Path,
        intensity_histograms: List[np.ndarray],
        prior_face_masks: List[np.ndarray],
        alpha: float,
        lamdba_: float,
    ) -> List[np.ndarray]:
        """
        Combine intensity histograms and prior faces to get a skin mask.

        in:
            img_path: `Path`: path to input img
            intensity_histograms: `List[np.ndarray]`: List of each faces intensity histograms
            prior_face_masks: `List[np.ndarray]`: List of prior face masks for each face

        out:
            skin_pixel_maps: `List[np.ndarray]`
        """
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        skin_pixel_maps = []

        if (a := len(intensity_histograms)) != (b := len(prior_face_masks)):
            raise ValueError(
                f"len(intensity_histograms) should be equal to len(prior_face_masks) | {a} != {b}"
            )

        for i in range(len(prior_face_masks)):
            intensity_hist = intensity_histograms[i]
            face_mask = prior_face_masks[i]

            intensity_prob_all_pixels = intensity_hist[img_gray]
            prior_mask_weighted = (face_mask + Utils.EPSILON) ** alpha

            combined_prob_map = intensity_prob_all_pixels * prior_mask_weighted

            skin_mask = np.zeros_like(img_gray, dtype=np.uint8)
            skin_mask[combined_prob_map > lamdba_] = 255

            skin_pixel_maps.append(skin_mask)

        return skin_pixel_maps
