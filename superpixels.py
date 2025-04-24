import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


class SuperpixelFeatureExtractor:
    def __init__(self, region_size: int = 10, neighborhood_size: int = 3):
        """
        initializes the superpixel feature extractor with specified region and neighborhood sizes

        in:
            region_size (int): size of the superpixel region
            neighborhood_size (int): size of the neighborhood for entropy calculation
        """
        self.region_size = region_size
        self.neighborhood_size = neighborhood_size
        self.algorithm = cv2.ximgproc.SLIC

    def get_slic_superpixels(
        self, img_path: str
    ) -> Tuple[cv2.ximgproc.SuperpixelSLIC, np.ndarray, np.ndarray, int]:
        """
        computes superpixels using the SLIC algorithm and returns the results

        in:
            img_path (str): path to the input image file

        out:
            slic (cv2.ximgproc.SuperpixelSLIC): SLIC superpixel object
            contoured_img (np.ndarray): RGB image with superpixel contours overlaid
            labels (np.ndarray): array of superpixel labels for each pixel
            num_superpixels (int): number of superpixels detected
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_rgb, algorithm=self.algorithm, region_size=self.region_size
        )
        slic.iterate(10)

        contour_mask = slic.getLabelContourMask()

        contour_colored_mask = np.zeros_like(img_rgb)
        contour_colored_mask[contour_mask == 255] = [0, 0, 255]

        contoured_img = cv2.addWeighted(img_rgb, 1, contour_colored_mask, 1, 0)

        return slic, contoured_img, slic.getLabels(), slic.getNumberOfSuperpixels()

    def basic_superpixel_features(
        self, img_path: str, slic_superpixels: cv2.ximgproc.SuperpixelSLIC
    ) -> List[dict]:
        """
        extracts basic features (mean intensity, std intensity, entropy) for each superpixel

        in:
            img_path (str): path to the input image file
            slic_superpixels (cv2.ximgproc.SuperpixelSLIC): the precomputed SLIC superpixel object

        out:
            features (list[dict]): list of feature dictionaries for each superpixel, containing:
                - label (int): superpixel label
                - mean_intensity (float): mean intensity of the superpixel
                - std_intensity (float): standard deviation of the intensity
                - entropy (float): entropy of the intensity distribution within the superpixel
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        features: List[dict] = []

        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        for label in unique_labels:
            superpixel_pixels = img[labels == label]
            mean_intensity = np.mean(superpixel_pixels)
            std_intensity = np.std(superpixel_pixels)

            entropy_map = np.zeros_like(img, dtype=float)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    r = self.neighborhood_size

                    x_min, x_max = max(i - r, 0), min(i + r + 1, img.shape[0])
                    y_min, y_max = max(j - r, 0), min(j + r + 1, img.shape[1])
                    neighborhood = img[x_min:x_max, y_min:y_max]

                    hist, _ = np.histogram(
                        neighborhood, bins=256, range=(0, 256), density=True
                    )
                    entropy_map[i, j] = -np.sum(hist * np.log(hist + 1e-7))

            mean_entropy = np.mean(entropy_map[labels == label])

            features.append(
                {
                    "label": label,
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity,
                    "entropy": mean_entropy,
                }
            )

        return features

    def display_superpixels(self, contoured_img: np.ndarray) -> None:
        """
        displays an RGB image with superpixel contours using matplotlib

        in:
            contoured_img (np.ndarray): RGB image with overlaid superpixel contours
        """
        plt.imshow(contoured_img)
        plt.axis("off")
        plt.show()
