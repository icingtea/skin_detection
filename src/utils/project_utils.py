import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path

from matplotlib import pyplot as plt


class Utils:
    EPSILON = 1e-9

    @staticmethod
    def display(
        img: np.ndarray, title: str | None = None, cmap: str | None = None
    ) -> None:
        """
        Display an image

        in:
            img: (`np.ndarray`): The image to display
            title: (`str`) [DEFAULT `None`]: self explanatory
        """

        plt.imshow(img, cmap=cmap)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def visualize_histograms(histograms: List[np.ndarray]) -> None:
        """
        Visualize a list of histograms

        in:
            histograms: (`List[np.ndarray]`): The list of histograms
        """

        n_maps = len(histograms)
        _, axes = plt.subplots(n_maps, 1, figsize=(10, 3 * n_maps))

        if n_maps == 1:
            axes = [axes]

        for i, hist in enumerate(histograms):
            x = np.arange(256)
            axes[i].bar(x, hist, color="blue", alpha=0.7)
            axes[i].set_title(f"Probability Distribution - Mask {i+1}")
            axes[i].set_xlabel("Pixel Intensity")
            axes[i].set_ylabel("Probability")

            non_zero = hist[hist > 0]
            if len(non_zero) > 0:
                max_prob = np.max(hist)
                mean = np.sum(x * hist) / np.sum(hist)
                axes[i].text(
                    0.02,
                    0.95,
                    f"Max Probability: {max_prob:.4f}\nMean: {mean:.1f}",
                    transform=axes[i].transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.show()

    @staticmethod
    def apply_intensity_probability_map(
        img_path: Path, intensity_histogram: np.ndarray
    ) -> None:
        """
        Apply a single intensity histogram to an image

        in:
            img_path: `Path`: Path to input image
            intensity_histogram: `np.ndarray`: Intensity Histogram to apply
        """
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        colormap = plt.get_cmap("plasma")

        non_zero_mask = intensity_histogram > 0
        if np.any(non_zero_mask):
            non_zero_values = intensity_histogram[non_zero_mask]

            min_prob = np.min(non_zero_values)
            max_prob = np.max(non_zero_values)

            enhanced_prob = np.zeros_like(intensity_histogram)

            # Apply stretching to non-zero values only
            if max_prob > min_prob:
                enhanced_prob[non_zero_mask] = (
                    intensity_histogram[non_zero_mask] - min_prob
                ) / (max_prob - min_prob)
            else:
                enhanced_prob[non_zero_mask] = 1.0
        else:
            enhanced_prob = intensity_histogram

        for intensity in range(256):
            if intensity_histogram[intensity] > 0:
                prob = enhanced_prob[intensity]
                rgba = colormap(float(prob))

                r, g, b = [int(255 * c) for c in rgba[:3]]
                img_rgb[img_gray == intensity] = [r, g, b]

        Utils.display(img_rgb)

    @staticmethod
    def phi_k(normalized_distance: float) -> float:
        """
        Corrected function to convert normalized distance to probability using 1 / (1 + exp(-1 / x)) (Eq 6 from the paper).

        Args:
            normalized_distance (float): The distance x, scaled typically between 0 and 1+. Must be non-negative.

        Returns:
            float: Probability value. Approaches 1 as distance -> 0+. approaches 0.5 as distance -> infinity.
        """
        assert isinstance(
            normalized_distance, np.float64
        ), f"Expected np.float64, got {type(normalized_distance)} | {normalized_distance}"

        x = np.maximum(0.0, normalized_distance)

        if x < Utils.EPSILON:
            return np.float64(1.0)
        else:
            exponent_term = np.float64(-1.0) / x
            exp_val = np.exp(exponent_term)
            probability = np.float64(1.0) / (np.float64(1.0) + exp_val)

            return np.clip(probability, np.float64(0.0), np.float64(1.0))

    @staticmethod
    def morphological_cleanup(
        img: np.ndarray,
        kernel_shape: int = cv2.MORPH_ELLIPSE,
        kernel_size: Tuple[int, int] = (3, 3),
        iterations: int = 1,
    ) -> np.ndarray:
        """
        Applies morphological opening followed by closing to clean up a mask.

        int:
            mask: `np.ndarray`: Input binary mask (0s and 255s)
            kernel_shape: `int`: OpenCV kernel shape constant (default: ELLIPSE)
            kernel_size: `Tuple[int, int]`: kernel size (default: (3,3))
            iterations : `int`: Number of times to apply each operation (default: 1)

        out:
            cleaned: `np.ndarray`

        """
        kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        return cleaned

    @staticmethod
    def display_selected_superpixels(
        img_path: Path,
        labels_array: np.ndarray,
        selected_labels: List[int],
        title: str | None = None,
        cmap: str | None = None,
    ) -> None:
        """
        Displays an image with only the selected superpixels visible.

        Args:
            img (np.ndarray): The original image
            labels_array (np.ndarray): Array of superpixel labels
            selected_labels (List[int]): List of superpixel labels to display
            title (str, optional): Title for the plot. Defaults to None.
            cmap (str, optional): Colormap for grayscale images. Defaults to None.
            highlight_color (tuple, optional): RGB color for highlighting selected superpixels. Defaults to red.
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros_like(labels_array, dtype=bool)
        for label in selected_labels:
            mask = np.logical_or(mask, labels_array == label)

        result = np.zeros_like(img)
        result[mask] = img[mask]

        plt.imshow(result, cmap=cmap)
        plt.axis("off")
        if title:
            plt.title(title)
        plt.show()
