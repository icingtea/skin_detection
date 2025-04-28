import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


class SuperpixelExtractor:
    def __init__(self, region_size: int = 10, neighborhood_size: int = 3):
        """
        initializes the superpixel feature extractor with specified region and neighborhood sizes.

        in:
            region_size `int`: size of the superpixel region
            neighborhood_size `int`: size of the neighborhood for entropy calculation
        """
        self.region_size = region_size
        self.neighborhood_size = neighborhood_size
        self.algorithm = cv2.ximgproc.SLIC

    def get_slic_superpixels(
        self, img_path: Path
    ) -> Tuple[cv2.ximgproc.SuperpixelSLIC, np.ndarray, np.ndarray, int]:
        """
        computes superpixels using the slic algorithm and returns the results.

        in:
            img_path: `Path`: path to the input image file

        out:
            slic: `cv2.ximgproc.SuperpixelSLIC`: slic superpixel object
            contoured_img: `np.ndarray`: rgb image with superpixel contours overlaid
            labels: `np.ndarray`: array of superpixel labels for each pixel
            num_superpixels: `int`: number of superpixels detected
        """
        img_read = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_GRAY2RGB)

        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_rgb, algorithm=self.algorithm, region_size=self.region_size
        )
        slic.iterate(10)

        contour_mask = slic.getLabelContourMask()
        contour_colored_mask = np.zeros_like(img_rgb)
        contour_colored_mask[contour_mask == 255] = [0, 0, 255]
        contoured_img = cv2.addWeighted(img_rgb, 1, contour_colored_mask, 1, 0)

        return slic, contoured_img, slic.getLabels(), slic.getNumberOfSuperpixels()
