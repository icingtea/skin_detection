from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern


@dataclass
class FeatureDivergence:
    label: Tuple[int, int]
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.float64 | None


@dataclass
class Feature:
    label: int
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.ndarray | None

    def __add__(self, other: "Feature") -> "Feature":
        if not isinstance(other, Feature):
            return NotImplemented

        if self.label != other.label:
            raise ValueError(
                f"Cannot add features with different labels: {self.label} != {other.label}"
            )

        def pick(a, b):
            return a if a is not None else b

        return Feature(
            label=self.label,
            mean_intensity=pick(self.mean_intensity, other.mean_intensity),
            std_intensity=pick(self.std_intensity, other.std_intensity),
            entropy=pick(self.entropy, other.entropy),
            lacunarity_vector=pick(
                (
                    self.lacunarity_vector.copy()
                    if self.lacunarity_vector is not None
                    else None
                ),
                (
                    other.lacunarity_vector.copy()
                    if other.lacunarity_vector is not None
                    else None
                ),
            ),
        )

    def get_divergence(self, other: "Feature") -> "FeatureDivergence":
        """
        Get the divergence vector for any two feature vectors. Calculates divergence as such:
        - mean_intensity: `euclidean_dist`
        - std_intensity: `euclidean_dist`
        - entropy: `euclidean_dist`
        - lacunarity_vector: `euclidean_dist`

        in:
            other: `Feature`: Feature to get divergence against

        out:
            divergence: `FeatureDivergence`
        """
        if not isinstance(other, Feature):
            return NotImplemented

        def euclidean(a, b):
            if a is None or b is None:
                return None
            return np.linalg.norm(a - b)

        return FeatureDivergence(
            label=(self.label, other.label),
            mean_intensity=euclidean(self.mean_intensity, other.mean_intensity),
            std_intensity=euclidean(self.std_intensity, other.std_intensity),
            entropy=euclidean(self.entropy, other.entropy),
            lacunarity_vector=euclidean(
                self.lacunarity_vector, other.lacunarity_vector
            ),
        )


class FeatureExtractor:
    def __init__(self, region_size: int = 10, neighborhood_size: int = 3):
        """
        initializes the feature extractor with specified region and neighborhood sizes.

        in:
            region_size `int`: size of the superpixel region
            neighborhood_size `int`: size of the neighborhood for entropy calculation
        """
        self.region_size = region_size
        self.neighborhood_size = neighborhood_size

        self.p_r_values = [
            (4, 1),
            (4, 3),
            (8, 1),
            (8, 3),
            (16, 2),
            (16, 5),
            (4, 2),
            (4, 4),
            (8, 2),
            (8, 4),
            (16, 3),
            (16, 7),
        ]
        self.k_values = {
            4: [1, 2, 4],
            8: [1, 4, 7, 8],
            16: [1, 8, 15, 16],
        }

    def _extract_lbp_riu_maps(
        self, img_map: np.ndarray, P: float, R: float
    ) -> np.ndarray:
        """
        DOC TODO
        """
        default_lbp_map: np.ndarray = local_binary_pattern(
            img_map, P, R, method="default"
        )
        default_lbp_map = default_lbp_map.astype(np.uint8)

        lookup_table = np.empty(2**P, dtype=np.uint8)

        for code in range(2**P):
            code_bits = [(code >> i) & 1 for i in range(P)]
            num_code_transitions = sum(
                code_bits[i] != code_bits[(i + 1) % P] for i in range(P)
            )
            if num_code_transitions <= 2:
                lookup_table[code] = sum(code_bits)
            else:
                lookup_table[code] = P + 1

        riu2_map = lookup_table[default_lbp_map]
        return riu2_map

    def extract_basic_features_superpixels(
        self, img_path: Path, slic_superpixels: cv2.ximgproc.SuperpixelSLIC
    ) -> List[Feature]:
        """
        Extract the following features from the slic_superpixels object
            - mean_intensity
            - std_intensity
            - entropy

        in:
            img_path: `Path`: Path to img
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: Superpixel object

        out:
            basic_features: `List[Feature]`: Partial Feature Vectors for each superpixel
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype=np.uint8)

        entropy_map = entropy(img, disk(self.neighborhood_size))

        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        features: List[Feature] = []
        for label in unique_labels:
            mask = labels == label
            region_pixels = img[mask]

            region_entropy = entropy_map[mask]

            features.append(
                Feature(
                    label=int(label),
                    mean_intensity=np.float64(np.mean(region_pixels)),
                    std_intensity=np.float64(np.std(region_pixels)),
                    entropy=np.float64(np.mean(region_entropy)),
                    lacunarity_vector=None
                )
            )

        return features

    def extract_local_binary_patterns_superpixels(
        self, img_path: Path, slic_superpixels: cv2.ximgproc.SuperpixelSLIC
    ) -> Tuple[List[Feature], Dict[Tuple[int, int], np.ndarray]]:
        """
        Extract the following features from the slic_superpixels object
            - lacunarity_vector

        in:
            img_path: `Path`: Path to img
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: Superpixel object

        out:
            lacunarity_features: `List[Feature]`: Partial Feature Vectors for each superpixel
            lbp_maps: `Dict[Tuple[int, int], np.ndarray]`: Raw lbp pixel values per (p, r) pair
        """
        img_read = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        lbp_maps = {}
        feature_dicts = []

        for p, r in self.p_r_values:
            lbp_map = self._extract_lbp_riu_maps(img_read, p, r)
            lbp_maps[(p, r)] = lbp_map

        for label in unique_labels:
            mask = labels == label
            label_lbp_maps = {}
            label_lacunarities = []

            for (p, r), map in lbp_maps.items():
                region_lbp_map = map[mask]
                label_lbp_maps[(p, r)] = region_lbp_map

            for (p, r), label_map in label_lbp_maps.items():
                for k in self.k_values[p]:
                    bin_map = (label_map == k).astype(np.uint8)
                    num_zeros = np.sum(bin_map == 0)
                    total_pixels = label_map.size
                    lacunarity = num_zeros / total_pixels
                    label_lacunarities.append(lacunarity)

            feature_dicts.append(
                Feature(
                    label=int(label),
                    mean_intensity=None,
                    std_intensity=None,
                    entropy=None,
                    lacunarity_vector=np.array(label_lacunarities)
                )
            )

        return feature_dicts, lbp_maps

    def extract_all_features_superpixels(
        self, img_path: Path, slic_superpixels: cv2.ximgproc.SuperpixelSLIC
    ) -> List[Feature]:
        """
        Combination of basic and lbp feature extractors to provide single feature vector for a superpixel.

        in:
            img_path: `Path`: path to image
            slic: `cv2.ximgprov.SuperpixelSLIC`: SLIC object

        out:
            feature_vectors: `List[Feature]`: List of complete feature vectors per superpixel
        """
        basic_features = self.extract_basic_features_superpixels(
            img_path, slic_superpixels
        )
        lacunarity_features, _ = self.extract_local_binary_patterns_superpixels(
            img_path, slic_superpixels
        )

        basic_features.sort(key=lambda x: x.label)
        lacunarity_features.sort(key=lambda x: x.label)

        if not ((a := len(basic_features)) == (b := len(lacunarity_features))):
            raise ValueError(f"Feature lists are not the same length {a} != {b}")

        combined_features = []
        for b, l in zip(basic_features, lacunarity_features):
            combined = b + l
            combined_features.append(combined)

        return combined_features
