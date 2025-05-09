from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.ndimage import generic_filter
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
from enum import Enum
from src.utils.project_utils import Utils


class EFeature(Enum):
    MEAN_INTENSITY = "mean_intensity"
    STD_INTENSITY = "std_intensity"
    ENTROPY = "entropy"
    LACUNARITY_VECTOR = "lacunarity_vector"


@dataclass
class PhiN:
    label: Tuple[int, int]
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.float64 | None

    def get_learned_probability(
        self, feature_selection: List["EFeature"]
    ) -> np.float64:
        """
        Returns the learned probability of this divergence vector

        in:
            feature_selection: `List[EFeature]`: List of features to be considered while calculation of learned probability

        out:
            learned_probability: `np.float64`
        """
        skin_probability = np.float64(1.0)

        if (
            EFeature.MEAN_INTENSITY in feature_selection
            and self.mean_intensity is not None
        ):
            skin_probability *= self.mean_intensity

        if (
            EFeature.STD_INTENSITY in feature_selection
            and self.std_intensity is not None
        ):
            skin_probability *= self.std_intensity

        if EFeature.ENTROPY in feature_selection and self.entropy is not None:
            skin_probability *= self.entropy

        if (
            EFeature.LACUNARITY_VECTOR in feature_selection
            and self.lacunarity_vector is not None
        ):
            skin_probability *= self.lacunarity_vector

        return skin_probability


@dataclass
class FeatureDivergence:
    label: Tuple[int, int] | None
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.float64 | None

    def get_self_label(self) -> int:
        return self.label[0]

    def get_other_label(self) -> int:
        return self.label[1]

    def get_phi_n(self) -> "PhiN":
        """
        Returns Phi vector, which is element wise phi_k of divergence vector

        out:
            phi_vector: `PhiN`
        """
        return PhiN(
            label=self.label,
            mean_intensity=(
                Utils.phi_k(self.mean_intensity)
                if self.mean_intensity is not None
                else None
            ),
            std_intensity=(
                Utils.phi_k(self.std_intensity)
                if self.std_intensity is not None
                else None
            ),
            entropy=Utils.phi_k(self.entropy) if self.entropy is not None else None,
            lacunarity_vector=(
                Utils.phi_k(self.lacunarity_vector)
                if self.lacunarity_vector is not None
                else None
            ),
        )

    def __add__(self, other: "FeatureDivergence") -> "FeatureDivergence":
        """
        Helper method to easily add two divergnce vectors. Performs element wise sum
        """
        if not isinstance(other, FeatureDivergence):
            return NotImplemented

        def add_or_none(a, b):
            if a is not None and b is not None:
                return np.float64(a + b)
            else:
                return None

        return FeatureDivergence(
            label=None,  # Label is nonsensical when considering sum of two divergences
            mean_intensity=add_or_none(self.mean_intensity, other.mean_intensity),
            std_intensity=add_or_none(self.std_intensity, other.std_intensity),
            entropy=add_or_none(self.entropy, other.entropy),
            lacunarity_vector=add_or_none(
                self.lacunarity_vector, other.lacunarity_vector
            ),
        )

    def div(self, divisor: np.float64) -> "FeatureDivergence":
        """
        Helper method to divide divergence vector with a scalar

        in:
            division: `np.float64`

        out:
            divergence_vector: `FeatureDivergence`
        """
        if not isinstance(divisor, np.float64):
            return NotImplemented

        def div_or_none(val):
            if val is not None:
                return np.float64(val / divisor)

        return FeatureDivergence(
            label=self.label,
            mean_intensity=div_or_none(self.mean_intensity),
            std_intensity=div_or_none(self.std_intensity),
            entropy=div_or_none(self.entropy),
            lacunarity_vector=div_or_none(self.lacunarity_vector),
        )


@dataclass
class Feature:
    label: int
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.ndarray | None

    def __add__(self, other: "Feature") -> "Feature":
        """
        Helper function to concatenate two partial Feature vectors together. Performs label validation.

        in:
            other: `Feature`

        out:
            feature: `Feature`
        """
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

    def get_most_favorable_divergence(
        self,
        comparing_feature_vectors: List["Feature"],
        feature_selection: List[EFeature],
    ) -> "FeatureDivergence":
        """
        Returns divergence vector with highest learned probability against comparing set

        in:
            comparing_feature_vectors: `List[Feature]`: List of features against which to compare

        out:
            most_favorable_divergence: `FeatureDivergence`
        """
        all_divergences = [
            self.get_divergence(cfv) for cfv in comparing_feature_vectors
        ]
        return max(
            all_divergences,
            key=lambda div: div.get_phi_n().get_learned_probability(feature_selection),
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
        self, img_map: np.ndarray, P: np.uint32, R: np.uint32
    ) -> np.ndarray:
        """
        Extract rotationally invariant default binary pattern map (riu2) for a given P and R

        in:
            img_path: `Path`: Path to img
            P: `np.int32`: Number of neighbors used to create a binary pattern
            R: `np.int32`: Radius of circular neighborhood

        out:
            riu2_map: `np.ndarray`: Resulting rotationally invariant LBP map
        """
        default_lbp_map: np.ndarray = local_binary_pattern(
            img_map, P, R, method="default"
        )
        default_lbp_map = default_lbp_map.astype(np.uint32)

        lookup_table = np.empty(2**P, dtype=np.uint32)

        for code in range(2**P):
            code_bits = [(code >> i) & 1 for i in range(P)]
            num_code_transitions = sum(
                code_bits[i] != code_bits[(i + 1) % P] for i in range(P)
            )
            if num_code_transitions <= 2:
                lookup_table[code] = code
            else:
                lookup_table[code] = P + 1

        riu2_map = lookup_table[default_lbp_map]
        return riu2_map

    def extract_basic_features_superpixels(
        self, img_path: Path, slic_superpixels: cv2.ximgproc.SuperpixelSLIC
    ) -> List[Feature]:
        """
        Extract the following features from the slic_superpixels object:
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
        img = np.array(img, dtype=np.float64)

        square_size = (self.neighborhood_size, self.neighborhood_size)

        def compute_entropy(values):
            hist, _ = np.histogram(values, bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))

        entropy_map = generic_filter(img, compute_entropy, size=square_size)

        def std_func(values):
            return np.std(values)

        std_map = generic_filter(img, std_func, size=square_size)

        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        features: List[Feature] = []
        for label in unique_labels:
            mask = labels == label
            region_pixels = img[mask]
            region_entropy = entropy_map[mask]
            region_std = std_map[mask]

            features.append(
                Feature(
                    label=int(label),
                    mean_intensity=np.float64(np.mean(region_pixels)),
                    std_intensity=np.float64(np.mean(region_std)),
                    entropy=np.float64(np.mean(region_entropy)),
                    lacunarity_vector=None,
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
                    bin_map = (label_map == k).astype(np.uint16)
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
                    lacunarity_vector=np.array(label_lacunarities),
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

    @staticmethod
    def separate_feature_vectors(
        feature_vectors: List[Feature], mask_labels: List[int]
    ) -> Tuple[List[Feature], List[Feature]]:
        """
        Separate mask and non mask feature vectors:

        in:
            feature_vectors: `List[Feature]`
            mask_labels: `List[int]`

        out:
            mask_feature_vectors: `List[Feature]`: List of features inside the mask
            non_mask_feature_vectors: `List[Feature]`: List of features outside the mask
        """
        mask_label_set = set(mask_labels)

        mask_feature_vectors = []
        non_mask_feature_vectors = []

        for fv in feature_vectors:
            if fv.label in mask_label_set:
                mask_feature_vectors.append(fv)
            else:
                non_mask_feature_vectors.append(fv)

        return (mask_feature_vectors, non_mask_feature_vectors)
