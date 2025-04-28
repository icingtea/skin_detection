import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import cv2
from pathlib import Path

from src.feature import FeatureDivergence, FeatureExtractor, Feature, EFeature
from src.superpixels import SuperpixelExtractor

class Region:
    def __init__(
        self,
        feature_selection: List[EFeature],
        kappa: np.float64 = np.float64(0.5)
    ) -> None:
        self.feature_selection = feature_selection
        self.superpixel_probability = SuperpixelProbability(feature_selection)

        self.skin_pixel_labels: List[int] = []
        self.non_skin_pixel_labels: List[int] = []

    def separate_feature_vectors(
        self,
        feature_vectors: List[Feature],
        mask_labels: List[int]
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
        mask_feature_vectors = [fv for fv in feature_vectors if fv.label in mask_labels]
        non_mask_feature_vectors = [fv for fv in feature_vectors if fv.label not in mask_labels]

        return (mask_feature_vectors, non_mask_feature_vectors)

    def get_most_favorable_divergence(
        self,
        mask_feature_vectors: List[Feature],
        feature_vector: Feature
    ) -> FeatureDivergence:
        """
        Calculates skin probability for each superpixel and returns the divergence vector with highest such probability
        """
        all_divergences = [feature_vector.get_divergence(mfv) for mfv in mask_feature_vectors]

        most_favorable_divergence = all_divergences[0]
        max_probability = self.superpixel_probability.get_superpixel_skin_probability(all_divergences[0])
        
        for divergence in all_divergences:
            skin_prob = self.superpixel_probability.get_superpixel_skin_probability(divergence)
            if skin_prob > max_probability:
                max_probability = skin_prob
                most_favorable_divergence = divergence
        
        return most_favorable_divergence

    def select_seed_superpixel_labels(
        self,
        feature_divergences: List[FeatureDivergence],
        percentile_threshold: np.float64 = np.float64(0.9)
    ) -> List[int]:
        """
        
        """

        if not feature_divergences:
            return []
        
        probabilities_and_labels = [
            (self.superpixel_probability.get_superpixel_skin_probability(fd), fd.label)
            for fd in feature_divergences
        ]

        probabilities = [prob for prob, _ in probabilities_and_labels]

        cutoff_probability = np.percentile(probabilities, percentile_threshold * 100)

        selected_labels = [
            label for prob, (label, _) in probabilities_and_labels
            if prob >= cutoff_probability
        ]

        return selected_labels

    def get_threshold():
        pass



class SuperpixelProbability:
    def __init__(self, feature_selection: List[EFeature]):
        self.feature_selection = feature_selection

    def phi_k(self, feature: np.float64) -> np.float64:
        """
        Map a distance value to a probability value for any float-like feature passed in

        in:
            feature_divergence: `FeatureDivergence`: A computed Kullback-Leibler divergence value

        out:
            phi: `np.float64`: f(x) = 1 / (1 + np.exp(-1 / x))
        """
        phi = 1 / (1 + np.exp(-1 / feature))
        return phi

    def get_phi_values(self, feature_divergence: FeatureDivergence) -> 'Phi_N':
        """
        Calculate phi (learned probability) for each feature divergence value in a FeatureDivergence object

        in:
            feature_divergence: `FeatureDivergence`: An object containing computed, feature-wise Kullback-Leibler divergence values

        out:
            feature_phi_vals: `Phi_N`: An object containing computed, feature-wise phi values
        """
        mean_intensity_phi = self.phi_k(feature_divergence.mean_intensity)
        std_intensity_phi = self.phi_k(feature_divergence.std_intensity)
        entropy_phi = self.phi_k(feature_divergence.entropy)
        lacunarity_vector_phi = self.phi_k(feature_divergence.lacunarity_vector)

        feature_phi_vals = Phi_N(
            label=feature_divergence.label,
            mean_intensity=mean_intensity_phi,
            std_intensity=std_intensity_phi,
            entropy=entropy_phi,
            lacunarity_vector=lacunarity_vector_phi,
        )

        return feature_phi_vals

    def get_superpixel_skin_probability(self, feature_divergence: FeatureDivergence) -> np.float64:
        """
        Calculate the probability of a superpixel being skin via feature-wise learned probabilities

        in:
            phi_values: `Phi_N`: An object containing computed, feature-wise phi probability values
            feature_selection: `List[EFeature]`: List of features to use while calculation of skin probability

        out:
            learned_probability: `np.float`: A probability value indicating the likelihood of skin
        """
        skin_probability = 1
        phi_values = self.get_phi_values(feature_divergence)

        if EFeature.MEAN_INTENSITY in self.feature_selection:
            skin_probability *= phi_values.mean_intensity

        if EFeature.STD_INTENSITY in self.feature_selection:
            skin_probability *= phi_values.std_intensity

        if EFeature.ENTROPY in self.feature_selection:
            skin_probability *= phi_values.entropy

        if EFeature.LACUNARITY_VECTOR in self.feature_selection:
            skin_probability *= phi_values.lacunarity_vector

        return skin_probability


@dataclass
class Phi_N:
    label: Tuple[int, int]
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.float64 | None
