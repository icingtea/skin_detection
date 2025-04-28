import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from src.feature import FeatureDivergence


@dataclass
class Phi_N:
    label: Tuple[int, int]
    mean_intensity: np.float64 | None
    std_intensity: np.float64 | None
    entropy: np.float64 | None
    lacunarity_vector: np.float64 | None


class SuperpixelProbability:
    def __init__(self, feature_divergence: FeatureDivergence):
        self.feature_divergences = feature_divergence
        self.learned_probabilities = self.get_phi_values(self.feature_divergences)
        self.superpixel_skin_probability = self.get_superpixel_skin_probability(
            self.learned_probabilities
        )

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

    def get_phi_values(self, feature_divergence: FeatureDivergence) -> Phi_N:
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
            mean_intensity_phi=mean_intensity_phi,
            std_intensity_phi=std_intensity_phi,
            entropy_phi=entropy_phi,
            lacunarity_vector_phi=lacunarity_vector_phi,
        )

        return feature_phi_vals

    def get_superpixel_skin_probability(self, phi_values: Phi_N) -> np.float64:
        """
        Calculate the probability of a superpixel being skin via feature-wise learned probabilities

        in:
            phi_values: `Phi_N`: An object containing computed, feature-wise phi probability values

        out:
            learned_probability: `np.float`: A probability value indicating the likelihood of skin
        """
        phi_n = [
            phi_values.entropy,
            phi_values.mean_intensity,
            phi_values.std_intensity,
            phi_values.lacunarity_vector,
        ]

        skin_probability = 1
        for phi_k in phi_n:
            skin_probability *= phi_k

        return skin_probability
