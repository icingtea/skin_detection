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
    ) -> None:
        self.feature_selection = feature_selection
        self.superpixel_probability = SuperpixelProbability(feature_selection)

        self.skin_pixel_labels: List[int] = []
        self.non_skin_pixel_labels: List[int] = []

    def separate_feature_vectors(
        self, feature_vectors: List[Feature], mask_labels: List[int]
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
        non_mask_feature_vectors = [
            fv for fv in feature_vectors if fv.label not in mask_labels
        ]

        return (mask_feature_vectors, non_mask_feature_vectors)

    def get_most_favorable_divergence(
        self, mask_feature_vectors: List[Feature], feature_vector: Feature
    ) -> FeatureDivergence:
        """
        Calculates skin probability for each superpixel and returns the divergence vector with highest such probability
        """
        all_divergences = [
            feature_vector.get_divergence(mfv) for mfv in mask_feature_vectors
        ]

        most_favorable_divergence = all_divergences[0]
        max_probability = self.superpixel_probability.get_superpixel_skin_probability(
            all_divergences[0]
        )

        for divergence in all_divergences:
            skin_prob = self.superpixel_probability.get_superpixel_skin_probability(
                divergence
            )
            if skin_prob > max_probability:
                max_probability = skin_prob
                most_favorable_divergence = divergence

        return most_favorable_divergence

    def select_seed_superpixel_labels(
        self,
        feature_divergences: List[FeatureDivergence],
        percentile_threshold: np.float64 = np.float64(0.9),
    ) -> List[int]:
        """
        Selects seed superpixels based on most favorable divergence for each superpixel. Calculates probability of superpixel being a skin path and then picks top percentile

        in:
            feature_divergences: `List[FeatureDivergence]`: List of most favorable divergences for each superpixel
            percentile_threshold: `np.float64`: Threshold for selecting seed superpixels. (0.9 represents any given superpixel must be in the top 10% to get selected)

        out:
            seed_superpixel_labels: `List[int]`: List of labels of all seed superpixels
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
            label
            for prob, (label, _) in probabilities_and_labels
            if prob >= cutoff_probability
        ]

        return selected_labels

    def get_threshold():
        pass

    def _get_neighbour_superpixels(
        self,
        slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
        label: int,
        invalid_neighbours: List[int] = [],
        valid_neighbours: List[int] = [],
    ) -> List[int]:
        """
        Returns labels of all valid neighbour superpixels to `label`. A neighbour is valid if it is not present in `invalid_neighbours`

        If `valid_neighbours` is non-empty, only neihbours in `valid_neighbours` are accepted.
        Otherwise, a neighbour is valid if it is not present in `invalid_neighbours`.

        in:
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: Superpixel object
            label: `int`: Label to find neighbours of
            invalid_neighbours: `List[int]`: Labels of all superpixels that arent valid. Union of skin and non_skin superpixel_labels

        out:
            neighbour_labels: `List[int]`: List of all valid neighbouring superpixels
        """
        all_labels = slic_superpixels.getLabels()

        mask: np.ndarray = (all_labels == label).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        edge_mask = dilated_mask - mask

        neighbour_labels = set()
        ys, xs = np.where(edge_mask == 1)
        for y, x in zip(ys, xs):
            neighbour_label = all_labels[y, x]
            if neighbour_label == label:
                continue

            if valid_neighbours:
                if neighbour_label in valid_neighbours:
                    neighbour_labels.add(neighbour_label)
            else:
                if neighbour_label not in invalid_neighbours:
                    neighbour_labels.add(neighbour_label)

        return list(neighbour_labels)

    def _get_p_new(
        self,
        feature_vector: Feature,
        divergence_vector: FeatureDivergence,
        all_feature_vectors: List[Feature],
        skin_superpixel_labels: List[int],
        slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
    ) -> np.float64:
        p_node = self._get_p_node(divergence_vector)
        p_edge = self._get_p_edge(
            feature_vector.label,
            feature_vector,
            all_feature_vectors,
            skin_superpixel_labels,
            slic_superpixels,
        )
        
        return p_node * p_edge

    def _get_p_node(self, feature_divergence: FeatureDivergence) -> np.float64:
        return self.superpixel_probability.get_superpixel_skin_probability(
            feature_divergence
        )

    def _get_p_edge(
        self,
        current_label: int,
        current_feature_vector: Feature,
        all_feature_vectors: List[Feature],
        skin_superpixel_labels: List[int],
        slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
    ) -> np.float64:
        neighbour_skin_superpixels = self._get_neighbour_superpixels(
            slic_superpixels,
            current_label,
            invalid_neighbours=[],
            valid_neighbours=skin_superpixel_labels,
        )

        if not neighbour_skin_superpixels:
            return np.float64(1.0)

        feature_vector_map = {fv.label: fv for fv in all_feature_vectors}

        # Initialize total distances
        total_distance = {
            EFeature.MEAN_INTENSITY: 0.0,
            EFeature.STD_INTENSITY: 0.0,
            EFeature.ENTROPY: 0.0,
            EFeature.LACUNARITY_VECTOR: 0.0,
        }

        num_neighbours = len(neighbour_skin_superpixels)

        for neighbour_label in neighbour_skin_superpixels:
            neighbour_fv = feature_vector_map.get(neighbour_label)
            if neighbour_fv is None:
                continue  # Just skip if feature vector not found

            divergence = current_feature_vector.get_divergence(neighbour_fv)

            if EFeature.MEAN_INTENSITY in self.feature_selection and divergence.mean_intensity is not None:
                total_distance[EFeature.MEAN_INTENSITY] += divergence.mean_intensity

            if EFeature.STD_INTENSITY in self.feature_selection and divergence.std_intensity is not None:
                total_distance[EFeature.STD_INTENSITY] += divergence.std_intensity

            if EFeature.ENTROPY in self.feature_selection and divergence.entropy is not None:
                total_distance[EFeature.ENTROPY] += divergence.entropy

            if EFeature.LACUNARITY_VECTOR in self.feature_selection and divergence.lacunarity_vector is not None:
                total_distance[EFeature.LACUNARITY_VECTOR] += divergence.lacunarity_vector

        # Now compute Φ values over averaged distances
        p_edge = 1.0
        for feature in self.feature_selection:
            avg_distance = total_distance[feature] / num_neighbours
            p_edge *= self.superpixel_probability.phi_k(avg_distance)

        return np.float64(p_edge)

    def _get_threshold(
        self,
        kappa: np.float64,
        seed_superpixel_labels: List[int],
        most_favorable_divergence: List[FeatureDivergence]
    ) -> np.float64:
        divergence_map = {fd.label: fd for fd in most_favorable_divergence}

        total_probability = 0.0
        for label in seed_superpixel_labels:
            fd = divergence_map.get(label)
            if fd is None:
                continue  # Skip if divergence vector is missing
            prob = self.superpixel_probability.get_superpixel_skin_probability(fd)
            total_probability += prob

        average_probability = total_probability / len(seed_superpixel_labels)
        threshold = kappa * average_probability

        return np.float64(threshold) 

    def grow(
        self,
        slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
        seed_superpixel_labels: List[int],
        all_feature_vectors: List[Feature],
        most_favorable_divergence: List[FeatureDivergence],
        num_iterations: int,
        kappa: np.float64 = np.float64(0.5),
    ):
        print("hello")
        skin_superpixel_labels = seed_superpixel_labels
        non_skin_superpixel_labels: List[int] = []

        for i in range(num_iterations):
            print(skin_superpixel_labels)
            for current_label in skin_superpixel_labels:
                candidate_neighbour_labels = self._get_neighbour_superpixels(
                    slic_superpixels,
                    current_label,
                    skin_superpixel_labels + non_skin_superpixel_labels,
                )

                print(candidate_neighbour_labels)

                for neighbour_label in candidate_neighbour_labels:
                    neighbour_fv = [
                        fv for fv in all_feature_vectors if fv.label == neighbour_label
                    ][0]

                    neighbour_fvd = [
                        fvd
                        for fvd in most_favorable_divergence
                        if fvd.label[0] == neighbour_label
                    ]
                    if not neighbour_fvd:
                        print("sex")
                        continue
                    neighbour_fvd = neighbour_fvd[0]

                    p_new = self._get_p_new(
                        neighbour_fv,
                        neighbour_fvd,
                        all_feature_vectors,
                        skin_superpixel_labels,
                        slic_superpixels,
                    )

                    threshold = self._get_threshold(
                        kappa,
                        seed_superpixel_labels,
                        most_favorable_divergence
                    )
                    print(p_new, threshold)
                    if p_new > threshold:
                        skin_superpixel_labels.append(int(neighbour_label))
                    else:
                        non_skin_superpixel_labels.append(int(neighbour_label))

        return (skin_superpixel_labels, non_skin_superpixel_labels)

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

    def get_phi_values(self, feature_divergence: FeatureDivergence) -> "Phi_N":
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

    def get_superpixel_skin_probability(
        self, feature_divergence: FeatureDivergence
    ) -> np.float64:
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
