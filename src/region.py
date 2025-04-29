import numpy as np
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
import cv2
from pathlib import Path

from src.feature import FeatureDivergence, FeatureExtractor, Feature, EFeature
from src.superpixels import SuperpixelExtractor


class Region:
    def __init__(self) -> None:
        self.spe = SuperpixelExtractor()
        self.fe = FeatureExtractor()

    def select_seed_superpixels(
        self,
        most_favorable_divergence: List[FeatureDivergence],
        feature_selection: List[EFeature],
        percentile_threshold: np.float64 = np.float64(0.9),
    ) -> List[int]:
        """
        Returns top few superpixels with highest learned probability.

        in:
            most_favorable_divergence: `List[FeatureDivergence]`: list of divergence vectors with highest learned probability
            feature_selection: `List[EFeature]`: List of features to consider when calculating learned probability.
            percentile_threshold: `np.float64` [DEFAULT `np.float64(0.9)`]: The percentile threshold for valid seed superpixels

        out:
            seed_superpixel_labels: `List[int]`: List of all seed superpixel labels
        """
        if not most_favorable_divergence:
            return []

        prob_label_pairs = [
            (
                div.get_phi_n().get_learned_probability(feature_selection),
                div.get_self_label(),
            )
            for div in most_favorable_divergence
        ]
        probabilities = np.array([prob for prob, _ in prob_label_pairs])

        threshold = np.percentile(probabilities, percentile_threshold * 100)

        selected_labels = [
            label for prob, label in prob_label_pairs if prob >= threshold
        ]

        return selected_labels

    def grow(
        self,
        slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
        seed_superpixel_labels: List[int],
        all_feature_vectors: List[Feature],
        most_favorable_divergence: List[FeatureDivergence],
        mask_superpixel_labels: List[int],
        feature_selection: List[EFeature],
        num_iterations: int = 10,
        kappa: np.float64 = np.float64(0.5),
    ) -> Tuple[List[int], List[int]]:
        """
        Region growing algorithm. See report for detailed working.

        in:
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: Precomputed SLIC superpixel object used for accessing superpixel adjacency and label information.
            seed_superpixel_labels: `List[int]`: Initial superpixel labels from which the region growing starts.
            all_feature_vectors: `List[Feature]`: Feature vectors associated with each superpixel, used for computing similarity.
            most_favorable_divergence: `List[FeatureDivergence]`: Divergence scores indicating how favorable each superpixel is to be added to the region.
            mask_superpixel_labels: `List[int]`: Superpixel labels allowed for consideration during growing (typically a binary mask).
            feature_selection: `List[EFeature]`: Subset of features to be used in divergence calculations.
            num_iterations: `int`: Number of region growing iterations to perform (default = 10).
            kappa: `np.float64`: Divergence threshold scaling factor controlling the strictness of acceptance (default = 0.5).

        out:
            skin_superpixel_labels: `List[int]`
            non_skin_superpixel_labels: `List[int]`
        """
        self.feature_selection = feature_selection
        self.slic_superpixels = slic_superpixels
        self.seed_superpixel_labels = seed_superpixel_labels

        self.all_feature_vectors = all_feature_vectors
        self.label_to_feature: Dict[int, Feature] = {
            f.label: f for f in all_feature_vectors
        }

        most_favorable_divergences: List[FeatureDivergence] = most_favorable_divergence
        self.label_to_divergence0: Dict[int, FeatureDivergence] = {
            fd.get_self_label(): fd for fd in most_favorable_divergences
        }

        self.skin_superpixel_label_set: Set[int] = set(mask_superpixel_labels) | set(
            seed_superpixel_labels
        )
        self.non_skin_superpixel_label_set: Set[int] = set()

        threshold: np.float64 = self._get_threshold(kappa)

        print("Starting Iterations of Region Growing Algorithm")
        for iter in range(num_iterations):
            print(f"----- Iteration {iter+1}/{num_iterations} -----")

            new_skin_labels: Set[int] = set()
            num_new_skin = 0
            new_non_skin_labels: Set[int] = set()
            num_new_non_skin = 0

            for skin_sp_label in self.skin_superpixel_label_set:
                all_neighbours = self.spe.get_neighbouring_superpixel_labels(
                    slic_superpixels, skin_sp_label
                )
                valid_neighbours = [
                    n
                    for n in all_neighbours
                    if n not in self.skin_superpixel_label_set
                    and n not in self.non_skin_superpixel_label_set
                ]

                for current_label in valid_neighbours:
                    current_feature = self.label_to_feature.get(current_label)
                    current_divergence = self.label_to_divergence0.get(current_label)

                    if current_feature is None or current_divergence is None:
                        print(
                            f"[ERROR]: Missing feature/divergence vector for label {current_feature}"
                        )
                        continue

                    p_new = self._get_p_new(current_feature, current_divergence)

                    if p_new > threshold:
                        new_skin_labels.add(current_label)
                        num_new_skin += 1
                    else:
                        new_non_skin_labels.add(current_label)
                        num_new_non_skin += 1

            print(f"Added {num_new_skin} superpixels: {list(new_skin_labels)}")
            print(
                f"Rejected {num_new_non_skin} superpixels: {list(new_non_skin_labels)}"
            )

            self.skin_superpixel_label_set.update(new_skin_labels)
            self.non_skin_superpixel_label_set.update(new_non_skin_labels)

            if not new_skin_labels and not new_non_skin_labels:
                print("No Additions or Rejections. Stopping Early.")
                break

            print("\n")

        return list(self.skin_superpixel_label_set), list(
            self.non_skin_superpixel_label_set
        )

    def _get_threshold(self, kappa: np.float64) -> np.float64:
        """
        Computer threshold

        in:
            kappa: `np.float64`

        out:
            threshold: `np.float64`
        """
        total_learned_probability: np.float64 = np.float64(0.0)
        num_valid = 0

        for seed_superpixel_label in self.seed_superpixel_labels:
            divergence = self.label_to_divergence0.get(seed_superpixel_label)

            if divergence is None:
                print(
                    f"[ERROR]: Missing divergence vector for label {seed_superpixel_label}"
                )
                continue

            learned_probability = divergence.get_phi_n().get_learned_probability(
                self.feature_selection
            )
            num_valid += 1
            total_learned_probability += learned_probability

        if num_valid == 0:
            return np.float64(0.0)

        mean_learned_probability = total_learned_probability / num_valid
        threshold = kappa * mean_learned_probability
        return np.clip(threshold, 0.0, 1.0).astype(np.float64)

    def _get_p_new(
        self,
        current_feature_vector: Feature,
        current_feature_divergence: FeatureDivergence,
    ) -> np.float64:
        """
        Get p_new for given superpixel

        in:
            current_feature_vector: `Feature`: Feature vector of superpixel for which to calculate p_new
            current_feature_divergence: `FeatureDivergence`: Divergence vector of superpixel for which to calculate p_new

        out:
            p_new: `np.float64`
        """
        p_node = self._get_p_node(current_feature_divergence)
        p_edge = self._get_p_edge(current_feature_vector)
        return p_node * p_edge

    def _get_p_node(self, current_feature_divergence: FeatureDivergence) -> np.float64:
        """
        Simple learned_probability wrapped

        in:
            current_feature_divergence: `FeatureDivergence`

        out:
            p_node: `FeatureDivergence`: Learned probability for current superpixel
        """
        return current_feature_divergence.get_phi_n().get_learned_probability(
            self.feature_selection
        )

    def _get_p_edge(self, current_feature_vector: Feature) -> np.float64:
        """
        Calculates p_edge for current superpixel by comparing against its neighbours

        in:
            current_feature_vector: `Feature`: Feature vector for current superpixel

        out:
            p_edge: `np.float64`: Local learned probability
        """
        current_label = current_feature_vector.label

        all_neighbours = self.spe.get_neighbouring_superpixel_labels(
            self.slic_superpixels, current_label
        )
        valid_neighbours = [
            n for n in all_neighbours if n in self.skin_superpixel_label_set
        ]

        divergence_sum: FeatureDivergence = None
        num_valid = 0

        for neighbour_label in valid_neighbours:
            neighbour_feature = self.label_to_feature.get(neighbour_label)

            if neighbour_feature is None:
                print(
                    f"[ERROR]: Missing neighbour feature vector for label {neighbour_label}"
                )
                continue

            divergence = current_feature_vector.get_divergence(neighbour_feature)
            if divergence_sum is not None:
                divergence_sum = divergence_sum + divergence
            else:
                divergence_sum = divergence

            num_valid += 1

        if num_valid == 0:
            return np.float64(1.0)

        avg_divergence = divergence_sum.div(np.float64(num_valid))
        phi_n = avg_divergence.get_phi_n()
        return phi_n.get_learned_probability(self.feature_selection)
