
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

    # Revised _get_threshold method
    def _get_threshold(
            self,
            kappa: np.float64,
            seed_superpixel_labels: List[int],  # List of non_mask_labels that are seeds
            most_favorable_divergence: List[FeatureDivergence]  # List of divergences FROM non_mask TO best_mask
    ) -> np.float64:

        # Create a map from non_mask_label to its most favorable divergence object
        # Ensure label[0] is indeed the non_mask_label and label[1] the best mask_label
        div_map_for_non_mask = {fd.label[0]: fd for fd in most_favorable_divergence}
        print(f"Size of div_map_for_non_mask: {len(div_map_for_non_mask)}")  # Debug size

        total_probability = 0.0
        valid_seeds_count = 0
        missing_seeds = []

        seed_set = set(seed_superpixel_labels)  # Use set for faster checking if needed

        for label in seed_superpixel_labels:  # Iterate through the seed labels (non_mask_labels)
            # Look up the divergence object corresponding to this non_mask_label seed
            fd = div_map_for_non_mask.get(label)
            # print(f"Checking seed label: {label}, Found divergence: {'Yes' if fd is not None else 'No'}") # Verbose debug

            if fd is None:
                missing_seeds.append(label)
                continue  # Skip if divergence vector not found for this specific seed label

            # Calculate P_node for this seed using its most favorable divergence
            prob = self.superpixel_probability.get_superpixel_skin_probability(fd)
            total_probability += prob
            valid_seeds_count += 1

        if missing_seeds:
            print(
                f"Warning: Could not find divergence vectors for {len(missing_seeds)} seed labels: {missing_seeds[:20]}...")  # Print some missing ones

        if valid_seeds_count == 0:
            print("Warning: No valid seed divergences found to calculate average probability. Returning threshold 1.0.")
            return np.float64(1.0)

        average_probability = total_probability / valid_seeds_count
        threshold = kappa * average_probability

        print(
            f"Kappa: {kappa:.4f}, Avg Seed Prob: {average_probability:.4f}, Valid Seeds Count: {valid_seeds_count}, Calculated Threshold: {threshold:.4f}")

        return np.float64(threshold)

    # Inside the Region class:

    def grow(
            self,
            slic_superpixels: cv2.ximgproc.SuperpixelSLIC,
            seed_superpixel_labels: List[int],  # Initial seeds (non_mask_labels)
            all_feature_vectors: List[Feature],
            most_favorable_divergence: List[FeatureDivergence],  # List of divergences FROM non_mask TO best_mask
            num_iterations: int,
            kappa: np.float64 = np.float64(0.5),
    ):
        # --- Pre-computation ---
        # 1. Create efficient lookups
        feature_vector_map = {fv.label: fv for fv in all_feature_vectors}

        # Create map from non_mask_label to its most favorable divergence object
        # This map is used to get P_node for *any* non-mask superpixel
        divergence_map_for_non_mask = {fd.label[0]: fd for fd in most_favorable_divergence}
        print(f"Grow: Size of div_map_for_non_mask: {len(divergence_map_for_non_mask)}")

        # 2. Calculate threshold ONCE using the initial seeds
        if not seed_superpixel_labels:
            return [], []
        # Pass the original list, _get_threshold will build its own map now
        threshold = self._get_threshold(kappa, seed_superpixel_labels, most_favorable_divergence)
        print(f"Grow: Calculated Growth Threshold: {threshold:.4f}")

        # --- Initialization ---
        current_skin_labels_set = set(seed_superpixel_labels)
        # Start processed set only with seeds. Neighbors need evaluation.
        processed_as_candidate_set = set(seed_superpixel_labels)
        non_skin_labels_set = set()

        # --- Iterative Growth ---
        for i in range(num_iterations):
            newly_added_skin_this_iteration = set()
            # Use a queue for BFS-like expansion
            from collections import deque
            candidate_queue = deque()

            # Find initial candidates for this iteration: unprocessed neighbors of current skin
            print(f"\n--- Iteration {i + 1}/{num_iterations} ---")
            print(f"Current skin size: {len(current_skin_labels_set)}")
            current_border = set()  # Superpixels on the border of the current skin region
            for current_label in current_skin_labels_set:
                neighbors = self._get_neighbour_superpixels(
                    slic_superpixels,
                    current_label,
                    # Only find neighbors NOT already classified as skin or non-skin
                    invalid_neighbours=list(current_skin_labels_set | non_skin_labels_set)
                )
                # Add neighbors that haven't been queued this iteration yet
                for neighbor in neighbors:
                    if neighbor not in processed_as_candidate_set:
                        candidate_queue.append(neighbor)
                        processed_as_candidate_set.add(neighbor)  # Mark as queued/processed
                        current_border.add(neighbor)  # Keep track of border for this iter

            print(f"Found {len(candidate_queue)} candidates to evaluate this iteration.")
            if not candidate_queue:
                print("No new candidates found. Stopping early.")
                break

            # Evaluate candidates found in this pass
            num_added = 0
            num_rejected = 0
            while candidate_queue:  # Process all candidates added in this pass
                neighbour_label = candidate_queue.popleft()

                # Get features and divergence efficiently
                neighbour_fv = feature_vector_map.get(neighbour_label)
                # Get the MOST FAVORABLE divergence for this neighbor (calculated in Cell 13)
                neighbour_fvd = divergence_map_for_non_mask.get(neighbour_label)

                if neighbour_fv is None or neighbour_fvd is None:
                    # print(f"Skipping neighbor {neighbour_label}: Missing feature or divergence.")
                    num_rejected += 1
                    non_skin_labels_set.add(neighbour_label)
                    continue

                # Calculate probability
                p_new = self._get_p_new(
                    neighbour_fv,
                    neighbour_fvd,  # Pass the specific most favorable divergence
                    all_feature_vectors,  # Pass the full list (or map)
                    list(current_skin_labels_set),  # Pass current skin labels
                    slic_superpixels,
                )

                # Check against threshold
                # print(f"  Neighbor: {neighbour_label}, P_node: {self._get_p_node(neighbour_fvd):.4f}, P_edge: TBD, P_new: {p_new:.4f}, Threshold: {threshold:.4f}, Pass: {p_new > threshold}")

                if p_new > threshold:
                    newly_added_skin_this_iteration.add(neighbour_label)
                    num_added += 1
                else:
                    non_skin_labels_set.add(neighbour_label)
                    num_rejected += 1

            # --- Update after evaluating all candidates for this iteration ---
            if not newly_added_skin_this_iteration:
                print("No new skin labels added in this iteration. Stopping early.")
                break  # Stop if no growth occurred

            print(f"Iteration {i + 1}: Added {num_added}, Rejected {num_rejected}. Updating skin set...")
            current_skin_labels_set.update(newly_added_skin_this_iteration)
            # Don't update processed_labels_set here, it tracks things ever evaluated

        # --- Return final results ---
        print(f"Growth finished. Final skin labels: {len(current_skin_labels_set)}")
        return list(current_skin_labels_set), list(non_skin_labels_set)


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


