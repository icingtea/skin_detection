import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Dict, Set, Tuple, Optional

from src.projectUtils import Utils
from src.region import FeatureDivergence


class RegionGrower:

    def __init__(self, kappa: float = 0.85, max_iters: int = 10, p_edge_if_no_neighbours: float = 0.5):
        self.kappa = kappa
        self.max_iters = max_iters
        self.p_edge_if_no_neighbours = p_edge_if_no_neighbours
        self.feature_keys = ['mean_intensity','std_intensity','entropy','lacunarity_vector']

    def _get_superpixel_neighbours(self, labels: np.ndarray) -> Dict[int, Set[int]]:
        neighbours = {label: set() for label in np.unique(labels) if label != -1}
        rows, cols = labels.shape

        for r in range(rows):
            for c in range(cols):
                current_label = labels[r, c]
                if current_label == -1: continue

                if c + 1 < cols:
                    right_label = labels[r, c + 1]
                    if right_label != -1 and current_label != right_label:
                        neighbours[current_label].add(right_label)
                        neighbours[right_label].add(current_label)

                if r + 1 < rows:
                    bottom_label = labels[r + 1, c]
                    if bottom_label != -1 and current_label != bottom_label:
                        neighbours[current_label].add(bottom_label)
                        neighbours[bottom_label].add(current_label)

        return neighbours

    def _calculate_threshold_tf(self, initial_combination_probs_list: List[Dict], seed_mask_labels: Set[int]) -> float:
        probs_map = {item['label']: item['initial_combined_prob'] for item in initial_combination_probs_list}
        seed_probs = [probs_map[l] for l in seed_mask_labels if l in probs_map]
        avg_seed_prob = np.mean(seed_probs)
        threshold_tf = self.kappa * avg_seed_prob
        threshold_tf = min(threshold_tf, 0.99)
        return threshold_tf

    def grow(self, all_superpixel_features: List[Dict], initial_combined_probs: List[Dict], labels: np.ndarray, seed_mask_labels: Set[int], norm_params: Dict[str, Dict]) -> Set[int]:
        features_dict = {f['label']: f for f in all_superpixel_features}
        p_node_map = {item['label']: item['initial_combined_prob'] for item in initial_combined_probs}
        all_valid_labels = set(features_dict.keys())

        valid_seed_labels = {lbl for lbl in seed_mask_labels if lbl in all_valid_labels and lbl in p_node_map}
        if len(valid_seed_labels) != len(seed_mask_labels):
            print("seed labels invalid or missing features/p_node")
        if not valid_seed_labels:
            print("no valid seed labels remain")
            return set()

        print(f"region growing with {len(valid_seed_labels)} initial seed labels")
        current_skin_labels = valid_seed_labels.copy()
        processed_labels = valid_seed_labels.copy()
        candidate_queue = deque()

        adjacency_list = self._get_superpixel_neighbours(labels)
        threshold_tf = self._calculate_threshold_tf(initial_combined_probs, valid_seed_labels)

        for label in current_skin_labels:
            if label not in adjacency_list: continue
            for neighbor in adjacency_list.get(label):
                if neighbor in all_valid_labels and neighbor not in processed_labels:
                    if neighbor not in candidate_queue:
                        candidate_queue.append(neighbor)

                    processed_labels.add(neighbor)

        iterations = 0
        added_in_iteration = True

        while candidate_queue and iterations < self.max_iters and added_in_iteration:
            iterations += 1
            added_in_iteration = False
            level_size = len(candidate_queue)

            for _ in range(level_size):
                current_label = candidate_queue.popleft()

                current_features = features_dict.get(current_label)
                if current_features is None:
                    continue

                p_node = p_node_map.get(current_label)

                if current_label not in adjacency_list: continue

                skin_neighbors = (adjacency_list.get(current_label) & current_skin_labels) & all_valid_labels

                p_edge = 1.0

                if not skin_neighbors:
                    p_edge = self.p_edge_if_no_neighbours
                else:
                    calculated_p_edge = 1.0
                    features_contributed_count = 0

                    for key in self.feature_keys:
                        current_feat_val = current_features.get(key)
                        if current_feat_val is None: continue

                        distance_to_neighbors = []
                        valid_neighbor_features = 0

                        for neighbor_label in skin_neighbors:
                            neighbor_features = features_dict.get(neighbor_label)
                            if neighbor_features is None: continue

                            neighbor_feat_val = neighbor_features.get(key)
                            if neighbor_feat_val is None: continue

                            try:
                                dist = FeatureDivergence.euclidean_distance(current_feat_val, neighbor_feat_val)
                                distance_to_neighbors.append(dist)
                                valid_neighbor_features += 1
                            except ValueError:
                                pass

                        if not distance_to_neighbors: continue

                        avg_dist = np.mean(distance_to_neighbors)
                        p = norm_params.get(key)
                        if p is None: continue

                        clamped_avg_distance = np.clip(avg_dist, p['min'], p['min'] + p['range'])
                        normalised_avg_distance = (clamped_avg_distance - p['min']) / p['range'] if p['range'] > Utils.EPSILON else 0.0

                        phi_k_avg_dist = Utils.phi_k(normalised_avg_distance)
                        calculated_p_edge *= (phi_k_avg_dist + Utils.EPSILON)
                        features_contributed_count += 1

                    if features_contributed_count > 0:
                        p_edge = np.clip(calculated_p_edge, 0.0, 1.0)

                p_new = p_node * p_edge

                if p_new > threshold_tf:
                    current_skin_labels.add(current_label)
                    added_in_iteration = True

                    for neighbor in adjacency_list.get(current_label):
                        if neighbor in all_valid_labels and neighbor not in processed_labels:
                            if neighbor not in candidate_queue:
                                candidate_queue.append(neighbor)
                            processed_labels.add(neighbor)

        return current_skin_labels




