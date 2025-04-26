import cv2
import numpy as np
from pathlib import Path
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import local_binary_pattern
from typing import List, Dict, Tuple

from src.projectUtils import Utils
from src.region import FeatureDivergence


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
        

    def get_slic_superpixels(self, img_path: str) -> tuple[cv2.ximgproc.SuperpixelSLIC, np.ndarray, np.ndarray, int]:
        """
        computes superpixels using the slic algorithm and returns the results.

        in:
            img_path: `str`: path to the input image file

        out:
            slic: `cv2.ximgproc.SuperpixelSLIC`: slic superpixel object
            contoured_img: `np.ndarray`: rgb image with superpixel contours overlaid
            labels: `np.ndarray`: array of superpixel labels for each pixel
            num_superpixels: `int`: number of superpixels detected
        """
        img_read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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


class FeatureExtractor:
    def __init__(self, region_size: int = 10, neighborhood_size: int = 3):
        """
        initializes the superpixel feature extractor with specified region and neighborhood sizes.

        in:
            region_size `int`: size of the superpixel region
            neighborhood_size `int`: size of the neighborhood for entropy calculation
        """
        self.region_size = region_size
        self.neighborhood_size = neighborhood_size
        # self.superpixel_extractor = SuperpixelFeatureExtractor(region_size, neighborhood_size)

        self.p_r_values = [(4, 1), (4, 3), (8, 1), (8, 3), (16, 2), (16, 5), (4, 2), (4, 4), (8, 2), (8, 4), (16, 3), (16, 7)]
        self.k_values = {
            4: [1, 2, 4],
            8: [1, 4, 7, 8],
            16: [1, 8, 15, 16],
        }

    def extract_basic_superpixel_features_superpixels(self, img_path: str, slic_superpixels: cv2.ximgproc.SuperpixelSLIC) -> list[dict]:
        """
        extracts basic features (mean intensity, std intensity, entropy) for each superpixel.

        in:
            img_path: `str`: path to the input image file
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: the precomputed slic superpixel object

        out:
            features `List[Dict]`: list of feature dictionaries for each superpixel, containing:
                - label: `int`: superpixel label
                - mean_intensity: `float`: mean intensity of the superpixel
                - std_intensity: `float`: standard deviation of the intensity
                - entropy: `float`: entropy of the intensity distribution within the superpixel
        """
        img_read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_read = np.array(img_read, dtype=np.uint8)

        entropy_map = entropy(img_read, disk(self.neighborhood_size))

        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        features = []

        for label in unique_labels:
            mask = (labels == label)
            region_pixels = img_read[mask]
            region_entropy = entropy_map[mask]

            feature_dict = {
                'label': int(label),
                'mean_intensity': float(np.mean(region_pixels)),
                'std_intensity': float(np.std(region_pixels)),
                'entropy': float(np.mean(region_entropy))
            }
            features.append(feature_dict)

        return features

    def extract_local_binary_patterns_superpixels(self, img_path: str, slic_superpixels: cv2.ximgproc.SuperpixelSLIC) -> list[dict]:
        """
        extracts local binary pattern (lbp) based features and lacunarity for each superpixel.

        in:
            img_path: `str`: path to the input image file
            slic_superpixels: `cv2.ximgproc.SuperpixelSLIC`: the precomputed slic superpixel object

        out:
            feature_dicts `List[Dict]`: list of dictionaries per superpixel, each containing:
                - lbp_maps `Dict[Tuple[int, int], np.ndarray]`: raw lbp pixel values per (p, r) pair
                - lacunarity_vector `np.ndarray`: vector of lacunarity values across lbp bins
        """
        img_read = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        labels = slic_superpixels.getLabels()
        unique_labels = np.unique(labels)

        lbp_maps = {}
        feature_dicts = []

        for (p, r) in self.p_r_values:
            lbp_map = local_binary_pattern(img_read, p, r, method="uniform")
            lbp_maps[(p, r)] = lbp_map

        for label in unique_labels:
            mask = (labels == label)
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
            
            label_feature_dict = {
                "label": int(label),
                "lacunarity_vector": np.array(label_lacunarities)
            }

            feature_dicts.append(label_feature_dict)

        return feature_dicts

    def extract_all_features_superpixels(self, img_path: Path, slic: cv2.ximgproc.SuperpixelSLIC) -> List[Dict]:
        """
        Combination of basic and lbp feature extractors to provide single feature vector for a superpixel.

        in:
            img_path: `Path`: path to image
            slic: `cv2.ximgprov.SuperpixelSLIC`: SLIC object
        
        out:
            feature_vectors: `List[Dict]`: List of feature vectors per superpixel
        """
        basic_features = self.extract_basic_superpixel_features_superpixels(img_path, slic)
        lbp_features = self.extract_local_binary_patterns_superpixels(img_path, slic)

        basic_features.sort(key = lambda x: x['label'])
        lbp_features.sort(key = lambda x: x['label'])

        if not (len(basic_features) == len(lbp_features)):
            raise ValueError("Feature lists are not the same length")
        
        combined_features = []
        for b, l in zip(basic_features, lbp_features):
            if b['label'] != l['label']:
                raise ValueError(f"Label mismatch after sorting: {b['label']} != {l['label']}")
        
            merged = { **b, **l }
            combined_features.append(merged)
        
        return combined_features

    def extract_all_features_mask(self, img_path: Path, mask: np.ndarray) -> Dict:

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        img = img.astype(np.uint8)
        mask = mask.astype(np.bool)

        masked_pixels = img[mask]
        mean_intensity = float(np.mean(masked_pixels))
        std_intensity = float(np.std(masked_pixels))

        entropy_map = entropy(img, disk(self.neighborhood_size))
        masked_entropy = entropy_map[mask]
        mean_entropy = float(np.mean(masked_entropy))

        # LBP & Lacunarity
        lacunarity_vector = []
        for (p, r) in self.p_r_values:
            lbp_map = local_binary_pattern(img, p, r, method="uniform")
            masked_lbp = lbp_map[mask]

            for k in self.k_values[p]:
                bin_map = (masked_lbp == k).astype(np.uint8)
                num_zeros = np.sum(bin_map == 0)
                total = masked_lbp.size
                lac = num_zeros / total if total > 0 else 0
                lacunarity_vector.append(lac)

        return {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "entropy": mean_entropy,
            "lacunarity_vector": np.array(lacunarity_vector),
        }

    def calculate_feature_learned_probability(self,
            all_superpixel_features: List[Dict],
            initial_seed_feature: Dict
    ) -> Tuple[List[Dict], Dict]:
        """
        calculates learned probability for the features in each superpixel, based on distance to initial seed features

        in:
            all_superpixel_features: (list[Dict]): list of features for every superpixel
            initial_seed_features (dict): average features for initial seed region

        out:
            list[Dict]: list of learned probability for each feature per superpixel
        """

        num_superpixels = len(all_superpixel_features)

        expected_feature_keys = ['mean_intensity', 'std_intensity', 'entropy', 'lacunarity_vector']

        distances = {key: [] for key in expected_feature_keys}

        seed_features_copy = initial_seed_feature.copy()

        for features in all_superpixel_features:
            features_copy = features.copy()
            dist_dict = FeatureDivergence.get_divergence(features_copy, seed_features_copy)

            for key in expected_feature_keys:
                distances[key].append(dist_dict[key])

        norm_params = {}
        for key in expected_feature_keys:
            curr_distances = distances[key]

            min_d = np.min(curr_distances)
            max_d = np.max(curr_distances)
            range_d = max_d - min_d + Utils.EPSILON

            norm_params[key] = {'min': min_d, 'range': max_d}

        feature_probabilities = []

        for i in range(num_superpixels):
            label = all_superpixel_features[i]['label']
            probs = {'label': label}

            for key in expected_feature_keys:
                dist = distances[key][i]
                p = norm_params[key]

                clamped_dist = np.clip(dist, p['min'], p['min'] + p['range'] - Utils.EPSILON)
                normalised_dist = (clamped_dist - p['min'] / p['range'])

                probs[key] = Utils.phi_k(normalised_dist)

            feature_probabilities.append(probs)

        return feature_probabilities, norm_params


    def calculate_combined_probability(self,
        individual_feature_probabilities: List[Dict]
    ) -> List[Dict]:
        """
        calculates combined initial prob for every superpixel by multiplying individual feature probabilities

        in:
            individual_feature_probabilities (list[Dict]): list of feature probabilities for each superpixel

        out:
            List[Dict]: list with each item containing label of the superpixel and it's combined initial probability
        """

        combined_probabilities = []

        probability_value_keys = ['mean_intensity', 'std_intensity', 'entropy', 'lacunarity_vector']

        for prob_dict in individual_feature_probabilities:
            label = prob_dict.get('label')

            p = 1.0
            keys_found = 0

            for pk in probability_value_keys:
                prob_value = prob_dict.get(pk)
                p *= prob_value
                keys_found += 1

            p_clipped = float(np.clip(p, 0.0, 1.0))
            combined_probabilities.append({'label': label, 'p': p_clipped})

        return combined_probabilities
