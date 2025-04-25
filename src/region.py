from typing import List, Tuple, Dict
import numpy as np

class FeatureDivergence:
    @staticmethod
    def get_divergence(feature_vector_1: Dict, feature_vector_2: Dict) -> float:
        feature_vector_1.pop('label', None)
        feature_vector_2.pop('label', None)
        
        if not (len(feature_vector_1.keys()) == len(feature_vector_2.keys())):
            raise ValueError("Keys are not the same across feature vectors")
        
        keys = feature_vector_1.keys()

        divergence_vector = {}

        for key in keys:
            if key == 'label':
                continue
            divergence_vector[key] = FeatureDivergence.euclidean_distance(feature_vector_1[key], feature_vector_2[key])

        return divergence_vector

    @staticmethod
    def euclidean_distance(val1: float | np.ndarray, val2: float | np.ndarray) -> float:
        if not(type(val1) == type(val2)):
            raise ValueError(f"Values are of different types. {type(val1)} != {type(val2)}")

        if type(val1) == float:
            return abs(val1 - val2)
        elif type(val1) == np.ndarray:
            if val1.shape != val2.shape:
                raise ValueError(f"Shape mismatch in arrays: {val1.shape} != {val2.shape}")
            return float(np.linalg.norm(val1 - val2))
        else:
            raise ValueError(f"Unknown type. {type(val1)}")