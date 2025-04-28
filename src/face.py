import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path


class FaceDetector:
    def __init__(self, model_path: Path = Path("assets/lbfmodel.yaml")):
        self.cascade_classifier: cv2.CascadeClassifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.landmark_model = cv2.face.createFacemarkLBF()
        self.landmark_model.loadModel(str(model_path))

    def detect(
        self, img_path: Path, display_indices: bool = False
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int]], List[np.ndarray]]:
        """
        Detects faces and 68-point landmarks in a grayscale image and draws them on a rgb copy

        in:
            img_path `str`: path to the input image file

        out:
            img_rgb `np.ndarray`: rgb image with face boxes, centers, and landmarks drawn
            face_rectangles `List[np.ndarray]`: list of face rectangles in (x, y, w, h) format
            face_centers `List[Tuple[int, int]]`: center coordinates of each detected face
            landmarks_all_faces `List[np.ndarray]`: list of arrays of 68 (x, y) points per face
        """

        img_read = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_GRAY2RGB)

        face_rectangles = self.cascade_classifier.detectMultiScale(
            img_rgb, scaleFactor=1.1, minNeighbors=5
        )
        face_centers: List[Tuple[int, int]] = []

        for x, y, width, height in face_rectangles:
            cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)

            center_x, center_y = x + (width // 2), y + (height // 2)
            cv2.circle(img_rgb, (center_x, center_y), 3, (0, 0, 255), -1)

            face_centers.append((center_x, center_y))

        _, landmarks_all_faces = self.landmark_model.fit(img_read, face_rectangles)
        landmarks_all_faces = [landmarks[0] for landmarks in landmarks_all_faces]

        for _, landmarks_face in enumerate(landmarks_all_faces):
            for idx, coords in enumerate(landmarks_face):
                x, y = coords
                cv2.circle(img_rgb, (int(x), int(y)), 1, (255, 0, 0), -1)
                if display_indices:
                    cv2.putText(
                        img_rgb,
                        str(idx),
                        (int(x) + 2, int(y) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
        return img_rgb, list(face_rectangles), face_centers, landmarks_all_faces
