import base64
import io
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose


class PoseEstimator:
    def __init__(self) -> None:
        # Use a single long-lived Pose instance for performance
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def decode_base64_image(image_b64: str) -> Optional[np.ndarray]:
        try:
            data = base64.b64decode(image_b64)
            image_array = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return img
        except Exception:
            return None

    def extract_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[List[mp_pose.PoseLandmark], List[Tuple[float, float]]]]:
        """
        Returns a list of (x, y) pixel coordinates for all pose landmarks if detected.
        """
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return None

        coords: List[Tuple[float, float]] = []
        for lm in result.pose_landmarks.landmark:
            coords.append((lm.x, lm.y))

        return list(mp_pose.PoseLandmark), coords

