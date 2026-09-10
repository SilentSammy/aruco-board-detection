"""Pose conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np


def vectors_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rotation/translation vectors to a 4x4 transform."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def matrix_to_vectors(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform to OpenCV rotation/translation vectors."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return rvec.reshape(3), transform[:3, 3].copy()
