"""Marker detection, board pose estimation, and planar projection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .boards import GridBoardConfig
from .pose import vectors_to_matrix


@dataclass(frozen=True)
class BoardDetection:
    """Successful detection of an ArUco board in an image."""

    corners: tuple[np.ndarray, ...]
    ids: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    board_to_camera: np.ndarray

    def project_image_point(
        self, point: tuple[float, float], *, height: float = 0.0
    ) -> tuple[float, float]:
        """Project an image pixel onto the board plane.

        ``height`` is the point's height above the board in meters. The result is
        expressed in the centered board coordinate system used by the pose.
        """
        rotation = self.board_to_camera[:3, :3]
        camera_in_board = -rotation.T @ self.translation_vector
        pixel = np.array([[[point[0], point[1]]]], dtype=np.float32)
        homography, _ = cv2.findHomography(self.image_points, self.object_points[:, :2])
        if homography is None:
            raise ValueError("could not calculate a board homography")
        plane_point = cv2.perspectiveTransform(pixel, homography)[0, 0]

        camera_height = camera_in_board[2]
        if np.isclose(camera_height, 0.0):
            raise ValueError("camera lies on the board plane")
        scale = (camera_height - height) / camera_height
        corrected = camera_in_board[:2] + scale * (
            plane_point - camera_in_board[:2]
        )
        return float(corrected[0]), float(corrected[1])


class BoardDetector:
    """Detect a configured grid board and estimate its camera-relative pose."""

    def __init__(
        self,
        config: GridBoardConfig,
        camera_matrix: np.ndarray,
        distortion: np.ndarray | None = None,
        *,
        min_points: int = 6,
    ) -> None:
        self.config = config
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("camera_matrix must have shape (3, 3)")
        self.distortion = (
            np.zeros(5, dtype=np.float64)
            if distortion is None
            else np.asarray(distortion, dtype=np.float64)
        )
        if min_points < 4:
            raise ValueError("min_points must be at least 4")
        self.min_points = min_points

    def detect(
        self, image: np.ndarray, *, annotation: np.ndarray | None = None
    ) -> BoardDetection | None:
        """Return a board detection, or ``None`` when pose cannot be estimated."""
        corners, ids, _ = self.config.detector.detectMarkers(image)
        if ids is None:
            return None
        if annotation is not None:
            cv2.aruco.drawDetectedMarkers(annotation, corners, ids)

        object_points, image_points = self.config.board.matchImagePoints(corners, ids)
        if object_points is None or len(object_points) < self.min_points:
            return None

        centered_points = np.asarray(object_points, dtype=np.float32) - self.config.center
        success, rvec, tvec = cv2.solvePnP(
            centered_points,
            image_points,
            self.camera_matrix,
            self.distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)
        return BoardDetection(
            corners=tuple(corners),
            ids=ids.reshape(-1).copy(),
            object_points=centered_points.reshape(-1, 3),
            image_points=np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            rotation_vector=rvec,
            translation_vector=tvec,
            board_to_camera=vectors_to_matrix(rvec, tvec),
        )
