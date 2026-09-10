"""Individual ArUco marker generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2


def generate_markers(
    dictionary: object,
    marker_ids: Iterable[int],
    output_directory: str | Path,
    *,
    size_px: int = 120,
) -> list[Path]:
    """Generate one PNG per marker and return the written paths."""
    if size_px <= 0:
        raise ValueError("size_px must be positive")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for marker_id in marker_ids:
        marker_id = int(marker_id)
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
        path = output_directory / f"aruco_{marker_id:02d}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write marker image to {path}")
        paths.append(path)
    return paths
