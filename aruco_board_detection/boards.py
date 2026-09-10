"""ArUco board definitions and image generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GridBoardConfig:
    """Physical and visual definition of an OpenCV ArUco GridBoard.

    ``size`` is ``(columns, rows)`` and all physical lengths are in meters.
    ``board_width`` includes markers and horizontal separation, but no margin.
    """

    dictionary: object
    size: tuple[int, int]
    marker_length: float
    board_width: float
    print_width: float | None = None
    marker_separation: float = field(init=False)
    board: object = field(init=False, repr=False)
    detector: object = field(init=False, repr=False)
    center: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        columns, rows = self.size
        if columns < 1 or rows < 1:
            raise ValueError("size values must be positive")
        if self.marker_length <= 0 or self.board_width <= 0:
            raise ValueError("marker_length and board_width must be positive")
        if self.print_width is not None and self.print_width < self.board_width:
            raise ValueError("print_width cannot be smaller than board_width")

        if columns == 1:
            if not np.isclose(self.board_width, self.marker_length):
                raise ValueError("a one-column board_width must equal marker_length")
            self.marker_separation = 0.0
        else:
            self.marker_separation = (
                self.board_width - columns * self.marker_length
            ) / (columns - 1)
            if self.marker_separation < 0:
                raise ValueError("board_width is too small for the requested markers")

        self.board = cv2.aruco.GridBoard(
            self.size,
            self.marker_length,
            self.marker_separation,
            self.dictionary,
        )
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
        width, height = self.dimensions
        self.center = np.array([width / 2.0, height / 2.0, 0.0])

    @property
    def dimensions(self) -> tuple[float, float]:
        """Return the board content width and height in meters."""
        columns, rows = self.size
        height = rows * self.marker_length + (rows - 1) * self.marker_separation
        return self.board_width, height

    @property
    def marker_ids(self) -> tuple[int, ...]:
        """Return marker IDs assigned to the board."""
        ids = np.asarray(self.board.getIds()).reshape(-1)
        return tuple(int(marker_id) for marker_id in ids)

    @property
    def print_dimensions(self) -> tuple[float, float]:
        """Return the configured PDF page dimensions in meters."""
        board_width, board_height = self.dimensions
        page_width = self.print_width or board_width
        margin = (page_width - board_width) / 2.0
        return page_width, board_height + 2.0 * margin

    def generate_image(
        self,
        width_px: int = 2160,
        *,
        margin_px: int = 0,
        output: str | Path | None = None,
    ) -> np.ndarray:
        """Generate a grayscale board image and optionally write it to disk."""
        if width_px <= 0 or margin_px < 0:
            raise ValueError("width_px must be positive and margin_px non-negative")

        width, height = self.dimensions
        height_px = round(width_px * height / width)
        image = self.board.generateImage(
            (width_px + 2 * margin_px, height_px + 2 * margin_px),
            marginSize=margin_px,
            borderBits=1,
        )
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"could not write board image to {path}")
        return image

    def generate_pdf(
        self,
        output: str | Path,
        *,
        image_width_px: int = 2160,
    ) -> Path:
        """Generate a PDF whose board and page have exact physical dimensions.

        PDF support requires the optional dependencies installed by
        ``pip install aruco-board-detection[pdf]``.
        """
        try:
            from PIL import Image
            from reportlab.lib.utils import ImageReader
            from reportlab.pdfgen.canvas import Canvas
        except ImportError as error:
            raise ImportError(
                "PDF export requires Pillow and ReportLab; install "
                "aruco-board-detection[pdf]"
            ) from error

        if image_width_px <= 0:
            raise ValueError("image_width_px must be positive")

        board_width, _ = self.dimensions
        page_width, page_height = self.print_dimensions
        margin_m = (page_width - board_width) / 2.0
        margin_px = round(image_width_px * margin_m / board_width)
        image = self.generate_image(image_width_px, margin_px=margin_px)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        points_per_meter = 72.0 / 0.0254
        page_size = (
            page_width * points_per_meter,
            page_height * points_per_meter,
        )
        canvas = Canvas(str(output_path), pagesize=page_size)
        canvas.drawImage(
            ImageReader(Image.fromarray(image)),
            0,
            0,
            width=page_size[0],
            height=page_size[1],
        )
        canvas.showPage()
        canvas.save()
        return output_path
