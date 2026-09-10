# ArUco board detection

A focused Python library for generating and detecting OpenCV ArUco grid boards,
estimating their 3D pose, and projecting image points onto the board plane.

This repository intentionally contains no robot controls, simulator integration,
camera-specific configuration, GUI, or plotting code.

## Installation

```console
python -m pip install -e .
```

For development:

```console
python -m pip install -e ".[dev]"
python -m pytest
```

PDF generation is optional:

```console
python -m pip install -e ".[pdf]"
```

## Quick start

```python
import cv2
import numpy as np

from aruco_board_detection import BoardDetector, GridBoardConfig

board = GridBoardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    size=(3, 4),
    marker_length=0.05,
    board_width=0.56,
    print_width=0.60,
)

board.generate_image(output="board.png")
board.generate_pdf("board.pdf")

detector = BoardDetector(
    board,
    camera_matrix=np.array(
        [[735.0, 0.0, 320.0], [0.0, 735.0, 240.0], [0.0, 0.0, 1.0]]
    ),
)

frame = cv2.imread("frame.png")
result = detector.detect(frame)
if result is not None:
    print(result.board_to_camera)
    print(result.project_image_point((320, 240)))
```

Generate a printable PNG, an exact-size PDF, or individual markers with
`GridBoardConfig.generate_image()`, `GridBoardConfig.generate_pdf()`, and
`generate_markers()`.

All physical dimensions are expressed in meters. The pose matrix transforms
coordinates from the centered board frame into the camera frame.
