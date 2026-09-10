import cv2
import numpy as np

from aruco_board_detection import (
    BoardDetector,
    GridBoardConfig,
    matrix_to_vectors,
    vectors_to_matrix,
)


def make_board():
    return GridBoardConfig(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        size=(3, 4),
        marker_length=0.05,
        board_width=0.25,
    )


def test_board_geometry_and_generation():
    board = make_board()
    assert np.allclose(board.dimensions, (0.25, 0.35))
    assert np.allclose(board.center, (0.125, 0.175, 0.0))
    assert board.marker_ids == tuple(range(12))
    assert board.generate_image(500, margin_px=20).shape == (740, 540)


def test_generates_exact_size_pdf(tmp_path):
    board = GridBoardConfig(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        size=(3, 4),
        marker_length=0.05,
        board_width=0.25,
        print_width=0.30,
    )
    assert np.allclose(board.print_dimensions, (0.30, 0.40))

    output = board.generate_pdf(tmp_path / "board.pdf", image_width_px=500)
    assert output.read_bytes().startswith(b"%PDF-")


def test_detects_generated_board():
    board = make_board()
    image = board.generate_image(500, margin_px=50)
    camera_matrix = np.array(
        [[800.0, 0.0, image.shape[1] / 2],
         [0.0, 800.0, image.shape[0] / 2],
         [0.0, 0.0, 1.0]]
    )
    result = BoardDetector(board, camera_matrix).detect(image)
    assert result is not None
    assert len(result.ids) == 12
    assert result.board_to_camera.shape == (4, 4)
    center = result.project_image_point(
        (image.shape[1] / 2, image.shape[0] / 2)
    )
    assert np.allclose(center, (0.0, 0.0), atol=0.002)


def test_pose_conversion_round_trip():
    rvec = np.array([0.1, -0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])
    recovered_rvec, recovered_tvec = matrix_to_vectors(
        vectors_to_matrix(rvec, tvec)
    )
    assert np.allclose(recovered_rvec, rvec)
    assert np.allclose(recovered_tvec, tvec)
