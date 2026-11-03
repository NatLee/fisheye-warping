import numpy as np
import pytest


@pytest.fixture
def fisheye_img():
    """A small square top-down "fisheye" image: coloured by angle and radius."""
    size = 64
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xx - size / 2, yy - size / 2
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 255
    img[..., 1] = np.clip(np.hypot(dx, dy) * 8, 0, 255)
    img[..., 2] = 200
    img[np.hypot(dx, dy) > size / 2] = 0
    return img


@pytest.fixture
def panorama_img():
    """A small panorama with vertical colour stripes."""
    img = np.zeros((40, 160, 3), np.uint8)
    for i in range(8):
        img[:, i * 20:(i + 1) * 20] = (i * 30, 255 - i * 30, 128)
    return img
