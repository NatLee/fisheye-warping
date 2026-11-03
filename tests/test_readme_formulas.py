"""Check the formulas in the README ("The maths") against what the package really does.

The images used here hold their own coordinates: every pixel of the fisheye test image stores
its (x, y), every pixel of the panorama its (u, v). After warping, each output pixel tells where
it was read from, and that is compared with the formulas.
"""
import numpy as np
import pytest

from fisheyewarping import FisheyeWarping

N = 128              # fisheye image size; 8-bit pixels can hold coordinates up to 255
C = R = N / 2        # centre (C, C) and radius R
W, H = int(np.pi * R), int(R)


@pytest.fixture(scope='module')
def warper():
    yy, xx = np.mgrid[0:N, 0:N]
    fisheye = np.zeros((N, N, 3), np.uint8)
    fisheye[..., 0], fisheye[..., 1] = xx, yy
    warper = FisheyeWarping(fisheye)
    warper.build_dewarp_mesh()
    warper.build_rewarp_mesh()
    return warper


def test_dewarp_formula(warper):
    panorama = warper.run_dewarp().astype(np.float64)
    assert panorama.shape[:2] == (H, W) == (64, 201)
    v, u = np.mgrid[0:H, 0:W]
    phi = 2 * np.pi * u / W
    r = R - v
    want_x = np.floor(C - r * np.sin(phi))
    want_y = np.floor(C + r * np.cos(phi))
    inner = (u >= 2) & (v >= 2)  # the first two rows and columns are left black by 1.0.x
    dx = np.abs(panorama[..., 0] - want_x)[inner]
    dy = np.abs(panorama[..., 1] - want_y)[inner]
    # Exact for almost every pixel; floating-point rounding moves a few by one pixel
    assert np.mean((dx == 0) & (dy == 0)) > 0.99
    assert max(dx.max(), dy.max()) <= 1


def test_rewarp_formula(warper):
    v, u = np.mgrid[0:H, 0:W]
    panorama = np.zeros((H, W, 3), np.uint8)
    panorama[..., 0], panorama[..., 1] = u, v * 3  # x3 keeps a third of a pixel of the blended v
    fisheye = warper.run_rewarp_with_mesh(panorama).astype(np.float64)
    fisheye[..., 1] /= 3

    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    r = np.hypot(xx - C, yy - C)
    psi = np.mod(np.arctan2(C - xx, yy - C), 2 * np.pi)
    want_u = W * psi / (2 * np.pi)
    want_v = R - r
    # Leave out the 6 o'clock seam (where u wraps from W to 0), the centre and the rim
    inner = (r >= 2) & (r <= R - 2) & (np.minimum(want_u, W - want_u) >= 3)
    du = np.abs(fisheye[..., 0] - want_u)[inner]
    dv = np.abs(fisheye[..., 1] - want_v)[inner]
    # Up to 1 px because the left half is read from a mirrored panorama, plus 0.5 px from 8-bit
    # rounding, plus up to 1/32 px because older OpenCV blends at fixed 1/32-pixel steps
    # (1.516 px with Python 3.6, 1.500 px with OpenCV 5)
    assert max(du.max(), dv.max()) <= 1.6
    assert np.all(fisheye[r > R + 1] == 0)  # black outside the circle
