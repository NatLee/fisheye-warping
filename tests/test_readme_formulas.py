"""Check the formulas in the README ("The maths") against what the package really does.

The images used here hold their own coordinates: every pixel of the fisheye test image stores
its (x, y), every pixel of the panorama its (u, v). After warping, each output pixel tells where
it was read from, and that is compared with the formulas.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fisheyewarping import FisheyeWarping

# (h, w) of the fisheye test images: square, wide and tall. 8-bit pixels hold coordinates up to 255.
SHAPES = [(128, 128), (96, 128), (128, 96)]


@pytest.fixture(scope='module', params=SHAPES, ids=lambda shape: '{}x{}'.format(shape[1], shape[0]))
def case(request):
    h, w = request.param
    radius = min(w, h) / 2  # the largest circle that fits, centred in the image
    yy, xx = np.mgrid[0:h, 0:w]
    fisheye = np.zeros((h, w, 3), np.uint8)
    fisheye[..., 0], fisheye[..., 1] = xx, yy
    warper = FisheyeWarping(fisheye)
    warper.build_dewarp_mesh()
    warper.build_rewarp_mesh()
    return SimpleNamespace(w=w, h=h, cx=w / 2, cy=h / 2, R=radius, W=int(np.pi * radius), H=int(radius),
                           warper=warper)


def test_dewarp_formula(case):
    panorama = case.warper.run_dewarp().astype(np.float64)
    assert panorama.shape[:2] == (case.H, case.W)
    v, u = np.mgrid[0:case.H, 0:case.W]
    phi = 2 * np.pi * u / case.W
    r = case.R - v
    want_x = np.floor(case.cx - r * np.sin(phi))
    want_y = np.floor(case.cy + r * np.cos(phi))
    inner = (u >= 2) & (v >= 2)  # the first two rows and columns are left black by 1.0.x
    dx = np.abs(panorama[..., 0] - want_x)[inner]
    dy = np.abs(panorama[..., 1] - want_y)[inner]
    # Exact for almost every pixel; floating-point rounding moves a few by one pixel
    # (measured: 99.4% to 100% exact on 128 to 512 px images)
    assert np.mean((dx == 0) & (dy == 0)) > 0.99
    assert max(dx.max(), dy.max()) <= 1


def test_rewarp_formula(case):
    v, u = np.mgrid[0:case.H, 0:case.W]
    panorama = np.zeros((case.H, case.W, 3), np.uint8)
    panorama[..., 0], panorama[..., 1] = u, v * 3  # x3 keeps a third of a pixel of the blended v
    fisheye = case.warper.run_rewarp_with_mesh(panorama).astype(np.float64)
    fisheye[..., 1] /= 3

    yy, xx = np.mgrid[0:case.h, 0:case.w].astype(np.float64)
    r = np.hypot(xx - case.cx, yy - case.cy)
    psi = np.mod(np.arctan2(case.cx - xx, yy - case.cy), 2 * np.pi)
    want_u = case.W * psi / (2 * np.pi)
    want_v = case.R - r
    # Leave out the 6 o'clock seam (where u wraps from W to 0), the centre and the rim
    inner = (r >= 2) & (r <= case.R - 2) & (np.minimum(want_u, case.W - want_u) >= 3)
    du = np.abs(fisheye[..., 0] - want_u)[inner]
    dv = np.abs(fisheye[..., 1] - want_v)[inner]
    # The bound adds up three known errors: up to 1 px because the left half is read from a
    # mirrored panorama, 0.5 px from storing the coordinates in 8-bit pixels, and up to 1/32 px
    # because OpenCV blends at fixed 1/32-pixel steps (measured: 1.516 px with OpenCV 4.6 on
    # Python 3.6, 1.500 px with OpenCV 5)
    mirror_px, rounding_px, blend_step_px = 1, 0.5, 1 / 32
    assert max(du.max(), dv.max()) <= mirror_px + rounding_px + blend_step_px
    assert np.all(fisheye[r > case.R + 1] == 0)  # black outside the circle
