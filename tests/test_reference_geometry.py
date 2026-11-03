"""Pin the output geometry: compare against the reference formulas, recomputed point by point.

For square images the reference is the 1.0.x code exactly. For other sizes it takes the centre
and the radius from the image's real centre and inscribed circle, which 1.0.2 and earlier got
wrong (they built the centre from (rows, cols) and the radius from one side).

If one of these tests fails, the meshes or the images the package produces have changed.
That may be on purpose, but then it is a visible change for users and needs a note in the
changelog.

The rendered images are compared with OpenCV calls made here, in the same environment,
because `cv2.remap` itself gives slightly different pixels across OpenCV versions.
"""
import math

import cv2
import numpy as np
import pytest

from fisheyewarping import FisheyeWarping


def polar_image(rows, cols):
    """A test image whose colour changes with angle, radius and position."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    dx, dy = xx - cols / 2, yy - rows / 2
    img = np.zeros((rows, cols, 3), np.uint8)
    img[..., 0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 255
    img[..., 1] = np.clip(np.hypot(dx, dy) * 8, 0, 255)
    img[..., 2] = (xx * 3 + yy * 5) % 256
    return img


def reference_dewarp_maps(rows, cols):
    c_x, c_y = int(cols / 2), int(rows / 2)
    r1, r2 = 0, min(int(cols) - c_x, int(rows) - c_y)  # 1.0.x: int(cols) - c_x
    width = int(2.0 * ((r2 + r1) / 2) * np.pi)
    height = r2 - r1
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    # 1.0.x leaves the last row and column at 0
    for y in range(height - 1):
        for x in range(width - 1):
            r = (float(y) / float(height)) * (r2 - r1) + r1
            theta = (float(x) / float(width)) * 2.0 * np.pi
            map_x[y, x] = int(c_x + r * np.sin(theta))
            map_y[y, x] = int(c_y + r * np.cos(theta))
    return map_x, map_y


def reference_rewarp_maps(rows, cols, panorama_width):
    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)
    mask = np.zeros((rows, cols), np.uint8)
    # 1.0.x: centre [int(rows / 2), int(cols / 2)], top [int(rows / 2), 0], radius rows / 2;
    # the same numbers for square images
    center = np.asarray([int(cols / 2), int(rows / 2)])  # (x, y)
    top = np.asarray([int(cols / 2), 0])
    radius = min(rows, cols) / 2
    up = (top - center) / np.linalg.norm(top - center)
    for y in range(rows):
        for x in range(cols):
            offset = np.asarray((x, y)) - center
            distance = np.linalg.norm(offset)
            if distance <= radius:
                with np.errstate(invalid='ignore'):  # the centre pixel gives NaN, as in 1.0.x
                    angle = np.arccos(np.clip(np.dot(up, offset / distance), -1.0, 1.0))
                map_x[y, x] = angle / (2 * np.pi) * panorama_width
                map_y[y, x] = distance
                mask[y, x] = 255
    return map_x, map_y, mask


def rotate_180(img):
    """Rotate around (w/2, h/2), as 1.0.x does. Not the same as np.rot90: it shifts by one pixel."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1), (w, h))


def reference_rewarp(panorama, map_x, map_y, mask):
    canvas = cv2.remap(panorama, map_x, map_y, cv2.INTER_LINEAR)
    right = cv2.remap(cv2.flip(panorama, 1), map_x, map_y, cv2.INTER_LINEAR)
    split = int(canvas.shape[1] / 2) + 1  # 1.0.x: int(canvas.shape[0] / 2) + 1
    canvas[:, split:, :] = right[:, split:, :]
    return cv2.add(canvas, np.zeros(canvas.shape, np.uint8), mask=mask)


SHAPES = [(64, 64), (33, 33), (40, 56), (56, 40)]


def assert_same(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('use_multiprocessing', [False, True], ids=['single', 'multi'])
@pytest.mark.parametrize('rows, cols', SHAPES)
def test_meshes_match_the_reference_formulas(rows, cols, use_multiprocessing):
    frd = FisheyeWarping(polar_image(rows, cols), use_multiprocessing=use_multiprocessing)
    panorama_shape, map_x, map_y = frd.build_dewarp_mesh()
    ref_x, ref_y = reference_dewarp_maps(rows, cols)
    assert panorama_shape == (ref_x.shape[1], ref_x.shape[0])
    assert_same(map_x, ref_x)
    assert_same(map_y, ref_y)

    rewarp = frd.build_rewarp_mesh()
    for actual, expected in zip(rewarp, reference_rewarp_maps(rows, cols, ref_x.shape[1])):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)  # NaN at the centre equals NaN


@pytest.mark.parametrize('rows, cols', SHAPES)
def test_images_match_the_reference_pipeline(rows, cols, panorama_img):
    fisheye = polar_image(rows, cols)
    frd = FisheyeWarping(fisheye)
    _, map_x, map_y = frd.build_dewarp_mesh()
    rewarp_x, rewarp_y, mask = frd.build_rewarp_mesh()

    plain_dewarp = cv2.remap(fisheye, map_x, map_y, cv2.INTER_LINEAR)
    assert_same(frd.dewarp(fisheye), plain_dewarp)
    assert_same(frd.run_dewarp(), rotate_180(plain_dewarp))

    assert_same(frd.rewarp(panorama_img), reference_rewarp(panorama_img, rewarp_x, rewarp_y, mask))
    resized = rotate_180(cv2.resize(panorama_img, (map_x.shape[1], map_x.shape[0])))
    assert_same(frd.run_rewarp_with_mesh(panorama_img, save_path=None),
                rotate_180(reference_rewarp(resized, rewarp_x, rewarp_y, mask)))
    assert_same(frd.run_rewarp(), rotate_180(reference_rewarp(plain_dewarp, rewarp_x, rewarp_y, mask)))


def test_panorama_size_follows_the_shorter_side():
    # width = int(pi * outer radius), height = outer radius; the outer radius is half the shorter side
    for rows, cols in ((64, 64), (101, 101), (512, 512), (60, 80), (80, 60)):
        radius = min(cols - int(cols / 2), rows - int(rows / 2))
        shape, _, _ = FisheyeWarping(polar_image(rows, cols)).build_dewarp_mesh()
        assert shape == (int(math.pi * radius), radius)
