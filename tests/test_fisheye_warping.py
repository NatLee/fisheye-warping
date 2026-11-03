import doctest
import pickle
import subprocess
import sys
import warnings

import cv2
import numpy as np
import pytest

from fisheyewarping import FisheyeWarping, MeshNotReadyError
from fisheyewarping.fisheyewarping import angle_between, angle_map, unit_vector

# The 64 x 64 `fisheye_img` fixture has radius 32, so its panorama is int(2 * pi * 16) = 100 pixels
# wide (the circle at half the radius) and 32 high (the radius)
PANORAMA_W, PANORAMA_H = 100, 32
FISHEYE_HW = (64, 64)


def test_create_obj():
    FisheyeWarping(None, use_multiprocessing=True)


def test_build_dewarp_mesh(fisheye_img):
    panorama_shape, map_x, map_y = FisheyeWarping(fisheye_img).build_dewarp_mesh()
    assert panorama_shape == (PANORAMA_W, PANORAMA_H)
    assert map_x.shape == map_y.shape == (PANORAMA_H, PANORAMA_W)
    assert map_x.max() > 0 and map_y.max() > 0


def test_run_dewarp_gives_panorama(fisheye_img):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    assert frd.run_dewarp().shape == (PANORAMA_H, PANORAMA_W, 3)


def test_multiprocessing_builds_the_same_meshes(fisheye_img):
    single = FisheyeWarping(fisheye_img, use_multiprocessing=False)
    multi = FisheyeWarping(fisheye_img, use_multiprocessing=True)
    _, sx, sy = single.build_dewarp_mesh()
    _, mx, my = multi.build_dewarp_mesh()
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_array_equal(sy, my)
    for a, b in zip(single.build_rewarp_mesh(), multi.build_rewarp_mesh()):
        np.testing.assert_array_equal(a, b)


def test_build_rewarp_mesh_and_rewarp_panorama(fisheye_img, panorama_img):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    map_x, map_y, mask = frd.build_rewarp_mesh()
    assert map_x.shape == map_y.shape == mask.shape == FISHEYE_HW
    assert mask[32, 32] == 255 and mask[0, 0] == 0  # inside vs. outside the circle
    result = frd.run_rewarp_with_mesh(panorama_img, save_path=None)
    assert result.shape == fisheye_img.shape


def test_saved_dewarp_mesh_can_be_loaded(fisheye_img, tmp_path):
    mesh = str(tmp_path / 'dewarp-mesh.pkl')
    builder = FisheyeWarping(fisheye_img)
    built = builder.build_dewarp_mesh(save_path=mesh)

    loader = FisheyeWarping(fisheye_img)
    loaded = loader.load_dewarp_mesh(mesh_path=mesh)
    assert loaded[0] == built[0] == (PANORAMA_W, PANORAMA_H)
    np.testing.assert_array_equal(loader.run_dewarp(), builder.run_dewarp())


def test_load_dewarp_mesh_saved_by_version_1_0_x(fisheye_img, tmp_path):
    # 1.0.x saved only (map x, map y); the panorama shape comes from the map size
    _, map_x, map_y = FisheyeWarping(fisheye_img).build_dewarp_mesh()
    mesh = tmp_path / 'old-dewarp-mesh.pkl'
    mesh.write_bytes(pickle.dumps((map_x, map_y)))

    panorama_shape, _, _ = FisheyeWarping(fisheye_img).load_dewarp_mesh(mesh_path=str(mesh))
    assert panorama_shape == (PANORAMA_W, PANORAMA_H)


def test_rewarp_panorama_with_loaded_meshes_only(fisheye_img, panorama_img, tmp_path):
    dewarp_mesh, rewarp_mesh = str(tmp_path / 'dewarp.pkl'), str(tmp_path / 'rewarp.pkl')
    builder = FisheyeWarping(fisheye_img)
    builder.build_dewarp_mesh(save_path=dewarp_mesh)
    builder.build_rewarp_mesh(save_path=rewarp_mesh)
    expected = builder.run_rewarp_with_mesh(panorama_img, save_path=None)

    # No fisheye image needed once both meshes exist (README "next time" example)
    frd = FisheyeWarping(None)
    frd.load_dewarp_mesh(mesh_path=dewarp_mesh)
    frd.load_rewarp_mesh(mesh_path=rewarp_mesh)
    output = str(tmp_path / 'rewarp-output.png')
    np.testing.assert_array_equal(frd.run_rewarp_with_mesh(panorama_img, save_path=output), expected)
    assert (tmp_path / 'rewarp-output.png').is_file()


def test_run_rewarp_round_trip_restores_the_fisheye(fisheye_img):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
    result = frd.run_rewarp()
    assert result.shape == fisheye_img.shape

    yy, xx = np.mgrid[0:64, 0:64]
    radius = np.hypot(xx - 32, yy - 32)
    inside = (radius >= 4) & (radius < 28)
    error = np.abs(result.astype(int) - fisheye_img.astype(int))[inside]
    assert np.median(error) <= 5  # fisheye -> panorama -> fisheye comes back close to the input
    assert (result[radius > 33] == 0).all()  # outside the circle stays black


def test_dewarp_works_on_grayscale_images(fisheye_img):
    gray = fisheye_img[..., 1].copy()
    frd = FisheyeWarping(gray)
    frd.build_dewarp_mesh()
    assert frd.run_dewarp().shape == (PANORAMA_H, PANORAMA_W)


def test_run_dewarp_and_run_rewarp_save_images(fisheye_img, tmp_path):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
    panorama = frd.run_dewarp(save_path=str(tmp_path / 'panorama.png'))
    fisheye = frd.run_rewarp(save_path=str(tmp_path / 'fisheye.png'))
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / 'panorama.png')), panorama)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / 'fisheye.png')), fisheye)


def test_saved_rewarp_mesh_round_trips(fisheye_img, tmp_path):
    mesh = str(tmp_path / 'rewarp-mesh.pkl')
    builder = FisheyeWarping(fisheye_img)
    builder.build_dewarp_mesh()
    built = builder.build_rewarp_mesh(save_path=mesh)
    loaded = FisheyeWarping(None).load_rewarp_mesh(mesh_path=mesh)
    for a, b in zip(built, loaded):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('call', [
    lambda frd, pano: frd.dewarp(pano),
    lambda frd, pano: frd.run_dewarp(),
    lambda frd, pano: frd.rewarp(pano),
    lambda frd, pano: frd.run_rewarp_with_mesh(pano, save_path=None),
    lambda frd, pano: frd.build_rewarp_mesh(),
], ids=['dewarp', 'run_dewarp', 'rewarp', 'run_rewarp_with_mesh', 'build_rewarp_mesh'])
def test_using_a_mesh_before_it_exists_raises_assertion_error(call, fisheye_img, panorama_img):
    # 1.0.x raised AssertionError here; code catching it must keep working
    with pytest.raises(AssertionError):
        call(FisheyeWarping(fisheye_img), panorama_img)


def test_angle_helpers():
    assert angle_between((1, 0, 0), (0, 1, 0)) == pytest.approx(np.pi / 2)
    assert angle_between((1, 0, 0), (1, 0, 0)) == pytest.approx(0)
    assert angle_between((1, 0, 0), (-1, 0, 0)) == pytest.approx(np.pi)
    np.testing.assert_allclose(unit_vector(np.asarray([3, 4])), [0.6, 0.8])


def test_zero_vector_gives_nan_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.isnan(unit_vector(np.asarray([0, 0]))).all()


def test_angle_map_gives_the_share_of_a_turn_from_the_top():
    center, top = np.asarray([10, 10]), np.asarray([10, 0])
    point, share, distance = angle_map(((10, 4), center, top, 10))
    assert point == (10, 4) and share == pytest.approx(0) and distance == pytest.approx(6)
    _, share, _ = angle_map(((10, 16), center, top, 10))
    assert share == pytest.approx(0.5)  # straight down: half a turn
    _, share, distance = angle_map(((0, 0), center, top, 10))
    assert share is None and distance > 10  # outside the circle


def test_building_meshes_leaves_numpy_warning_settings_alone(fisheye_img):
    # 1.0.x switched off NumPy's "invalid value" warnings for the whole program
    with np.errstate(invalid='warn'):
        frd = FisheyeWarping(fisheye_img)
        frd.build_dewarp_mesh()
        frd.build_rewarp_mesh()
        assert np.geterr()['invalid'] == 'warn'


def test_rewarp_mesh_can_be_built_from_a_grayscale_fisheye(fisheye_img, panorama_img):
    frd = FisheyeWarping(fisheye_img[..., 1].copy())
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
    assert frd.run_rewarp_with_mesh(panorama_img).shape == fisheye_img.shape


def test_grayscale_fisheye_round_trip(fisheye_img):
    # Dewarping always handled grayscale images; rewarping used to index three channels and crash
    gray = fisheye_img[..., 1].copy()
    frd = FisheyeWarping(gray)
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
    result = frd.run_rewarp()
    assert result.shape == gray.shape and result.dtype == gray.dtype

    # Each channel is warped on its own, so a grayscale image gives the same pixels as that channel
    colour = FisheyeWarping(fisheye_img)
    colour.build_dewarp_mesh()
    colour.build_rewarp_mesh()
    np.testing.assert_array_equal(result, colour.run_rewarp()[..., 1])


def test_rewarp_keeps_16_bit_images(fisheye_img, panorama_img):
    # The circle mask used to be applied with an 8-bit array, which OpenCV refuses for 16-bit images
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    _, _, mask = frd.build_rewarp_mesh()
    deep = panorama_img.astype(np.uint16) * 257  # 8-bit values stretched to the 16-bit range
    result = frd.rewarp(deep)
    assert result.dtype == np.uint16 and result.shape == fisheye_img.shape
    assert (result[mask == 0] == 0).all()  # black outside the circle
    assert result[mask > 0].max() > 255  # the 16-bit range survives


def test_docstring_examples():
    from fisheyewarping import fisheyewarping as core
    result = doctest.testmod(core)
    assert result.attempted > 0 and result.failed == 0


def test_missing_mesh_error_says_what_to_call(fisheye_img, panorama_img):
    with pytest.raises(MeshNotReadyError, match=r'build_dewarp_mesh\(\) or load_dewarp_mesh\(\)'):
        FisheyeWarping(fisheye_img).run_dewarp()
    with pytest.raises(MeshNotReadyError, match=r'build_rewarp_mesh\(\) or load_rewarp_mesh\(\)'):
        FisheyeWarping(fisheye_img).rewarp(panorama_img)
    assert issubclass(MeshNotReadyError, AssertionError) and issubclass(MeshNotReadyError, RuntimeError)


def test_missing_mesh_is_reported_under_python_O():
    # 1.0.x used assert, which python -O removes; then OpenCV failed with a confusing message
    code = ('import numpy as np\n'
            'from fisheyewarping import FisheyeWarping\n'
            'FisheyeWarping(np.zeros((8, 8, 3), np.uint8)).run_dewarp()')
    proc = subprocess.run([sys.executable, '-O', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    assert proc.returncode != 0
    assert 'MeshNotReadyError' in proc.stderr


@pytest.mark.parametrize('call', [
    lambda frd, mesh: frd.build_dewarp_mesh(),
    lambda frd, mesh: (frd.load_dewarp_mesh(mesh_path=mesh), frd.run_dewarp()),
    lambda frd, mesh: (frd.load_dewarp_mesh(mesh_path=mesh), frd.build_rewarp_mesh()),
    lambda frd, mesh: (frd.load_dewarp_mesh(mesh_path=mesh), frd.run_rewarp_with_mesh(None)),
], ids=['build_dewarp_mesh', 'run_dewarp', 'build_rewarp_mesh', 'run_rewarp_with_mesh'])
def test_missing_image_explains_cv2_imread(call, fisheye_img, tmp_path):
    mesh = str(tmp_path / 'dewarp.pkl')
    FisheyeWarping(fisheye_img).build_dewarp_mesh(save_path=mesh)
    with pytest.raises(TypeError, match='cv2.imread returns None'):
        call(FisheyeWarping(None), mesh)


@pytest.fixture
def saved_meshes(fisheye_img, tmp_path):
    frd = FisheyeWarping(fisheye_img)
    dewarp, rewarp = str(tmp_path / 'dewarp.pkl'), str(tmp_path / 'rewarp.pkl')
    _, map_x, map_y = frd.build_dewarp_mesh(save_path=dewarp)
    frd.build_rewarp_mesh(save_path=rewarp)
    old_dewarp = tmp_path / 'old-dewarp.pkl'
    old_dewarp.write_bytes(pickle.dumps((map_x, map_y)))
    not_a_mesh = tmp_path / 'not-a-mesh.pkl'
    not_a_mesh.write_bytes(pickle.dumps({'hello': 'world'}))
    return {'dewarp': dewarp, 'rewarp': rewarp, 'old dewarp': str(old_dewarp), 'not a mesh': str(not_a_mesh)}


def test_swapped_mesh_files_are_named_as_such(saved_meshes):
    # With OpenCV 5, a rewarp mesh loaded as a dewarp mesh used to give a garbage image without any error
    with pytest.raises(ValueError, match='holds a rewarp mesh, not a dewarp mesh.*swapped'):
        FisheyeWarping(None).load_dewarp_mesh(mesh_path=saved_meshes['rewarp'])
    for name in ('dewarp', 'old dewarp'):
        with pytest.raises(ValueError, match='holds a dewarp mesh, not a rewarp mesh.*swapped'):
            FisheyeWarping(None).load_rewarp_mesh(mesh_path=saved_meshes[name])


def test_other_pickles_are_not_meshes(saved_meshes):
    with pytest.raises(ValueError, match='not a dewarp mesh saved by FisheyeWarping'):
        FisheyeWarping(None).load_dewarp_mesh(mesh_path=saved_meshes['not a mesh'])
    with pytest.raises(ValueError, match='not a rewarp mesh saved by FisheyeWarping'):
        FisheyeWarping(None).load_rewarp_mesh(mesh_path=saved_meshes['not a mesh'])


def test_failed_image_write_warns(fisheye_img, tmp_path):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    with pytest.warns(UserWarning, match='could not write'):
        result = frd.run_dewarp(save_path=str(tmp_path / 'no-such-folder' / 'panorama.png'))
    assert result.shape == (PANORAMA_H, PANORAMA_W, 3)


def test_paths_can_be_pathlib_paths(fisheye_img, panorama_img, tmp_path):
    # 1.0.x silently saved nothing when save_path was a pathlib.Path
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh(save_path=tmp_path / 'dewarp.pkl')
    frd.build_rewarp_mesh(save_path=tmp_path / 'rewarp.pkl')
    frd.run_dewarp(save_path=tmp_path / 'panorama.png')
    frd.run_rewarp(save_path=tmp_path / 'round-trip.png')
    frd.run_rewarp_with_mesh(panorama_img, save_path=tmp_path / 'fisheye.png')
    for name in ('dewarp.pkl', 'rewarp.pkl', 'panorama.png', 'round-trip.png', 'fisheye.png'):
        assert (tmp_path / name).is_file()

    loaded = FisheyeWarping(None)
    loaded.load_dewarp_mesh(mesh_path=tmp_path / 'dewarp.pkl')
    loaded.load_rewarp_mesh(mesh_path=tmp_path / 'rewarp.pkl')
    np.testing.assert_array_equal(loaded.run_rewarp_with_mesh(panorama_img),
                                  cv2.imread(str(tmp_path / 'fisheye.png')))


def test_no_save_path_saves_nothing(fisheye_img, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh(save_path='')
    frd.build_rewarp_mesh(save_path=None)
    frd.run_dewarp(save_path='')
    assert list(tmp_path.iterdir()) == []


def polar_fisheye(rows, cols):
    """A smooth test image, coloured by angle and radius around the true centre of a rows x cols image."""
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    dx, dy = xx - cols / 2, yy - rows / 2
    img = np.zeros((rows, cols, 3), np.uint8)
    img[..., 0] = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 255
    img[..., 1] = np.clip(np.hypot(dx, dy) * 8, 0, 255)
    img[..., 2] = 200
    return img


NON_SQUARE = [(40, 56), (56, 40), (48, 64)]


@pytest.mark.parametrize('rows, cols', NON_SQUARE)
def test_non_square_dewarp_reads_the_largest_circle_that_fits(rows, cols):
    # The circle is centred in the image and its radius is half the shorter side
    shape, map_x, map_y = FisheyeWarping(polar_fisheye(rows, cols)).build_dewarp_mesh()
    radius = min(cols - cols // 2, rows - rows // 2)
    assert shape == (int(np.pi * radius), radius)
    assert 0 <= map_x.min() and map_x.max() <= cols  # never reads outside the image
    assert 0 <= map_y.min() and map_y.max() <= rows


@pytest.mark.parametrize('rows, cols', NON_SQUARE)
def test_non_square_rewarp_draws_the_circle_around_the_centre(rows, cols):
    frd = FisheyeWarping(polar_fisheye(rows, cols))
    frd.build_dewarp_mesh()
    _, _, mask = frd.build_rewarp_mesh()
    yy, xx = np.mgrid[0:rows, 0:cols]
    np.testing.assert_array_equal(mask > 0, np.hypot(xx - cols // 2, yy - rows // 2) <= min(rows, cols) / 2)


@pytest.mark.parametrize('rows, cols', NON_SQUARE)
def test_non_square_round_trip_restores_the_fisheye(rows, cols):
    img = polar_fisheye(rows, cols)
    frd = FisheyeWarping(img)
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
    result = frd.run_rewarp()

    yy, xx = np.mgrid[0:rows, 0:cols]
    radius = np.hypot(xx - cols / 2, yy - rows / 2)
    inside = (radius >= 4) & (radius < min(rows, cols) / 2 - 4)
    error = np.abs(result.astype(int) - img.astype(int))[inside]
    assert np.median(error) <= 5  # the same bound as for square images
