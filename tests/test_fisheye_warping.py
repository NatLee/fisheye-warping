import pickle

import numpy as np

from fisheyewarping import FisheyeWarping


def test_create_obj():
    FisheyeWarping(None, use_multiprocessing=True)


def test_build_dewarp_mesh(fisheye_img):
    # 64x64 image -> radius 32 -> panorama width int(2 * 16 * pi) = 100, height 32
    panorama_shape, map_x, map_y = FisheyeWarping(fisheye_img).build_dewarp_mesh()
    assert panorama_shape == (100, 32)
    assert map_x.shape == map_y.shape == (32, 100)
    assert map_x.max() > 0 and map_y.max() > 0


def test_run_dewarp_gives_panorama(fisheye_img):
    frd = FisheyeWarping(fisheye_img)
    frd.build_dewarp_mesh()
    assert frd.run_dewarp().shape == (32, 100, 3)


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
    assert map_x.shape == map_y.shape == mask.shape == (64, 64)
    assert mask[32, 32] == 255 and mask[0, 0] == 0  # inside vs. outside the circle
    result = frd.run_rewarp_with_mesh(panorama_img, save_path=None)
    assert result.shape == fisheye_img.shape


def test_saved_dewarp_mesh_can_be_loaded(fisheye_img, tmp_path):
    mesh = str(tmp_path / 'dewarp-mesh.pkl')
    builder = FisheyeWarping(fisheye_img)
    built = builder.build_dewarp_mesh(save_path=mesh)

    loader = FisheyeWarping(fisheye_img)
    loaded = loader.load_dewarp_mesh(mesh_path=mesh)
    assert loaded[0] == built[0] == (100, 32)
    np.testing.assert_array_equal(loader.run_dewarp(), builder.run_dewarp())


def test_load_dewarp_mesh_saved_by_version_1_0_x(fisheye_img, tmp_path):
    # 1.0.x saved only (map x, map y); the panorama shape comes from the map size
    _, map_x, map_y = FisheyeWarping(fisheye_img).build_dewarp_mesh()
    mesh = tmp_path / 'old-dewarp-mesh.pkl'
    mesh.write_bytes(pickle.dumps((map_x, map_y)))

    panorama_shape, _, _ = FisheyeWarping(fisheye_img).load_dewarp_mesh(mesh_path=str(mesh))
    assert panorama_shape == (100, 32)


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
