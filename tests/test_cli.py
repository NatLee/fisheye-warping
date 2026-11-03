import sys

import cv2
import pytest

from fisheyewarping import cli


@pytest.fixture
def workdir(tmp_path, monkeypatch, fisheye_img, panorama_img):
    """A temp working dir holding fisheye.png and panorama.png, like the README examples."""
    cv2.imwrite(str(tmp_path / 'fisheye.png'), fisheye_img)
    cv2.imwrite(str(tmp_path / 'panorama.png'), panorama_img)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['fisheyewarping', *args])
    cli.main()


def test_fisheye_only_builds_both_meshes(workdir, monkeypatch):
    # README: "build 2 mesh files for dewarping and rewarping by using one fisheye image"
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png')
    assert (workdir / 'dewarp-mesh.pkl').is_file()
    assert (workdir / 'rewarp-mesh.pkl').is_file()
    assert (workdir / 'dewarp-output.png').is_file()
