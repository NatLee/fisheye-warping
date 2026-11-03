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


def test_panorama_only_rewarps_with_both_loaded_meshes(workdir, monkeypatch):
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png')
    # README: "In the next time, you just load the meshes and feel free to go."
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--load_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--panorama_img_path', './panorama.png')
    output = cv2.imread(str(workdir / 'rewarp-output.png'))
    assert output.shape == (64, 64, 3)


def test_panorama_only_without_dewarp_mesh_explains_what_is_missing(workdir, monkeypatch, capsys):
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png')
    capsys.readouterr()
    run_cli(monkeypatch,
            '--load_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--panorama_img_path', './panorama.png')
    assert 'load_dewarp_mesh_path' in capsys.readouterr().out
    assert not (workdir / 'rewarp-output.png').exists()
