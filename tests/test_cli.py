import subprocess
import sys

import cv2
import numpy as np
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
    return cli.main()


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


def test_use_multiprocessing_false_turns_it_off(workdir, monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError('multiprocessing was used although it was turned off')

    monkeypatch.setattr('fisheyewarping.fisheyewarping.mp.Pool', no_pool)
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--use_multiprocessing', 'False')
    assert (workdir / 'rewarp-mesh.pkl').is_file()


@pytest.mark.parametrize('text, expected', [
    ('True', True), ('true', True), ('1', True), ('yes', True),
    ('False', False), ('false', False), ('0', False), ('no', False),
])
def test_str2bool(text, expected):
    assert cli.str2bool(text) is expected


def test_use_multiprocessing_rejects_other_values(workdir, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, '--fisheye_img_path', './fisheye.png', '--use_multiprocessing', 'maybe')


class InlinePool:
    """Stands in for multiprocessing.Pool: same results, no worker processes, and it notes that it was used."""
    used = False

    def __init__(self, *args, **kwargs):
        InlinePool.used = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, func, jobs):
        return list(map(func, jobs))


@pytest.fixture
def inline_pool(monkeypatch):
    InlinePool.used = False
    monkeypatch.setattr('fisheyewarping.fisheyewarping.mp.Pool', InlinePool)
    return InlinePool


def build_both_meshes(monkeypatch):
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png')


def test_multiprocessing_is_on_by_default(workdir, monkeypatch, inline_pool):
    build_both_meshes(monkeypatch)
    assert inline_pool.used


def test_both_images_dewarp_and_rewarp_in_one_run(workdir, monkeypatch, inline_pool):
    run_cli(monkeypatch,
            '--save_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--save_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--panorama_img_path', './panorama.png')
    for name in ('dewarp-mesh.pkl', 'rewarp-mesh.pkl', 'dewarp-output.png', 'rewarp-output.png'):
        assert (workdir / name).is_file()
    first = cv2.imread(str(workdir / 'rewarp-output.png'))

    for name in ('dewarp-output.png', 'rewarp-output.png'):
        (workdir / name).unlink()
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--load_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--panorama_img_path', './panorama.png')
    assert (workdir / 'dewarp-output.png').is_file()
    np.testing.assert_array_equal(cv2.imread(str(workdir / 'rewarp-output.png')), first)


def test_both_images_without_a_rewarp_mesh_only_dewarp(workdir, monkeypatch, inline_pool):
    build_both_meshes(monkeypatch)
    (workdir / 'dewarp-output.png').unlink()
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--panorama_img_path', './panorama.png')
    assert (workdir / 'dewarp-output.png').is_file()
    assert not (workdir / 'rewarp-output.png').exists()


def test_output_paths_can_be_changed(workdir, monkeypatch, inline_pool):
    build_both_meshes(monkeypatch)
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--panorama_output', './my-panorama.jpg')
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--load_rewarp_mesh_path', './rewarp-mesh.pkl',
            '--panorama_img_path', './panorama.png',
            '--fisheye_output', './my-fisheye.png')
    assert cv2.imread(str(workdir / 'my-panorama.jpg')).shape == (32, 100, 3)
    assert cv2.imread(str(workdir / 'my-fisheye.png')).shape == (64, 64, 3)


# Each of these stops before writing anything and says what is wrong. The process still ends
# with exit code 0, as in 1.0.x (whether it should be non-zero is an open decision).
STOPS = {
    'no input image': ([], ['fisheye_img_path', 'panorama_img_path']),
    'fisheye image missing': (['--fisheye_img_path', './missing.png', '--save_dewarp_mesh_path', './d.pkl'],
                              ['fisheye_img_path']),
    'panorama image missing': (['--panorama_img_path', './missing.png', '--load_dewarp_mesh_path', './d.pkl'],
                               ['panorama_img_path']),
    'not an image suffix': (['--fisheye_img_path', './notes.txt', '--save_dewarp_mesh_path', './d.pkl'], ['.txt']),
    'no dewarp mesh option': (['--fisheye_img_path', './fisheye.png'],
                              ['load_dewarp_mesh_path', 'save_dewarp_mesh_path']),
    'dewarp mesh to load is missing': (['--fisheye_img_path', './fisheye.png', '--load_dewarp_mesh_path', './nope.pkl'],
                                       ['does not exist']),
    'dewarp mesh to load is a folder': (['--fisheye_img_path', './fisheye.png', '--load_dewarp_mesh_path', '.'],
                                        ['not a file']),
    'dewarp mesh to save is a folder': (['--fisheye_img_path', './fisheye.png', '--save_dewarp_mesh_path', '.'],
                                        ['save_dewarp_mesh_path', 'directory']),
    'panorama without rewarp mesh': (['--panorama_img_path', './panorama.png', '--load_dewarp_mesh_path', './mesh/d.pkl'],
                                     ['load_rewarp_mesh_path']),
    'panorama with a missing rewarp mesh': (['--panorama_img_path', './panorama.png',
                                             '--load_dewarp_mesh_path', './mesh/d.pkl',
                                             '--load_rewarp_mesh_path', './nope.pkl'], ['does not exist']),
}


@pytest.mark.parametrize('args, expected', list(STOPS.values()), ids=list(STOPS))
def test_stops_early_and_says_why(args, expected, workdir, monkeypatch, capsys, inline_pool):
    (workdir / 'notes.txt').write_text('not an image')
    (workdir / 'mesh').mkdir()
    monkeypatch.chdir(workdir / 'mesh')
    build = ['--save_dewarp_mesh_path', './d.pkl', '--save_rewarp_mesh_path', './r.pkl',
                            '--fisheye_img_path', '../fisheye.png', '--use_multiprocessing', 'False']
    run_cli(monkeypatch, *build)
    monkeypatch.chdir(workdir)
    capsys.readouterr()
    before = sorted(p.name for p in workdir.iterdir())

    assert run_cli(monkeypatch, *args) is None  # returns normally: exit code 0
    out = capsys.readouterr().out
    for word in expected:
        assert word in out
    assert sorted(p.name for p in workdir.iterdir()) == before  # nothing written


def test_save_rewarp_mesh_path_that_is_a_folder_stops_after_dewarp(workdir, monkeypatch, capsys, inline_pool):
    assert run_cli(monkeypatch,
                   '--fisheye_img_path', './fisheye.png',
                   '--save_dewarp_mesh_path', './d.pkl',
                   '--save_rewarp_mesh_path', '.') is None
    assert 'save_rewarp_mesh_path' in capsys.readouterr().out
    assert (workdir / 'd.pkl').is_file()


def test_options_and_defaults_are_unchanged():
    # Scripts depend on these names and defaults; changing one is a breaking change
    parser = cli.build_parser()
    assert vars(parser.parse_args([])) == {
        'fisheye_img_path': None,
        'panorama_img_path': None,
        'save_dewarp_mesh_path': None,
        'save_rewarp_mesh_path': None,
        'load_dewarp_mesh_path': None,
        'load_rewarp_mesh_path': None,
        'panorama_output': './dewarp-output.png',
        'fisheye_output': './rewarp-output.png',
        'use_multiprocessing': True,
    }
    help_text = parser.format_help()
    for option in ('--fisheye_img_path', '--panorama_img_path', '--save_dewarp_mesh_path', '--save_rewarp_mesh_path',
                   '--load_dewarp_mesh_path', '--load_rewarp_mesh_path', '--panorama_output', '--fisheye_output',
                   '--use_multiprocessing', 'examples:'):
        assert option in help_text


def test_main_takes_arguments_directly(workdir, inline_pool):
    cli.main(['--fisheye_img_path', './fisheye.png', '--save_dewarp_mesh_path', './dewarp-mesh.pkl'])
    assert (workdir / 'dewarp-output.png').is_file()


def test_both_images_without_a_rewarp_mesh_says_so(workdir, monkeypatch, capsys, inline_pool):
    build_both_meshes(monkeypatch)
    capsys.readouterr()
    run_cli(monkeypatch,
            '--load_dewarp_mesh_path', './dewarp-mesh.pkl',
            '--fisheye_img_path', './fisheye.png',
            '--panorama_img_path', './panorama.png')
    assert 'not rewarped' in capsys.readouterr().out


# Where 1.0.x crashed with a traceback (exit code 1), the command now ends with one clear line
# and still exits with 1: SystemExit with a message makes Python print it and exit with 1.

def test_unreadable_image_fails_with_a_clear_message(workdir, monkeypatch):
    (workdir / 'broken.png').write_bytes(b'not really a png')
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch, '--fisheye_img_path', './broken.png', '--save_dewarp_mesh_path', './d.pkl')
    assert 'cannot read `broken.png` as an image' in str(stop.value.code)
    assert not (workdir / 'd.pkl').exists()


def test_swapped_meshes_fail_with_a_clear_message(workdir, monkeypatch, inline_pool):
    build_both_meshes(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        run_cli(monkeypatch,
                '--load_dewarp_mesh_path', './rewarp-mesh.pkl',
                '--load_rewarp_mesh_path', './dewarp-mesh.pkl',
                '--panorama_img_path', './panorama.png')
    assert '--load_dewarp_mesh_path' in str(stop.value.code)
    assert 'swapped' in str(stop.value.code)
    assert not (workdir / 'rewarp-output.png').exists()


def test_errors_end_with_exit_code_1(workdir):
    (workdir / 'broken.png').write_bytes(b'not really a png')
    code = 'from fisheyewarping.cli import main; main()'
    proc = subprocess.run([sys.executable, '-c', code, '--fisheye_img_path', 'broken.png',
                           '--save_dewarp_mesh_path', 'd.pkl'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines() == ['fisheyewarping: error: OpenCV cannot read `broken.png` as an image. '
                                                'Is the file complete?']
