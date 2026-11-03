"""The ``fisheyewarping`` command."""
import argparse
import textwrap
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from .fisheyewarping import FisheyeWarping

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')

EXAMPLES = """\
examples:
  first time: dewarp a fisheye image and save both meshes
    fisheyewarping --fisheye_img_path fisheye.jpg \\
      --save_dewarp_mesh_path dewarp-mesh.pkl --save_rewarp_mesh_path rewarp-mesh.pkl

  next time: dewarp another image of the same size with the saved mesh
    fisheyewarping --fisheye_img_path fisheye.jpg --load_dewarp_mesh_path dewarp-mesh.pkl

  rewarp any panorama into a fisheye image (needs both meshes)
    fisheyewarping --panorama_img_path panorama.jpg \\
      --load_dewarp_mesh_path dewarp-mesh.pkl --load_rewarp_mesh_path rewarp-mesh.pkl
"""


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keeps the examples as written, and never breaks a path such as ./dewarp-output.png at its hyphen."""

    def _split_lines(self, text: str, width: int) -> List[str]:
        return textwrap.wrap(text, width, break_on_hyphens=False)


class _Stop(Exception):
    """Ends the run after printing why. The exit code stays 0, as in 1.0.x."""


def str2bool(value: str) -> bool:
    """Read `True`/`False` (also yes/no, 1/0, any case) as a bool; argparse `type=bool` treats any text as True."""
    text = str(value).strip().lower()
    if text in ('true', 'yes', 'y', '1'):
        return True
    if text in ('false', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'expected `True` or `False`, got `{value}`')


def build_parser() -> argparse.ArgumentParser:
    """The command line options of ``fisheyewarping``."""
    parser = argparse.ArgumentParser(
        prog='fisheyewarping',
        description='Dewarp a top-down fisheye image into a panorama, '
                    'and rewarp a panorama back into a fisheye image.',
        epilog=EXAMPLES,
        formatter_class=_HelpFormatter,
    )
    images = parser.add_argument_group('input images (give one or both)')
    images.add_argument('--fisheye_img_path', metavar='PATH',
                        help='Fisheye image to dewarp into a panorama (.jpg, .jpeg or .png).')
    images.add_argument('--panorama_img_path', metavar='PATH',
                        help='Panorama to rewarp into a fisheye image (.jpg, .jpeg or .png). On its own it needs '
                             'both --load_*_mesh_path options; with --fisheye_img_path it uses the meshes of that image.')

    meshes = parser.add_argument_group('meshes (the slow part; build once, then load)')
    meshes.add_argument('--save_dewarp_mesh_path', metavar='PATH',
                        help='Build the dewarp mesh from the fisheye image and save it to this file.')
    meshes.add_argument('--save_rewarp_mesh_path', metavar='PATH',
                        help='Build the rewarp mesh from the fisheye image and save it to this file.')
    meshes.add_argument('--load_dewarp_mesh_path', metavar='PATH',
                        help='Load a saved dewarp mesh instead of building one.')
    meshes.add_argument('--load_rewarp_mesh_path', metavar='PATH',
                        help='Load a saved rewarp mesh instead of building one.')
    meshes.add_argument('--use_multiprocessing', metavar='True|False', type=str2bool, default=True,
                        help='Build meshes with one worker process per CPU core. Default: %(default)s')

    outputs = parser.add_argument_group('outputs')
    outputs.add_argument('--panorama_output', metavar='PATH', default='./dewarp-output.png',
                         help='Where to write the panorama. Default: %(default)s')
    outputs.add_argument('--fisheye_output', metavar='PATH', default='./rewarp-output.png',
                         help='Where to write the fisheye image made from the panorama. Default: %(default)s')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the ``fisheyewarping`` command with ``argv``, or with the command line when it is None."""
    args = build_parser().parse_args(argv)
    try:
        _run(args)
    except _Stop as stop:
        print(f'----- {stop}')


def _run(args: argparse.Namespace) -> None:
    print('========Start to process========')
    print(f'----- Panorama output image path will be `{args.panorama_output}`.')
    print(f'----- Fisheye output image path will be `{args.fisheye_output}`.')

    fisheye_path = _check_image_path(args.fisheye_img_path, 'fisheye_img_path')
    panorama_path = _check_image_path(args.panorama_img_path, 'panorama_img_path')
    if fisheye_path is None and panorama_path is None:
        raise _Stop('Give `--fisheye_img_path` (fisheye to panorama), `--panorama_img_path` '
                    '(panorama to fisheye), or both. See `fisheyewarping --help`.')

    print('===================')
    print(f'----- Build meshes with multiprocessing: `{args.use_multiprocessing}`')
    print('===================')

    if fisheye_path and panorama_path:
        _fisheye_and_panorama(args, fisheye_path, panorama_path)
    elif fisheye_path:
        _fisheye_only(args, fisheye_path)
    else:
        _panorama_only(args, panorama_path)
    print('========End of this process========')


def _fisheye_and_panorama(args: argparse.Namespace, fisheye_path: Path, panorama_path: Path) -> None:
    print('----- Dewarp the fisheye image into a panorama, then rewarp the panorama into a fisheye image.')
    start = time.time()
    fisheye_img = _read_image(fisheye_path)
    panorama_img = _read_image(panorama_path)
    frd = FisheyeWarping(fisheye_img, use_multiprocessing=args.use_multiprocessing)

    _dewarp(frd, args)
    dewarped = time.time()
    print(f'-------Dewarping Task Completed------- ({dewarped - start:.3f} s)')

    if args.load_rewarp_mesh_path:
        _load_mesh(frd.load_rewarp_mesh, args.load_rewarp_mesh_path, 'load_rewarp_mesh_path')
    elif args.save_rewarp_mesh_path:
        _build_mesh(frd.build_rewarp_mesh, args.save_rewarp_mesh_path, 'save_rewarp_mesh_path')
    else:
        print('----- No `--load_rewarp_mesh_path` or `--save_rewarp_mesh_path` given, so the panorama is not rewarped.')
        print(f'-------All Task Completed------- ({time.time() - start:.3f} s)')
        return
    frd.run_rewarp_with_mesh(panorama_img, save_path=args.fisheye_output)
    end = time.time()
    print(f'-------Rewarping Task Completed------- ({end - dewarped:.3f} s)')
    print(f'-------All Task Completed------- ({end - start:.3f} s)')


def _fisheye_only(args: argparse.Namespace, fisheye_path: Path) -> None:
    print('----- Dewarp the fisheye image into a panorama.')
    start = time.time()
    frd = FisheyeWarping(_read_image(fisheye_path), use_multiprocessing=args.use_multiprocessing)
    _dewarp(frd, args)
    if args.save_rewarp_mesh_path:
        # Build the rewarp mesh from the same fisheye image, for rewarping panoramas later
        _build_mesh(frd.build_rewarp_mesh, args.save_rewarp_mesh_path, 'save_rewarp_mesh_path')
    print(f'-------Task Completed------- ({time.time() - start:.3f} s)')


def _panorama_only(args: argparse.Namespace, panorama_path: Path) -> None:
    print('----- Rewarp the panorama into a fisheye image.')
    start = time.time()
    frd = FisheyeWarping(None, use_multiprocessing=args.use_multiprocessing)
    panorama_img = _read_image(panorama_path)
    # Rewarping needs the panorama size, which is stored in the dewarp mesh
    if not args.load_dewarp_mesh_path:
        raise _Stop('Rewarping a panorama needs both saved meshes: give `--load_dewarp_mesh_path` and '
                    '`--load_rewarp_mesh_path`. The dewarp mesh holds the panorama size.')
    _load_mesh(frd.load_dewarp_mesh, args.load_dewarp_mesh_path, 'load_dewarp_mesh_path')
    if not args.load_rewarp_mesh_path:
        raise _Stop('Rewarping a panorama also needs `--load_rewarp_mesh_path`. Make one from a fisheye image '
                    'of the size you want with `--fisheye_img_path` and `--save_rewarp_mesh_path`.')
    _load_mesh(frd.load_rewarp_mesh, args.load_rewarp_mesh_path, 'load_rewarp_mesh_path')
    frd.run_rewarp_with_mesh(panorama_img, save_path=args.fisheye_output)
    print(f'-------All Tasks Completed------- ({time.time() - start:.3f} s)')


def _dewarp(frd: FisheyeWarping, args: argparse.Namespace) -> None:
    """Load or build the dewarp mesh, then write the panorama."""
    if args.load_dewarp_mesh_path:
        _load_mesh(frd.load_dewarp_mesh, args.load_dewarp_mesh_path, 'load_dewarp_mesh_path')
    elif args.save_dewarp_mesh_path:
        _build_mesh(frd.build_dewarp_mesh, args.save_dewarp_mesh_path, 'save_dewarp_mesh_path')
    else:
        raise _Stop('Give `--save_dewarp_mesh_path` to build the dewarp mesh (first time), '
                    'or `--load_dewarp_mesh_path` to load a saved one.')
    frd.run_dewarp(save_path=args.panorama_output)


def _check_file(path: Path, option: str) -> None:
    if not path.exists():
        raise _Stop(f'`--{option}` is `{path}`, which does not exist.')
    if not path.is_file():
        raise _Stop(f'`--{option}` is `{path}`, which is not a file.')


def _check_image_path(value: Optional[str], option: str) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    print(f'----- `--{option}`: `{path}`')
    _check_file(path, option)
    if path.suffix not in IMAGE_SUFFIXES:
        raise _Stop(f'`--{option}` must be a .jpg, .jpeg or .png image; `{path.name}` ends in `{path.suffix}`.')
    return path


def _read_image(path: Path) -> np.ndarray:
    return cv2.imread(path.as_posix())


def _load_mesh(load: Callable[..., object], value: str, option: str) -> None:
    print(f'----- Load the mesh in `--{option}`: `{value}`')
    _check_file(Path(value), option)
    load(mesh_path=value)


def _build_mesh(build: Callable[..., object], value: str, option: str) -> None:
    if Path(value).is_dir():
        raise _Stop(f'`--{option}` is `{value}`, which is a directory. Give a file name, such as `{value}/mesh.pkl`.')
    print(f'----- Build the mesh and save it to `--{option}`: `{value}`')
    build(save_path=value)
