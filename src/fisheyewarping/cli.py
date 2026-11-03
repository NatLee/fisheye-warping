"""The ``fisheyewarping`` command."""
import argparse
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .fisheyewarping import FisheyeWarping

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
#: The suffixes above as text for help and messages: '.jpg, .jpeg or .png'
_SUFFIXES_TEXT = ', '.join(IMAGE_SUFFIXES[:-1]) + ' or ' + IMAGE_SUFFIXES[-1]

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


class _UsageError(Exception):
    """Options that cannot work together. main() prints the usage and exits with 2, as argparse does."""


class _Failure(Exception):
    """The options are fine, but a file is missing or cannot be used. main() exits with 1."""


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
        formatter_class=argparse.RawDescriptionHelpFormatter,  # keeps the examples as written
    )
    images = parser.add_argument_group('input images (give one or both)')
    images.add_argument('--fisheye_img_path', metavar='PATH',
                        help=f'Fisheye image to dewarp into a panorama ({_SUFFIXES_TEXT}).')
    images.add_argument('--panorama_img_path', metavar='PATH',
                        help=f'Panorama to rewarp into a fisheye image ({_SUFFIXES_TEXT}). On its own it needs '
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
                        help='Default: %(default)s. Build meshes with one worker process per CPU core.')

    # Defaults come first in the help. argparse wraps help text and may break a path at a hyphen;
    # at the start of the text, the paths stay whole on terminals 55 columns wide or wider.
    outputs = parser.add_argument_group('outputs')
    outputs.add_argument('--panorama_output', metavar='PATH', default='./dewarp-output.png',
                         help='Default: %(default)s. Where to write the panorama.')
    outputs.add_argument('--fisheye_output', metavar='PATH', default='./rewarp-output.png',
                         help='Default: %(default)s. Where to write the fisheye image made from the panorama.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the ``fisheyewarping`` command with ``argv``, or with the command line when it is None.

    This is the one place that decides the exit code: 0 when it worked, 1 when a file is missing
    or cannot be used, 2 for wrong or missing options (argparse's own convention, with the usage
    line). Either way nothing more is written once the problem is found.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _check_options(args)
        _run(args)
    except _UsageError as error:
        parser.error(str(error))
    except _Failure as error:
        raise SystemExit(f'{parser.prog}: error: {error}') from None


def _check_options(args: argparse.Namespace) -> None:
    """Refuse option combinations that cannot work, before any file is read or written."""
    if not args.fisheye_img_path and not args.panorama_img_path:
        raise _UsageError('Give `--fisheye_img_path` (fisheye to panorama), `--panorama_img_path` '
                          '(panorama to fisheye), or both. See `fisheyewarping --help`.')
    if args.fisheye_img_path and not (args.load_dewarp_mesh_path or args.save_dewarp_mesh_path):
        raise _UsageError('Give `--save_dewarp_mesh_path` to build the dewarp mesh (first time), '
                          'or `--load_dewarp_mesh_path` to load a saved one.')
    if args.panorama_img_path and not args.fisheye_img_path:
        # Rewarping needs the panorama size, which is stored in the dewarp mesh
        if not args.load_dewarp_mesh_path:
            raise _UsageError('Rewarping a panorama needs both saved meshes: give `--load_dewarp_mesh_path` and '
                              '`--load_rewarp_mesh_path`. The dewarp mesh holds the panorama size.')
        if not args.load_rewarp_mesh_path:
            raise _UsageError('Rewarping a panorama also needs `--load_rewarp_mesh_path`. Make one from a fisheye '
                              'image of the size you want with `--fisheye_img_path` and `--save_rewarp_mesh_path`.')


def _run(args: argparse.Namespace) -> None:
    print('========Start to process========')
    print(f'----- Panorama output image path will be `{args.panorama_output}`.')
    print(f'----- Fisheye output image path will be `{args.fisheye_output}`.')

    fisheye_path = _check_image_path(args.fisheye_img_path, 'fisheye_img_path')
    panorama_path = _check_image_path(args.panorama_img_path, 'panorama_img_path')

    print('===================')
    print(f'----- Build meshes with multiprocessing: `{args.use_multiprocessing}`')
    print('===================')

    if fisheye_path and panorama_path:
        _fisheye_and_panorama(args, fisheye_path, panorama_path)
    elif fisheye_path:
        _fisheye_only(args, fisheye_path)
    elif panorama_path:  # always true here: _check_options asks for at least one image
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
    _load_mesh(frd.load_dewarp_mesh, args.load_dewarp_mesh_path, 'load_dewarp_mesh_path')
    _load_mesh(frd.load_rewarp_mesh, args.load_rewarp_mesh_path, 'load_rewarp_mesh_path')
    frd.run_rewarp_with_mesh(panorama_img, save_path=args.fisheye_output)
    print(f'-------All Tasks Completed------- ({time.time() - start:.3f} s)')


def _dewarp(frd: FisheyeWarping, args: argparse.Namespace) -> None:
    """Load or build the dewarp mesh (_check_options made sure one option is given), then write the panorama."""
    if args.load_dewarp_mesh_path:
        _load_mesh(frd.load_dewarp_mesh, args.load_dewarp_mesh_path, 'load_dewarp_mesh_path')
    else:
        _build_mesh(frd.build_dewarp_mesh, args.save_dewarp_mesh_path, 'save_dewarp_mesh_path')
    frd.run_dewarp(save_path=args.panorama_output)


def _check_file(path: Path, option: str) -> None:
    if not path.exists():
        raise _Failure(f'`--{option}` is `{path}`, which does not exist.')
    if not path.is_file():
        raise _Failure(f'`--{option}` is `{path}`, which is not a file.')


def _check_image_path(value: Optional[str], option: str) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    print(f'----- `--{option}`: `{path}`')
    _check_file(path, option)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise _Failure(f'`--{option}` must be a {_SUFFIXES_TEXT} image; `{path.name}` ends in `{path.suffix}`.')
    return path


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(path.as_posix())
    if img is None:
        raise _Failure(f'OpenCV cannot read `{path}` as an image. Is the file complete?')
    return img


def _load_mesh(load: Callable[..., object], value: str, option: str) -> None:
    print(f'----- Load the mesh in `--{option}`: `{value}`')
    _check_file(Path(value), option)
    try:
        load(mesh_path=value)
    except (ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise _Failure(f'`--{option}`: {exc}') from exc


def _build_mesh(build: Callable[..., object], value: str, option: str) -> None:
    if Path(value).is_dir():
        raise _Failure(f'`--{option}` is `{value}`, which is a directory. Give a file name, such as `{value}/mesh.pkl`.')
    print(f'----- Build the mesh and save it to `--{option}`: `{value}`')
    build(save_path=value)
