"""
Dewarp a top-down fisheye image into a panorama, and rewarp a panorama back into a fisheye image.

Both directions use a *mesh*: for every output pixel it says which input pixel to read, and
``cv2.remap`` does the reading. Building a mesh is slow, so it can be saved to a file and loaded
again for any other image of the same size.
"""
import multiprocessing as mp
import os
import pickle
import time
import warnings
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

#: ``(width, height)`` of the panorama, the order ``cv2.resize`` takes
PanoramaShape = Tuple[int, int]
#: A file path: ``str`` or ``pathlib.Path``
PathLike = Union[str, os.PathLike]
#: What ``build_dewarp_mesh`` returns and saves: ``(panorama shape, map x, map y)``
DewarpMesh = Tuple[PanoramaShape, np.ndarray, np.ndarray]
#: What ``build_rewarp_mesh`` returns and saves: ``(map x, map y, mask)``
RewarpMesh = Tuple[np.ndarray, np.ndarray, np.ndarray]

Vector = Union[Sequence[float], np.ndarray]
Point = Tuple[int, int]


class MeshNotReadyError(RuntimeError, AssertionError):
    """A mesh is needed but has not been built or loaded yet.

    It is also an ``AssertionError``, which is what version 1.0.x raised here, so code that
    catches that keeps working. Unlike an ``assert``, it is still raised under ``python -O``.
    The ``AssertionError`` base can go in a major release, once callers catch this class instead.
    """


def unit_vector(vector: Vector) -> np.ndarray:
    """Return ``vector`` scaled to length 1. A zero vector gives NaNs."""
    vector = np.asarray(vector)
    norm = np.linalg.norm(vector)
    if norm == 0:
        # 0 / 0 is NaN on purpose. Silence that warning here only, not for the whole program.
        with np.errstate(invalid='ignore'):
            return vector / norm
    return vector / norm


def angle_between(v1: Vector, v2: Vector) -> float:
    """Return the angle in radians between vectors ``v1`` and ``v2``, from 0 to pi.

    >>> float(angle_between((1, 0, 0), (0, 1, 0)))
    1.5707963267948966
    >>> float(angle_between((1, 0, 0), (1, 0, 0)))
    0.0
    >>> float(angle_between((1, 0, 0), (-1, 0, 0)))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_map(points: Tuple[Point, np.ndarray, np.ndarray, float]) -> Tuple[Point, Optional[float], float]:
    """Place one fisheye pixel on the panorama; one job of the rewarp mesh.

    ``points`` is ``((x, y), center, top_point, radius)``. Returns ``((x, y), share, distance)``:
    ``share`` is the angle between the pixel and ``top_point`` seen from ``center``, as a share of
    a full turn (0 to 0.5), or None when the pixel is outside the circle; ``distance`` is how far
    the pixel is from ``center``.
    """
    point, center, top_point, radius = points
    point_np = np.asarray(point)
    distance = np.linalg.norm(point_np - center)
    share = None
    if distance <= radius:
        angle = angle_between(top_point - center, point_np - center)
        share = angle / (2 * np.pi)
    return point, share, distance


# Where the panorama comes from in a fisheye image, see `_panorama_geometry`
_Geometry = NamedTuple('_Geometry', [
    ('width', int), ('height', int),
    ('inner_radius', int), ('outer_radius', int),
    ('center_x', int), ('center_y', int),
])


def _panorama_geometry(fisheye_shape: Sequence[int]) -> _Geometry:
    """Size of the panorama for a fisheye image, and the circle it is read from.

    The circle sits in the middle of the image and its radius is half the image width. The
    panorama is as tall as that radius and as wide as the circle's circumference at half the
    radius.
    """
    rows, cols = fisheye_shape[:2]
    center_x, center_y = int(cols / 2), int(rows / 2)
    inner_radius = 0  # the panorama reaches all the way to the centre
    outer_radius = int(cols) - center_x
    width = int(2.0 * ((outer_radius + inner_radius) / 2) * np.pi)
    height = outer_radius - inner_radius
    return _Geometry(width, height, inner_radius, outer_radius, center_x, center_y)


def _dewarp_point(job: Tuple[int, int, _Geometry]) -> Tuple[Point, int, int]:
    """Find the fisheye pixel that panorama pixel ``(y, x)`` comes from; one job of the dewarp mesh."""
    y, x, (width, height, inner_radius, outer_radius, center_x, center_y) = job
    radius = (float(y) / float(height)) * (outer_radius - inner_radius) + inner_radius
    theta = (float(x) / float(width)) * 2.0 * np.pi
    source_x = int(center_x + radius * np.sin(theta))
    source_y = int(center_y + radius * np.cos(theta))
    return (y, x), source_x, source_y


def _run_jobs(func: Callable[[Any], Any], jobs: Iterable[Any], use_multiprocessing: bool, name: str) -> Iterable[Any]:
    """Run ``func`` on every job, in worker processes or right here. Results keep the order of the jobs."""
    if not use_multiprocessing:
        return map(func, jobs)
    jobs = list(jobs)
    start = time.time()
    print('-------Start multi pixel mapping for {}-------'.format(name))
    with mp.Pool() as pool:
        results = pool.map(func, jobs)
    print('--------Mapping Completed-------- ({:0.3f} s)'.format(time.time() - start))
    return results


def _panorama_shape_of(map_x: np.ndarray) -> PanoramaShape:
    """``(width, height)`` of the panorama that a dewarp map produces: the size of the map."""
    rows, cols = map_x.shape[:2]
    return cols, rows


def _build_dewarp_maps(img: np.ndarray, use_multiprocessing: bool) -> Tuple[np.ndarray, np.ndarray]:
    geometry = _panorama_geometry(img.shape)
    map_x = np.zeros((geometry.height, geometry.width), np.float32)
    map_y = np.zeros((geometry.height, geometry.width), np.float32)
    # The last row and column are never filled and stay 0, as in 1.0.x
    rows = tqdm(range(geometry.height - 1), desc='Building dewarp mesh')
    jobs = ((y, x, geometry) for y in rows for x in range(geometry.width - 1))
    for (y, x), source_x, source_y in _run_jobs(_dewarp_point, jobs, use_multiprocessing, 'dewarping'):
        map_x[y, x] = source_x
        map_y[y, x] = source_y
    return map_x, map_y


# Where a rewarped image puts the circle, see `_rewarp_circle`
_Circle = NamedTuple('_Circle', [
    ('center', np.ndarray), ('top_point', np.ndarray), ('radius', float), ('split_column', int),
])


def _rewarp_circle(rows: int, cols: int) -> _Circle:
    """The circle of a rewarped image, and the column where its mirrored right half starts.

    Kept from 1.0.x so that no output changes: it builds the centre ``(x, y)`` from
    ``(rows, cols)``, takes half the height as the radius and splits the halves at the middle
    row count. That is right for square images only; other sizes come out off-centre (README,
    "Limitations"). Fixing non-square images means changing this one function, and the 1.0.x
    reference in tests/test_reference_geometry.py with it.
    """
    center = np.asarray([int(rows / 2), int(cols / 2)])
    top_point = np.asarray([int(rows / 2), 0])
    return _Circle(center, top_point, rows / 2, int(rows / 2) + 1)


def _build_rewarp_maps(img: np.ndarray, panorama_width: int, use_multiprocessing: bool) -> RewarpMesh:
    rows, cols = img.shape[:2]
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    center, top_point, radius, _ = _rewarp_circle(rows, cols)
    jobs = (((x, y), center, top_point, radius)
            for y in tqdm(range(rows), desc='Building rewarp mesh') for x in range(cols))
    for (x, y), share, distance in _run_jobs(angle_map, jobs, use_multiprocessing, 'rewarping'):
        if share is not None:
            map_x[y, x] = share * panorama_width
            map_y[y, x] = distance
            mask[y, x] = 255
    return map_x, map_y, mask


def _rotate_180(img: np.ndarray) -> np.ndarray:
    """Turn an image upside down around ``(w/2, h/2)``, as 1.0.x does.

    Not quite ``cv2.rotate``: pixel ``x`` lands on ``w - x``, so the input's first row and
    column drop out and the output's first row and column are black.
    """
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1)
    return cv2.warpAffine(img, matrix, (w, h))


def _save_target(save_path: Optional[PathLike]) -> Optional[str]:
    """The file to save to, or None to save nothing. None and '' save nothing, as in 1.0.x."""
    if isinstance(save_path, os.PathLike):
        save_path = os.fspath(save_path)
    return save_path if isinstance(save_path, str) and save_path else None


def _save_pickle(path: PathLike, data: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def _load_pickle(path: PathLike) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_image(save_path: Optional[PathLike], img: np.ndarray) -> None:
    target = _save_target(save_path)
    if target and not cv2.imwrite(target, img):
        # OpenCV only returns False, for example when the folder does not exist
        warnings.warn(f'OpenCV could not write `{target}`. Does its folder exist?', stacklevel=3)


def _require_image(img: Optional[np.ndarray], needed_by: str) -> np.ndarray:
    if img is None:
        raise TypeError(f'{needed_by} needs an image but got None. '
                        '(cv2.imread returns None when it cannot read a file.)')
    return img


def _is_map(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 2


def _is_size(value: Any) -> bool:
    return (isinstance(value, (tuple, list)) and len(value) == 2
            and all(isinstance(v, (int, np.integer)) for v in value))


def _same_size_maps(maps: Sequence[Any]) -> bool:
    return all(_is_map(m) for m in maps) and len({m.shape for m in maps}) == 1


def _as_dewarp_mesh(data: Any) -> Optional[DewarpMesh]:
    """``data`` unpickled from a file, as a dewarp mesh; None when it is not one."""
    if not isinstance(data, (tuple, list)):
        return None
    if len(data) == 3 and _is_size(data[0]) and _same_size_maps(data[1:]):
        return data[0], data[1], data[2]
    if len(data) == 2 and _same_size_maps(data):
        # 1.0.x saved only (map x, map y). Read so that those files keep working; this branch
        # can go in a major release that no longer loads 1.0.x mesh files.
        return _panorama_shape_of(data[0]), data[0], data[1]
    return None


def _as_rewarp_mesh(data: Any) -> Optional[RewarpMesh]:
    """``data`` unpickled from a file, as a rewarp mesh; None when it is not one."""
    if isinstance(data, (tuple, list)) and len(data) == 3 and _same_size_maps(data):
        return data[0], data[1], data[2]
    return None


#: How to read each kind of mesh file
_MESH_READERS: Dict[str, Callable[[Any], Any]] = {'dewarp': _as_dewarp_mesh, 'rewarp': _as_rewarp_mesh}


def _load_mesh(mesh_path: PathLike, kind: str) -> Any:
    """Read a pickled mesh of ``kind``, a key of ``_MESH_READERS``. Says so when the file holds another kind."""
    data = _load_pickle(mesh_path)
    mesh = _MESH_READERS[kind](data)
    if mesh is not None:
        return mesh
    for other, read in _MESH_READERS.items():
        if other != kind and read(data) is not None:
            raise ValueError(f'`{mesh_path}` holds a {other} mesh, not a {kind} mesh. '
                             'Are the dewarp and rewarp mesh files swapped?')
    raise ValueError(f'`{mesh_path}` is not a {kind} mesh saved by FisheyeWarping.')


class FisheyeWarping:
    """Turn a top-down fisheye image into a panorama (dewarp) and a panorama into a fisheye image (rewarp).

    ``img`` is the fisheye image as OpenCV reads it (``cv2.imread``), or None when you only
    rewarp with meshes loaded from files. ``use_multiprocessing`` builds meshes with one worker
    process per CPU core; in a script, call it under ``if __name__ == '__main__':``.
    """

    def __init__(self, img: Optional[np.ndarray], use_multiprocessing: bool = False) -> None:
        self.img = img
        self.use_multiprocessing = use_multiprocessing

        self._dewarp_map_x = None  # type: Optional[np.ndarray]
        self._dewarp_map_y = None  # type: Optional[np.ndarray]
        self._rewarp_map_x = None  # type: Optional[np.ndarray]
        self._rewarp_map_y = None  # type: Optional[np.ndarray]
        self._rewarp_mask = None  # type: Optional[np.ndarray]
        self._panorama_shape = None  # type: Optional[PanoramaShape]

    # ---- dewarp: fisheye -> panorama ---------------------------------------------------

    def build_dewarp_mesh(self, save_path: Optional[PathLike] = None) -> DewarpMesh:
        """Build the mesh that turns this fisheye image into a panorama; save it when ``save_path`` is given.

        Returns ``(panorama shape, map x, map y)``, where the panorama shape is ``(width, height)``.
        """
        img = self._require_fisheye('build_dewarp_mesh()')
        self._dewarp_map_x, self._dewarp_map_y = _build_dewarp_maps(img, self.use_multiprocessing)
        self._panorama_shape = _panorama_shape_of(self._dewarp_map_x)
        mesh = (self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y)
        target = _save_target(save_path)
        if target:
            _save_pickle(target, mesh)
        print(f'Dewarp Map X shape -> {self._dewarp_map_x.shape}')
        print(f'Dewarp Map Y shape -> {self._dewarp_map_y.shape}')
        return mesh

    def load_dewarp_mesh(self, mesh_path: PathLike) -> DewarpMesh:
        """Load a mesh saved by ``build_dewarp_mesh`` (also the 1.0.x format). Returns it like ``build_dewarp_mesh``.

        Raises ``ValueError`` when the file holds something else, such as a rewarp mesh.
        """
        self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y = _load_mesh(mesh_path, 'dewarp')
        return self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y

    def run_dewarp(self, save_path: Optional[PathLike] = None) -> np.ndarray:
        """Dewarp this fisheye image into a panorama; save it when ``save_path`` is given."""
        self._require_dewarp_mesh()
        result = self.dewarp(self._require_fisheye('run_dewarp()'), flip=True)
        print(f'Panorama shape is `{result.shape}`')
        _save_image(save_path, result)
        return result

    def dewarp(self, img: np.ndarray, flip: bool = False) -> np.ndarray:
        """Apply the dewarp mesh to ``img``. ``flip`` turns the result upside down, as ``run_dewarp`` does."""
        map_x, map_y = self._require_dewarp_mesh()
        output = cv2.remap(_require_image(img, 'dewarp()'), map_x, map_y, cv2.INTER_LINEAR)
        if flip:
            output = _rotate_180(output)
        return output

    # ---- rewarp: panorama -> fisheye ---------------------------------------------------

    def build_rewarp_mesh(self, save_path: Optional[PathLike] = None) -> RewarpMesh:
        """Build the mesh that turns a panorama back into a fisheye image of this size; save it when ``save_path`` is given.

        Needs the dewarp mesh first (built or loaded). Returns ``(map x, map y, mask)``.
        """
        img = self._require_fisheye('build_rewarp_mesh()')
        dewarp_map_x, _ = self._require_dewarp_mesh()
        panorama_width = dewarp_map_x.shape[1]
        mesh = _build_rewarp_maps(img, panorama_width, self.use_multiprocessing)
        self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask = mesh
        target = _save_target(save_path)
        if target:
            _save_pickle(target, mesh)
        print(f'Rewarp Map X shape -> {mesh[0].shape}')
        print(f'Rewarp Map Y shape -> {mesh[1].shape}')
        print(f'Rewarp Map MASK shape -> {mesh[2].shape}')
        return mesh

    def load_rewarp_mesh(self, mesh_path: PathLike) -> RewarpMesh:
        """Load a mesh saved by ``build_rewarp_mesh``. Returns it like ``build_rewarp_mesh``.

        Raises ``ValueError`` when the file holds something else, such as a dewarp mesh.
        """
        self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask = _load_mesh(mesh_path, 'rewarp')
        return self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask

    def run_rewarp(self, save_path: Optional[PathLike] = None) -> np.ndarray:
        """Dewarp this fisheye image and rewarp the result: a round trip back to a fisheye image."""
        self._require_dewarp_mesh()
        dewarp_result = self.dewarp(self._require_fisheye('run_rewarp()'), flip=False)
        result = self.rewarp(dewarp_result, flip=True)
        print(f'Fisheye image shape is `{result.shape}`')
        _save_image(save_path, result)
        return result

    def run_rewarp_with_mesh(self, panorama_img: np.ndarray, save_path: Optional[PathLike] = None) -> np.ndarray:
        """Rewarp any panorama into a fisheye image; save it when ``save_path`` is given.

        Needs both meshes: the dewarp mesh tells the panorama size, the rewarp mesh does the work.
        The panorama is resized to the size the meshes were built for.
        """
        if self._panorama_shape is None:
            raise MeshNotReadyError('Rewarping a panorama needs the panorama size from the dewarp mesh: '
                                    'call build_dewarp_mesh() or load_dewarp_mesh() first.')
        panorama_img = cv2.resize(_require_image(panorama_img, 'run_rewarp_with_mesh()'), self._panorama_shape)
        panorama_img = _rotate_180(panorama_img)
        result = self.rewarp(panorama_img, flip=True)
        print(f'Fisheye image shape is `{result.shape}`')
        _save_image(save_path, result)
        return result

    def half_rewarp_map(self, panorama_img: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remap ``panorama_img`` and its mirror image with maps ``x`` and ``y``: the two halves of the circle."""
        left_output = cv2.remap(panorama_img, x, y, cv2.INTER_LINEAR)
        right_output = cv2.remap(cv2.flip(panorama_img, 1), x, y, cv2.INTER_LINEAR)
        return left_output, right_output

    def rewarp(self, panorama_img: np.ndarray, flip: bool = False) -> np.ndarray:
        """Apply the rewarp mesh to ``panorama_img``. ``flip`` turns the result upside down."""
        map_x, map_y, mask = self._require_rewarp_mesh()
        panorama_img = _require_image(panorama_img, 'rewarp()')
        canvas, right_output = self.half_rewarp_map(panorama_img, map_x, map_y)
        # The angle only covers half a turn, so the right half comes from the mirrored panorama.
        # Column slices and a 2-D mask work for any number of channels and any pixel type.
        split = _rewarp_circle(*mask.shape[:2]).split_column
        canvas[:, split:] = right_output[:, split:]
        canvas[mask == 0] = 0  # black outside the circle
        if flip:
            canvas = _rotate_180(canvas)
        return canvas

    # ---- checks ---------------------------------------------------------------------------

    def _require_fisheye(self, needed_by: str) -> np.ndarray:
        if self.img is None:
            raise TypeError(f'{needed_by} needs the fisheye image, but FisheyeWarping got None. '
                            '(cv2.imread returns None when it cannot read a file.) Without an image you can '
                            'still load both meshes and call run_rewarp_with_mesh().')
        return self.img

    def _require_dewarp_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._dewarp_map_x is None or self._dewarp_map_y is None:
            raise MeshNotReadyError('There is no dewarp mesh yet: call build_dewarp_mesh() or load_dewarp_mesh() first.')
        return self._dewarp_map_x, self._dewarp_map_y

    def _require_rewarp_mesh(self) -> RewarpMesh:
        if self._rewarp_map_x is None or self._rewarp_map_y is None or self._rewarp_mask is None:
            raise MeshNotReadyError('There is no rewarp mesh yet: call build_rewarp_mesh() or load_rewarp_mesh() first.')
        return self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask
