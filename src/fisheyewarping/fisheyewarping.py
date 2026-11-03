"""
Dewarp a top-down fisheye image into a panorama, and rewarp a panorama back into a fisheye image.

Both directions use a *mesh*: for every output pixel it says which input pixel to read, and
``cv2.remap`` does the reading. Building a mesh is slow, so it can be saved to a file and loaded
again for any other image of the same size.
"""
import multiprocessing as mp
import pickle
import time
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

#: ``(width, height)`` of the panorama, the order ``cv2.resize`` takes
PanoramaShape = Tuple[int, int]
#: What ``build_dewarp_mesh`` returns and saves: ``(panorama shape, map x, map y)``
DewarpMesh = Tuple[PanoramaShape, np.ndarray, np.ndarray]
#: What ``build_rewarp_mesh`` returns and saves: ``(map x, map y, mask)``
RewarpMesh = Tuple[np.ndarray, np.ndarray, np.ndarray]

Vector = Union[Sequence[float], np.ndarray]
Point = Tuple[int, int]


def unit_vector(vector: Vector) -> np.ndarray:
    """Return ``vector`` scaled to length 1. A zero vector gives NaNs."""
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


def _build_rewarp_maps(img: np.ndarray, panorama_width: int, use_multiprocessing: bool) -> RewarpMesh:
    rows, cols = img.shape[:2]
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    # 1.0.x builds the centre (x, y) from (rows, cols) and takes half the height as the radius.
    # That is only right for square images; kept as it is so that no output changes.
    center = np.asarray([int(rows / 2), int(cols / 2)])
    top_point = np.asarray([int(rows / 2), 0])
    radius = rows / 2
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


def _wants_saving(save_path: Optional[str]) -> bool:
    return bool(save_path) and isinstance(save_path, str)


def _save_pickle(path: str, data: Any) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def _load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_image(save_path: Optional[str], img: np.ndarray) -> None:
    if _wants_saving(save_path):
        cv2.imwrite(save_path, img)


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

    def build_dewarp_mesh(self, save_path: Optional[str] = None) -> DewarpMesh:
        """Build the mesh that turns this fisheye image into a panorama; save it when ``save_path`` is given.

        Returns ``(panorama shape, map x, map y)``, where the panorama shape is ``(width, height)``.
        """
        self._dewarp_map_x, self._dewarp_map_y = _build_dewarp_maps(self.img, self.use_multiprocessing)
        rows, cols = self._dewarp_map_x.shape
        self._panorama_shape = (cols, rows)
        mesh = (self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y)
        if _wants_saving(save_path):
            _save_pickle(save_path, mesh)
        print(f'Dewarp Map X shape -> {self._dewarp_map_x.shape}')
        print(f'Dewarp Map Y shape -> {self._dewarp_map_y.shape}')
        return mesh

    def load_dewarp_mesh(self, mesh_path: str) -> DewarpMesh:
        """Load a mesh saved by ``build_dewarp_mesh`` (also the 1.0.x format). Returns it like ``build_dewarp_mesh``."""
        mesh = _load_pickle(mesh_path)
        if len(mesh) == 2:
            # Files saved by version 1.0.x only hold (map x, map y)
            self._dewarp_map_x, self._dewarp_map_y = mesh
            rows, cols = self._dewarp_map_x.shape
            self._panorama_shape = (cols, rows)
        else:
            self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y = mesh
        return self._panorama_shape, self._dewarp_map_x, self._dewarp_map_y

    def run_dewarp(self, save_path: Optional[str] = None) -> np.ndarray:
        """Dewarp this fisheye image into a panorama; save it when ``save_path`` is given."""
        result = self.dewarp(self.img, flip=True)
        print(f'Panorama shape is `{result.shape}`')
        _save_image(save_path, result)
        return result

    def dewarp(self, img: np.ndarray, flip: bool = False) -> np.ndarray:
        """Apply the dewarp mesh to ``img``. ``flip`` turns the result upside down, as ``run_dewarp`` does."""
        self._require_dewarp_mesh()
        output = cv2.remap(img, self._dewarp_map_x, self._dewarp_map_y, cv2.INTER_LINEAR)
        if flip:
            output = _rotate_180(output)
        return output

    # ---- rewarp: panorama -> fisheye ---------------------------------------------------

    def build_rewarp_mesh(self, save_path: Optional[str] = None) -> RewarpMesh:
        """Build the mesh that turns a panorama back into a fisheye image of this size; save it when ``save_path`` is given.

        Needs the dewarp mesh first (built or loaded). Returns ``(map x, map y, mask)``.
        """
        self._require_dewarp_mesh()
        panorama_width = self._dewarp_map_x.shape[1]
        mesh = _build_rewarp_maps(self.img, panorama_width, self.use_multiprocessing)
        self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask = mesh
        if _wants_saving(save_path):
            _save_pickle(save_path, mesh)
        print(f'Rewarp Map X shape -> {self._rewarp_map_x.shape}')
        print(f'Rewarp Map Y shape -> {self._rewarp_map_y.shape}')
        print(f'Rewarp Map MASK shape -> {self._rewarp_mask.shape}')
        return mesh

    def load_rewarp_mesh(self, mesh_path: str) -> RewarpMesh:
        """Load a mesh saved by ``build_rewarp_mesh``. Returns it like ``build_rewarp_mesh``."""
        self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask = _load_pickle(mesh_path)
        return self._rewarp_map_x, self._rewarp_map_y, self._rewarp_mask

    def run_rewarp(self, save_path: Optional[str] = None) -> np.ndarray:
        """Dewarp this fisheye image and rewarp the result: a round trip back to a fisheye image."""
        dewarp_result = self.dewarp(self.img, flip=False)
        result = self.rewarp(dewarp_result, flip=True)
        print(f'Fisheye image shape is `{result.shape}`')
        _save_image(save_path, result)
        return result

    def run_rewarp_with_mesh(self, panorama_img: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """Rewarp any panorama into a fisheye image; save it when ``save_path`` is given.

        Needs both meshes: the dewarp mesh tells the panorama size, the rewarp mesh does the work.
        The panorama is resized to the size the meshes were built for.
        """
        assert self._panorama_shape is not None, \
            "Rewarp needs the shape of panorama generated from `run_dewarp`. Please run it first."
        panorama_img = cv2.resize(panorama_img, self._panorama_shape)
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
        self._require_rewarp_mesh()
        canvas, right_output = self.half_rewarp_map(panorama_img, self._rewarp_map_x, self._rewarp_map_y)
        # The angle only covers half a turn, so the right half comes from the mirrored panorama.
        # 1.0.x splits at the middle row count, which is the middle column for square images.
        split = int(canvas.shape[0] / 2) + 1
        canvas[:, split:, :] = right_output[:, split:, :]
        # Black outside the circle
        canvas = cv2.add(canvas, np.zeros(canvas.shape, dtype=np.uint8), mask=self._rewarp_mask)
        if flip:
            canvas = _rotate_180(canvas)
        return canvas

    # ---- checks ---------------------------------------------------------------------------

    def _require_dewarp_mesh(self) -> None:
        assert self._dewarp_map_x is not None and self._dewarp_map_y is not None, \
            "Dewarp mesh have not been created! Please run `build_dewarp_mesh` first."

    def _require_rewarp_mesh(self) -> None:
        assert self._rewarp_map_x is not None and self._rewarp_map_y is not None and self._rewarp_mask is not None, \
            "Rewarp mesh have not been created! Please run `build_rewarp_mesh` first."
