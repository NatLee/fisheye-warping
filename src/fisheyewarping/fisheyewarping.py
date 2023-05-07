"""
Fisheye Rewarp And Dewarp
"""
import pickle
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    np.seterr(invalid='ignore')
    norm = np.linalg.norm(vector)
    return vector / norm

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_map(points):
    point, center, top_point, radius = points
    point_np = np.asarray(point)
    distance = np.linalg.norm(point_np-center)
    length_percentage = None
    if distance <= radius:
        angle = angle_between(top_point-center, point_np-center)
        length_percentage = angle / (2 * np.pi)
    return point, length_percentage, distance

class FisheyeWarping:

    def __init__(self, img, use_multiprocessing=False):
        self.img = img
        self.use_multiprocessing = use_multiprocessing

        self.__dewarp_map_x, self.__dewarp_map_y = None, None
        self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask = None, None, None

        self.__panorama_shape = None

    def build_dewarp_mesh(self, save_path=None):
        if self.use_multiprocessing:
            self.__dewarp_map_x, self.__dewarp_map_y = self.__build_dewarp_map_with_mp(self.img)
        else:
            self.__dewarp_map_x, self.__dewarp_map_y = self.__build_dewarp_map(self.img)
        if save_path and isinstance(save_path, str):
            with open(save_path, 'wb') as f:
                pickle.dump((self.__dewarp_map_x, self.__dewarp_map_y), f)
        print(f'Dewarp Map X shape -> {self.__dewarp_map_x.shape}')
        print(f'Dewarp Map Y shape -> {self.__dewarp_map_y.shape}')
        h, w = self.__dewarp_map_x.shape
        self.__panorama_shape = (w, h)
        return  self.__panorama_shape, self.__dewarp_map_x, self.__dewarp_map_y

    def load_dewarp_mesh(self, mesh_path:str):
        with open(mesh_path, 'rb') as f:
             self.__panorama_shape, self.__dewarp_map_x, self.__dewarp_map_y = pickle.load(f)
        return self.__panorama_shape, self.__dewarp_map_x, self.__dewarp_map_y

    def run_dewarp(self, save_path=None):
        result = self.dewarp(self.img, flip=True)
        print(f'Panorama shape is `{result.shape}`')
        if save_path and isinstance(save_path, str):
            cv2.imwrite(save_path, result)
        return result

    def build_rewarp_mesh(self, save_path=None):
        if self.use_multiprocessing:
            self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask = self.__build_rewarp_map_with_mp()
        else:
            self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask = self.__build_rewarp_map()
        if save_path and isinstance(save_path, str):
            with open(save_path, 'wb') as f:
                pickle.dump((self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask), f)
        print(f'Rewarp Map X shape -> {self.__rewarp_map_x.shape}')
        print(f'Rewarp Map Y shape -> {self.__rewarp_map_y.shape}')
        print(f'Rewarp Map MASK shape -> {self.__rewarp_mask.shape}')
        return self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask

    def load_rewarp_mesh(self, mesh_path:str):
        with open(mesh_path, 'rb') as f:
            self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask = pickle.load(f)
        return self.__rewarp_map_x, self.__rewarp_map_y, self.__rewarp_mask

    def run_rewarp(self, save_path=None):
        dewarp_result = self.dewarp(self.img, flip=False)
        result = self.rewarp(dewarp_result, flip=True)
        print(f'Fisheye image shape is `{result.shape}`')
        if save_path and isinstance(save_path, str):
            cv2.imwrite(save_path, result)
        return result

    def run_rewarp_with_mesh(self, panorama_img, save_path):
        warning_msg = "Rewarp needs the shape of panorama generated from `run_dewarp`. Please run it first."
        assert self.__panorama_shape != None, warning_msg
        panorama_img = cv2.resize(panorama_img, self.__panorama_shape)
        panorama_img = self.__wrap(panorama_img, rotate_angle=180, scale=1)
        result = self.rewarp(
            panorama_img,
            flip=True
        )
        print(f'Fisheye image shape is `{result.shape}`')
        if save_path and isinstance(save_path, str):
            cv2.imwrite(save_path, result)
        return result

    def __get_fisheye_img_data(self, img):
        # Center
        c_x = int(img.shape[1]/2)
        c_y = int(img.shape[0]/2)
        # Inner circle, now is zero
        r1_x = int(img.shape[1]/2)
        #r1_y = int(img.shape[0]/2)
        r1 = r1_x - c_x
        # Outter circle
        r2_x = int(img.shape[1])
        #r2_y = 0
        r2 = r2_x - c_x
        # Rectangle width and height
        w_d = int(2.0 * (( r2 + r1 ) / 2) * np.pi)
        h_d = (r2 - r1)
        return w_d, h_d, r1, r2, c_x, c_y

    def _dewarp_map_job(self, point):
        y, x, img_details = point
        w_d, h_d, r1, r2, c_x, c_y = img_details
        r = (float(y) / float(h_d)) * (r2 - r1) + r1
        theta = (float(x) / float(w_d)) * 2.0 * np.pi
        x_s = int(c_x + r * np.sin(theta))
        y_s = int(c_y + r * np.cos(theta))
        return (y, x), x_s, y_s

    def __build_dewarp_map(self, img):
        img_details = self.__get_fisheye_img_data(img)
        w_d, h_d, _, _, _, _ = img_details
        mapx = np.zeros((h_d, w_d), np.float32)
        mapy = np.zeros((h_d, w_d), np.float32)
        for y in tqdm(range(0, int(h_d-1)), desc='Run dewarp job...'):
            for x in range(0, int(w_d-1)):
                _, x_s, y_s = self._dewarp_map_job((y, x, img_details))
                mapx.itemset((y, x), x_s)
                mapy.itemset((y, x), y_s)
        return mapx, mapy

    def __build_dewarp_map_with_mp(self, img):
        img_details = self.__get_fisheye_img_data(img)
        w_d, h_d, _, _, _, _ = img_details
        mapx = np.zeros((h_d, w_d), np.float32)
        mapy = np.zeros((h_d, w_d), np.float32)

        jobList = list()
        for y in tqdm(range(0, int(h_d-1)), desc='Getting (x, y) for rectangle...'):
            for x in range(0, int(w_d-1)):
                jobList.append((y, x, img_details))
        st = time.time()
        print('-------Start multi pixel mapping for dewarpping-------')
        with mp.Pool() as p:
            results = p.map(self._dewarp_map_job, jobList)
        print('--------Mapping Completed-------- ({:0.3f} s)'.format(time.time() - st))
        for result in tqdm(results, desc='Mapping values to rectangle...'):
            point, x_s, y_s = result
            mapx.itemset(point, x_s)
            mapy.itemset(point, y_s)
        
        return mapx, mapy

    def dewarp(self, img, flip=False):
        warning_msg = "Dewarp mesh have not been created! Please run `build_dewarp_mesh` first."
        assert self.__dewarp_map_x is not None, warning_msg
        assert self.__dewarp_map_y is not None, warning_msg
        output = cv2.remap(
            img,
            self.__dewarp_map_x,
            self.__dewarp_map_y,
            cv2.INTER_LINEAR
        )
        if flip:
            output = self.__wrap(output, 180, scale=1)
        return output

    def __wrap(self, img, rotate_angle, scale=1/3):
        h, w = img.shape[:2]
        # get the center
        center = (w / 2, h / 2)
        r_matrix = cv2.getRotationMatrix2D(center, rotate_angle, 1)
        img = cv2.warpAffine(img, r_matrix, (w, h))
        img = cv2.resize(
            img,
            dsize=(int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
        return img

    def __build_rewarp_map(self):
        width, height, _  = self.img.shape
        xmap = np.zeros((width, height), dtype=np.float32)
        ymap = np.zeros((width, height), dtype=np.float32)
        mask = np.zeros((width, height), dtype=np.uint8)
        center = np.asarray([int(width/2), int(height/2)])
        top_point = np.asarray([int(width/2), 0])
        radius = width/2
        results = list()
        for y, channel in enumerate(tqdm(self.img, desc='Run rewarp job...')):
            for x, _ in enumerate(channel):
                points = ((x, y), center, top_point, radius)
                results.append(angle_map(points))

        dewarp_result = self.dewarp(self.img, flip=False)
        for result in results:
            (x, y), length_percentage, distance = result
            if length_percentage is not None:
                point = (y, x)
                length = length_percentage * dewarp_result.shape[1]
                xmap.itemset(point, length)
                ymap.itemset(point, distance)
                mask.itemset(point, 255)
        
        return xmap, ymap, mask

    def __build_rewarp_map_with_mp(self):
        width, height, _  = self.img.shape
        xmap = np.zeros((width, height), dtype=np.float32)
        ymap = np.zeros((width, height), dtype=np.float32)
        mask = np.zeros((width, height), dtype=np.uint8)
        center = np.asarray([int(width/2), int(height/2)])
        top_point = np.asarray([int(width/2), 0])
        radius = width / 2
        jobs = list()
        for y, channel in enumerate(tqdm(self.img, desc='Getting (x, y) for circle...')):
            for x, _ in enumerate(channel):
                points = ((x, y), center, top_point, radius)
                jobs.append(points)
        s = time.time()
        print('-------Start multi pixel mapping for rewarpping-------')
        with mp.Pool() as p:
            results = p.map(angle_map, jobs)
        print('--------Mapping Completed-------- ({:0.3f} s)'.format(time.time()-s))
        dewarp_result = self.dewarp(self.img, flip=False)
        for result in tqdm(results, desc='Mapping values to circle...'):
            (x, y), length_percentage, distance = result
            if length_percentage is not None:
                point = (y, x)
                length = length_percentage * dewarp_result.shape[1]
                xmap.itemset(point, length)
                ymap.itemset(point, distance)
                mask.itemset(point, 255)

        return xmap, ymap, mask

    # =================================================================

    def __remap(self, img, x, y):
        return cv2.remap(img, x, y, cv2.INTER_LINEAR)

    def half_rewarp_map(self, panorama_img, x, y):
        # get left part
        left_output = self.__remap(panorama_img, x, y)
        # get right part
        right_output = self.__remap(cv2.flip(panorama_img, 1), x, y)
        return left_output, right_output

    def rewarp(self, panorama_img, flip=False):
        warning_msg = "Rewarp mesh have not been created! Please run `build_rewarp_mesh` first."
        assert self.__rewarp_map_x is not None, warning_msg
        assert self.__rewarp_map_y is not None, warning_msg
        assert self.__rewarp_mask is not None, warning_msg
        left_output, right_output = self.half_rewarp_map(
            panorama_img,
            self.__rewarp_map_x,
            self.__rewarp_map_y
        )

        re_render_canvas = left_output
        # find the center of the image
        vertical_center = int(re_render_canvas.shape[0] / 2) + 1
        # combine 2 parts
        re_render_canvas[:, vertical_center:, :] = right_output[:, vertical_center:, :]

        re_render_canvas = cv2.add(
            re_render_canvas,
            np.zeros(re_render_canvas.shape, dtype=np.uint8),
            mask=self.__rewarp_mask
        )

        if flip:
            re_render_canvas = self.__wrap(re_render_canvas, 180, scale=1)

        return re_render_canvas

    # =================================================================
