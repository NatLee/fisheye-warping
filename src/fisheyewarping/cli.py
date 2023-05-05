import time
import argparse
from pathlib import Path

import cv2

from fisheyewarping import FisheyeWarping

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None, help='Specific path of image as `input`.')
    parser.add_argument('--output', type=str, default='./result.png', help='Specific path for `output`. Default is `./result.png`.')

    parser.add_argument('--save_dewarp_mesh_path', type=str, default=None, help='Specific path for saving mesh data for `dewarping`. Default is `None`.')
    parser.add_argument('--save_rewarp_mesh_path', type=str, default=None, help='Specific path for saving mesh data for `rewarping`. Default is `None`.')
    
    parser.add_argument('--load_dewarp_mesh_path', type=str, default=None, help='Specific path for loading mesh data for `dewarping`. Default is `None`.')
    parser.add_argument('--load_rewarp_mesh_path', type=str, default=None, help='Specific path for loading mesh data for `rewarping`. Default is `None`.')
    
    
    parser.add_argument('--use_multiprocessing', type=bool, default=True, help='Use multiprocessing to get mesh. Default is `True`.')

    # ---------------------------------------------------------------

    args = parser.parse_args()

    if not args.input:
        print('----- Please specific image for input.')
        return

    img_path = Path(args.input)

    if not img_path.is_file():
        print('----- `img_path` is not a file!')
        return

    if img_path.suffix not in ['.jpg', '.jpeg', '.png']:
        print('----- Only support common types of image `.jpg` and `.png`.')
        return

    if not isinstance(args.use_multiprocessing, bool):
        print('----- `use_multiprocessing` is only use `True` or `False`')
        return   

    img_path = Path(args.input)
    use_multiprocessing = args.use_multiprocessing
    save_dewarp_mesh_path = args.save_dewarp_mesh_path
    save_rewarp_mesh_path = args.save_rewarp_mesh_path
    load_dewarp_mesh_path = args.load_dewarp_mesh_path
    load_rewarp_mesh_path = args.load_rewarp_mesh_path

    st = time.time()
    img = cv2.imread(img_path)
    frd = FisheyeWarping(img, use_multiprocessing=use_multiprocessing)

    if isinstance(load_dewarp_mesh_path, str):
        frd.load_dewarp_mesh(mesh_path=load_dewarp_mesh_path)
    else:
        frd.build_dewarp_mesh(save_path=save_dewarp_mesh_path)

    if isinstance(load_rewarp_mesh_path, str):
        frd.load_rewarp_mesh(mesh_path=load_rewarp_mesh_path)
    else:
        frd.build_rewarp_mesh(save_path=save_rewarp_mesh_path)

    # TODO

    panorama = frd.run_dewarp(save_path='./doc/mesh-panorama.jpg')
    fisheye = frd.run_rewarp(save_path='./doc/mesh-fisheye.jpg')
    et = time.time()
    print(f'-------All Tasks Completed------- ({et-st:.3f} s)')

    st = time.time()
    img_path = './doc/101-panorama.jpg'
    img = cv2.imread(img_path)
    frd.run_rewarp_with_mesh(img, save_path='./doc/101-fisheye.jpg')
    et = time.time()
    print(f'-------Wrapping 1 Image------- ({et-st:.3f} s)')


    return

