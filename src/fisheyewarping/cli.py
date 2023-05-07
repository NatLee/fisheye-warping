import time
import argparse
from pathlib import Path

import cv2

from fisheyewarping import FisheyeWarping

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--panorama_output', type=str, default='./dewarp-output.png', help='Specific path for `output`. Default is `./dewarp-output.png`.')
    parser.add_argument('--fisheye_output', type=str, default='./rewarp-output.png', help='Specific path for `output`. Default is `./rewarp-output.png`.')

    parser.add_argument('--save_dewarp_mesh_path', type=str, default=None, help='Specific path for saving mesh data for `dewarping`. Default is `None`.')
    parser.add_argument('--save_rewarp_mesh_path', type=str, default=None, help='Specific path for saving mesh data for `rewarping`. Default is `None`.')

    parser.add_argument('--load_dewarp_mesh_path', type=str, default=None, help='Specific path for loading mesh data for `dewarping`. Default is `None`.')
    parser.add_argument('--load_rewarp_mesh_path', type=str, default=None, help='Specific path for loading mesh data for `rewarping`. Default is `None`.')

    parser.add_argument('--fisheye_img_path', type=str, default=None, help='Specific path of your fisheye image for dewarping to a panorama.')
    parser.add_argument('--panorama_img_path', type=str, default=None, help='Specific path of your panorama image for rewarping to a fisheye image.')

    parser.add_argument('--use_multiprocessing', type=bool, default=True, help='Use multiprocessing to get mesh. Default is `True`.')

    # ---------------------------------------------------------------

    args = parser.parse_args()

    print('========Start to process========')

    panorama_output_path = args.panorama_output
    print(f'----- Panorama output image path will be `{panorama_output_path}`.')
    fisheye_output_path = args.fisheye_output
    print(f'----- Fisheye output image path will be `{fisheye_output_path}`.')

    # check path
    if args.fisheye_img_path:
        fisheye_img_path = Path(args.fisheye_img_path)
        print(f'----- Detect `fisheye_img_path` is `{fisheye_img_path}`.')
        if not fisheye_img_path.is_file():
            print('----- Your `fisheye_img_path` is not a file!')
            return
        print(f'----- Your image path of the input path is `{fisheye_img_path}`')
        # check extension
        suffix = fisheye_img_path.suffix
        if suffix not in ['.jpg', '.jpeg', '.png']:
            print('----- Only support common types of image `.jpg` and `.png`.')
            print(f'----- Your suffix of the input is `{suffix}`.')
            return

    if args.panorama_img_path:
        panorama_img_path = Path(args.panorama_img_path)
        print(f'----- Detect `panorama_img_path` is {panorama_img_path}.')
        if not panorama_img_path.is_file():
            print('----- Your `panorama_img_path` is not a file!')
            return
        print(f'----- Your image path of the input path is `{panorama_img_path}`')
        # check extension
        suffix = panorama_img_path.suffix
        if suffix not in ['.jpg', '.jpeg', '.png']:
            print('----- Only support common types of image `.jpg` and `.png`.')
            print(f'----- Your suffix of the input is `{suffix}`.')
            return

    if not args.fisheye_img_path and not args.panorama_img_path:
        print('----- Please specific a path for `fisheye_img_path` or `panorama_img_path`!')
        return

    print('===================')

    # check multiprocessing
    if not isinstance(args.use_multiprocessing, bool):
        print('----- `use_multiprocessing` is only use `True` or `False`')
        return
    else:
        use_multiprocessing = args.use_multiprocessing
        print(f'----- Multiprocessing generate mesh flag is `{use_multiprocessing}`')

    print('===================')

    # run fisheye -> panorama -> fisheye
    if args.fisheye_img_path and args.panorama_img_path:
        print('----- Use `Dewarp` method to dewarp image from a fisheye image to a panorama image.')
        print('----- And use `Rewarp` method to rewarp the follow output from a panorama image to fisheye image.')

        load_dewarp_mesh_path = args.load_dewarp_mesh_path
        save_dewarp_mesh_path = args.save_dewarp_mesh_path
        load_rewarp_mesh_path = args.load_rewarp_mesh_path
        save_rewarp_mesh_path = args.save_rewarp_mesh_path

        # =====================================
        st = time.time()
        st_dewarp = st
        fisheye_img = cv2.imread(fisheye_img_path.as_posix())
        panorama_img = cv2.imread(panorama_img_path.as_posix())
        frd = FisheyeWarping(fisheye_img, use_multiprocessing=use_multiprocessing)
        # =====================================

        if load_dewarp_mesh_path:
            print(f'----- Detect `load_dewarp_mesh_path` is {load_dewarp_mesh_path}')
            print(f'----- Load mesh from `{load_dewarp_mesh_path}`!')

            # use already built mesh 

            # ===================================
            if not Path(load_dewarp_mesh_path).exists():
                print('----- Your input path of the mesh does not exists!')
                return
            if Path(load_dewarp_mesh_path).is_file():
                frd.load_dewarp_mesh(mesh_path=load_dewarp_mesh_path)
            else:
                print('----- Your input path of the mesh is not a file!')
                return
            # ===================================
            frd.run_dewarp(save_path=panorama_output_path)

        elif save_dewarp_mesh_path:

            # build mesh

            if Path(save_dewarp_mesh_path).is_dir():
                print('----- `save_dewarp_mesh_path` is a directory!')
                return
            print(f'----- Detect `save_dewarp_mesh_path` is {save_dewarp_mesh_path}')
            print(f'----- We will save the mesh to `{save_dewarp_mesh_path}` when this process has been finished!')
            frd.build_dewarp_mesh(save_path=save_dewarp_mesh_path)
            frd.run_dewarp(save_path=panorama_output_path)

        else:
            print('----- You must specify a path to `load_dewarp_mesh_path` or `save_dewarp_mesh_path`!')
            return

        et = time.time()
        print(f'-------Dewarping Task Completed------- ({et-st_dewarp:.3f} s)')

        st_rewarp = time.time()

        if load_rewarp_mesh_path:
            print(f'----- Detect `load_rewarp_mesh_path` is {load_rewarp_mesh_path}')
            print(f'----- Load mesh from `{load_rewarp_mesh_path}`!')

            # use already built mesh 

            # ===================================
            if not Path(load_rewarp_mesh_path).exists():
                print('----- Your input path of the mesh does not exists!')
                return
            if Path(load_rewarp_mesh_path).is_file():
                frd.load_rewarp_mesh(mesh_path=load_rewarp_mesh_path)
            else:
                print('----- Your input path of the mesh is not a file!')
                return
            # ===================================
            frd.run_rewarp_with_mesh(panorama_img, save_path=fisheye_output_path)

        elif save_rewarp_mesh_path:

            # build mesh

            if Path(save_rewarp_mesh_path).is_dir():
                print('----- `save_rewarp_mesh_path` is a directory!')
                return
            print(f'----- Detect `save_rewarp_mesh_path` is {save_rewarp_mesh_path}')
            print(f'----- We will save the mesh to `{save_rewarp_mesh_path}` when this process has been finished!')
            frd.build_rewarp_mesh(save_path=save_rewarp_mesh_path)
            frd.run_rewarp_with_mesh(panorama_img, save_path=fisheye_output_path)

        et = time.time()
        print(f'-------Rewarping Task Completed------- ({et-st_rewarp:.3f} s)')

        print(f'-------All Task Completed------- ({et-st:.3f} s)')

    # run fisheye -> panorama method
    elif args.fisheye_img_path:
        print('----- Use `Dewarp` method to dewarp image from a fisheye image to a panorama image.')

        # =====================================
        st = time.time()
        fisheye_img = cv2.imread(fisheye_img_path.as_posix())
        frd = FisheyeWarping(fisheye_img, use_multiprocessing=use_multiprocessing)
        # =====================================

        load_dewarp_mesh_path = args.load_dewarp_mesh_path
        save_dewarp_mesh_path = args.save_dewarp_mesh_path
        if load_dewarp_mesh_path:
            print(f'----- Detect `load_dewarp_mesh_path` is {load_dewarp_mesh_path}')
            print(f'----- Load mesh from `{load_dewarp_mesh_path}`!')

            # use already built mesh 

            # ===================================
            if not Path(load_dewarp_mesh_path).exists():
                print('----- Your input path of the mesh does not exists!')
                return
            if Path(load_dewarp_mesh_path).is_file():
                frd.load_dewarp_mesh(mesh_path=load_dewarp_mesh_path)
            else:
                print('----- Your input path of the mesh is not a file!')
                return
            # ===================================
            frd.run_dewarp(save_path=panorama_output_path)

        elif save_dewarp_mesh_path:

            # build mesh

            if Path(save_dewarp_mesh_path).is_dir():
                print('----- `save_dewarp_mesh_path` is a directory!')
                return
            print(f'----- Detect `save_dewarp_mesh_path` is {save_dewarp_mesh_path}')
            print(f'----- We will save the mesh to `{save_dewarp_mesh_path}` when this process has been finished!')
            frd.build_dewarp_mesh(save_path=save_dewarp_mesh_path)
            frd.run_dewarp(save_path=panorama_output_path)

        else:
            print('----- You must specify a path to `load_dewarp_mesh_path` or `save_dewarp_mesh_path`!')
            return

        et = time.time()
        print(f'-------Task Completed------- ({et-st:.3f} s)')

    # run panorama -> fisheye method
    elif args.panorama_img_path:
        print('----- Use `Rewarp` method to rewarp an image from a panorama image to fisheye image.')

        load_rewarp_mesh_path = args.load_rewarp_mesh_path
        save_rewarp_mesh_path = args.save_rewarp_mesh_path

        # =====================================
        st = time.time()
        frd = FisheyeWarping(None, use_multiprocessing=use_multiprocessing)
        panorama_img = cv2.imread(panorama_img_path.as_posix())
        # =====================================

        if load_rewarp_mesh_path:
            print(f'----- Detect `load_rewarp_mesh_path` is {load_rewarp_mesh_path}')
            print(f'----- Load mesh from `{load_rewarp_mesh_path}`!')

            # use already built mesh 

            # ===================================
            if not Path(load_rewarp_mesh_path).exists():
                print('----- Your input path of the mesh does not exists!')
                return
            if Path(load_rewarp_mesh_path).is_file():
                frd.load_rewarp_mesh(mesh_path=load_rewarp_mesh_path)
            else:
                print('----- Your input path of the mesh is not a file!')
                return
            # ===================================

            frd.run_rewarp_with_mesh(panorama_img, save_path=fisheye_output_path)

        else:
            print('----- You must specify a path to `load_rewarp_mesh_path`!')
            return
        
        et = time.time()
        print(f'-------All Tasks Completed------- ({et-st:.3f} s)')

    print('========End of this process========')

    return

