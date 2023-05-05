import time
import cv2

from fisheye_processor import FisheyeProcessor

if __name__ == '__main__':
    
    st = time.time()
    img_path = './doc/mesh-fisheye.jpg'
    img = cv2.imread(img_path)
    frd = FisheyeProcessor(img, use_multiprocessing=True)
    frd.build_dewarp_mesh()
    frd.build_rewarp_mesh()
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