import time
import cv2

from fisheyewarping import FisheyeWarping

if __name__ == '__main__':
    
    st = time.time()
    img_path = './doc/mesh-fisheye.jpg'
    img = cv2.imread(img_path)
    frd = FisheyeWarping(img, use_multiprocessing=True)
    frd.build_dewarp_mesh(save_path='./dewarp-mesh.pkl')
    frd.build_rewarp_mesh(save_path='./rewarp-mesh.pkl')
    #frd.load_dewarp_mesh(mesh_path='./dewarp-mesh.pkl')
    #frd.load_rewarp_mesh(mesh_path='./rewarp-mesh.pkl')
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