import unittest
from fisheyewarping import FisheyeWarping

class TestFisheyeWarping(unittest.TestCase):

    def test_create_obj(self):
        frd = FisheyeWarping(None, use_multiprocessing=True) 


if __name__ == '__main__':
    unittest.main()