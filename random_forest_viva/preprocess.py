import av
import numpy as np
import cv2
import matplotlib.pyplot as plt


class PreProcess:
    def __init__(self):
        pass

    def _read_one_avi(self, path):
        res = []
        cap = av.open(path)
        for frame in cap.decode(video=0):
            # print(frame.planes[0].buffer_size)
            # for i in range(3):
            #     print(frame.planes[i].height, frame.planes[i].width, frame.planes[i].buffer_size)
            #     # res.append(np.frombuffer(frame.planes[i], np.uint8).reshape(frame.height, 256))
            # x = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, 256)
            # res = np.array(res)
            # plt.imshow(np.rollaxis(res, 0, 2))
            x = np.asarray(frame.to_image())
            print(np.all(x[:, :, 0] == x[:, :, 1]))
            plt.imshow(x)
            plt.show()
            # print(np.all(x == 128))
            break


if __name__ == '__main__':
    pp = PreProcess()
    pp._read_one_avi('./data/03_10_02.avi')
