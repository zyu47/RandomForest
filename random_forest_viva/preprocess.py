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
            x = np.asarray(frame.to_image())[:, :, 0]  # one frame
            res.append(x)

        return res


if __name__ == '__main__':
    pp = PreProcess()
    res = pp._read_one_avi('./data/03_10_02.avi')
    plt.figure(figsize=(10,10))
    for i in range(5):
        for j in range(5):
            plt.subplot(i+1, j+1)
            plt.imshow(res[i*5 + j])

    plt.show()
