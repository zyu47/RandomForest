import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from src.handRecognition.realtime_hand_recognition import RealTimeHandRecognition as HandRec


# load clip information
clips_all = np.load('./result/usable_videos.npy').item()

# load hand resnet
model = HandRec('RH', 32)


def center_crop_normalize(img, width=128):
    img = img[img.shape[0] // 2 - 64:img.shape[0] // 2 + 64, img.shape[1] // 2 - 64: img.shape[1] // 2 + 64]
    img = img.astype(np.float32)

    return (img - img[width // 2, width // 2]) / 255.0

import random
random.seed(0)
def get_working_clips(classes_cnt=5, clip_per_class=30, window_sz=15, width=128):
    res = []  # result will be 5
    chosen_class_cnt = 0
    paths = []
    class_names = []
    for k, v in clips_all.items():
        if len(v) < clip_per_class:
            continue
        res.append([])
        paths.append([])
        class_names.append(k.split('/')[-1])
        videos = random.sample(v, clip_per_class)
        for v in videos:
            # v contains all the available clips, randomly choose a start point to pick a 15-frame window
            start_point = random.randint(0, len(v)-15)
            clip = []
            paths[-1].append(os.path.join(k, v[start_point]))
            for i in range(15):
                # i is the file name of each frame
#                 print(os.path.join(k, v[start_point+i]))
                img = cv2.imread(os.path.join(k, v[start_point+i]), 0)
                img = center_crop_normalize(img)
#                 print(img.shape)
                clip.append(img)
            res[-1].append(clip)
        chosen_class_cnt += 1
        if chosen_class_cnt == classes_cnt:
            break
    if len(class_names) != classes_cnt:
        raise ValueError('Not enough classes')
    return np.array(res), paths, class_names

test_set_raw, paths, class_names = get_working_clips(classes_cnt=6)


from forest import Forest

f = Forest()


