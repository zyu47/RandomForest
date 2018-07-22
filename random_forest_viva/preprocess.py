from collections import namedtuple
import os

import av
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt

off_line_aug_num = 4
HParams = namedtuple("HParams",
                     ["final_depth", "final_img_shape",
                      "angle_range", "scaling_range", "x_translate_range", "y_translate_range",
                      "sigma", "alpha",
                      "ted_sigma", "t_scaling_range", "t_translate_range"])

class GetParams:
    def __init__(self, hps):
        self.hps = hps
        # parameters for first step preprecessing (making video same length, resize half)
        # consistent across all epochs
        self.final_depth = self.hps.final_depth  # 32
        self.final_img_shape = self.hps.final_img_shape  # (57, 125)
        # affine transformation parameters
        self.rotate_ang = np.random.rand() * (self.hps.angle_range[1] - self.hps.angle_range[0]) + \
                          self.hps.angle_range[0]
        self.scaling = 1 + np.random.rand() * (self.hps.scaling_range[1] - self.hps.scaling_range[0]) + \
                       self.hps.scaling_range[0]
        self.translate_x = np.random.rand() * (self.hps.x_translate_range[1] - self.hps.x_translate_range[0]) + \
                           self.hps.x_translate_range[0]
        self.translate_y = np.random.rand() * (self.hps.y_translate_range[1] - self.hps.y_translate_range[0]) + \
                           self.hps.y_translate_range[0]
        # spatial elastic deformation parameters
        self.sed_indices = self._get_sed_param(self.hps.sigma, self.hps.alpha)
        # fixed pattern dropout parameters
        self.dropout_mask_depth = self._get_mask('depth')
        self.dropout_mask_intensity = self._get_mask('intensity')
        # temporal augmentation parameters
        self.t_scaling = 1 + np.random.rand() * (self.hps.t_scaling_range[1] - self.hps.t_scaling_range[0]) + \
                         self.hps.t_scaling_range[0]
        self.t_translate = np.random.rand() * (self.hps.t_translate_range[1] - self.hps.t_translate_range[0]) + \
                           self.hps.t_translate_range[0]
        self.t_aug_indices = self._get_temporal_mapping(self.hps.ted_sigma)

    def _get_sed_param(self, sigma, alpha):
        dx = np.random.rand(*self.final_img_shape) * 2 - 1.0
        dy = np.random.rand(*self.final_img_shape) * 2 - 1.0
        dx = gaussian_filter(dx, sigma, mode='constant', cval=0.0) * alpha
        dy = gaussian_filter(dy, sigma, mode='constant', cval=0.0) * alpha

        x, y = np.meshgrid(range(self.final_img_shape[1]), range(self.final_img_shape[0]))
        return np.reshape(y + dy, (1, -1)), np.reshape(x + dx, (1, -1))

    def _get_mask(self, type):
        mask = np.random.choice([True, False], self.final_img_shape)
        mask = np.repeat([mask], self.final_depth, axis=0)[:, :, :, np.newaxis]
        if type == 'depth':
            mask = np.concatenate((np.zeros((self.final_depth, *self.final_img_shape, 1), dtype=np.bool), mask),
                                  axis=3)
        else:
            mask = np.concatenate((mask, np.zeros((self.final_depth, *self.final_img_shape, 1), dtype=np.bool)),
                                  axis=3)
        return mask

    def _get_temporal_mapping(self, sigma):
        indices = self._get_ted_param(sigma)  # ATTENTION: 1-based index
        indices = indices * self.t_scaling - self.final_depth * (self.t_scaling - 1) / 2  # scaling
        indices = shift(indices, self.t_translate, mode='nearest')  # translation

        return np.clip(np.round(indices), 1, self.final_depth) - 1

    def _get_ted_param(self, sigma):
        M = self.final_depth / 2
        n = np.random.normal(M, sigma)
        m = np.random.normal(n, 4 * (1 - np.abs(n - M) / M))
        x = [1, n, self.final_depth]
        y = [1, m, self.final_depth]
        poly_params = np.polyfit(x, y, 2)

        # indices are  1-based float
        return np.clip(np.polyval(poly_params, range(1, self.final_depth+1)), 1, self.final_depth)


class SpatialAug:
    def __init__(self,
                 rotate_ang, scaling, translate_x, translate_y,
                 sed_indices,
                 dropout_mask_depth, dropout_mask_intensity):
        self.rotate_ang = rotate_ang
        self.scaling = scaling
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.sed_indices = sed_indices
        self.dropout_mask_depth = dropout_mask_depth
        self.dropout_mask_intensity = dropout_mask_intensity

    def affine_video(self, video_in):
        for depth in range(video_in.shape[0]):
            for channel in range(video_in.shape[3]):
                video_in[depth, :, :, channel] = \
                    self._affine_img(video_in[depth, :, :, channel])

    def sed_video(self, video_in):
        for depth in range(video_in.shape[0]):
            for channel in range(video_in.shape[3]):
                video_in[depth, :, :, channel] = \
                    self._spatial_elastic_deform(video_in[depth, :, :, channel])

    def fixed_pattern_dropout(self, video_in):
        """
        Change video in place
        :param video_in:
        :return: None
        """
        video_in[self.dropout_mask_depth] = 0
        video_in[self.dropout_mask_intensity] = 0

    def random_dropout(self, video_in):
        """
        Change video in place. We only need one mask because it's randomly generated,
        therefore two channels are individually dropped out.
        :param video_in:
        :return: None
        """
        mask = np.random.choice([True, False], video_in.shape)
        video_in[mask] = 0

    def _affine_img(self, img_in):
        M = cv2.getRotationMatrix2D((img_in.shape[1] // 2, img_in.shape[0] // 2),
                                    self.rotate_ang, self.scaling)
        M += np.array([[0, 0, self.translate_x],
                       [0, 0, self.translate_y]])

        return cv2.warpAffine(img_in, M, (img_in.shape[1], img_in.shape[0]))

    def _spatial_elastic_deform(self, img_in):
        return map_coordinates(img_in, self.sed_indices, order=1).reshape(img_in.shape)


class TemporalAug:
    def __init__(self, temporal_aug_ind):
        self.temporal_aug_ind = temporal_aug_ind.astype(np.int16)

    def temporal_augmentation(self, video_in):
        new_video = np.copy(video_in)
        for i in range(new_video.shape[0]):
            new_video[i] = video_in[self.temporal_aug_ind[i]]
        return new_video


class Augmentation(GetParams):
    def __init__(self, hps):
        GetParams.__init__(self, hps)
        self.spatial_aug = SpatialAug(self.rotate_ang, self.scaling, self.translate_x, self.translate_y,
                                      self.sed_indices,
                                      self.dropout_mask_depth, self.dropout_mask_intensity)
        self.temporal_aug = TemporalAug(self.t_aug_indices)

    def offline_aug(self, video_input, type=0):
        """
        Offline augmentation of video by reversing, mirroing and/or both. It returns a copy
        with the original copy intact.
        :param video_in: input video with first axis as time, second axis as height and third as width
        :param type: 0 - original, 1 - reversing, 2 - mirroring, 3 - reversing and mirroring
        :return: augmented video
        """
        video_in = np.copy(video_input)
        if type == 0:
            return video_in
        elif type == 1:
            return self._reverse_video(video_in)
        elif type == 2:
            return self._mirror_video(video_in)
        elif type == 3:
            return self._reverse_video(self._mirror_video(video_in))
        else:
            raise ValueError("Unrecognized type value, only accepting 0, 1, 2, 3")

    def online_aug(self, video_in):
        """
        In place change
        :param video_in:
        :param type:
        :return:
        """
        if self._perform_aug():
            self.spatial_aug.affine_video(video_in)
        if self._perform_aug():
            self.spatial_aug.sed_video(video_in)
        if self._perform_aug():
            self.spatial_aug.fixed_pattern_dropout(video_in)
        if self._perform_aug():
            self.spatial_aug.random_dropout(video_in)

    def _reverse_video(self, video_in):
        return np.flip(video_in, 0)

    def _mirror_video(self, video_in):
        return np.flip(video_in, 2)

    def _perform_aug(self):
        return np.random.choice([0, 1]) == 1


class PrimaryProcess:
    def __init__(self, final_depth, final_img_shape,
                 intensity_vid_root='./data/videos', depth_vid_root='./data/videos/depth'):
        self.final_depth = final_depth
        self.final_img_shape = final_img_shape
        self.intensity_vid_root = intensity_vid_root
        self.depth_vid_root = depth_vid_root

    def read_both_channels(self, file_name):
        """
        Read both channels, apply sobel on intensity and reshape frame to final size
        0-mean, unit length
        :param file_name:
        :return:
        """
        path_depth = os.path.join(self.depth_vid_root, file_name)
        depth_vid = self._read_and_process_one_avi(path_depth, 'depth')[:, :, :, np.newaxis]
        depth_vid = (depth_vid - np.mean(depth_vid)) / np.std(depth_vid)

        path_intensity = os.path.join(self.intensity_vid_root, file_name)
        intensity_vid = self._read_and_process_one_avi(path_intensity, 'intensity')[:, :, :, np.newaxis]
        intensity_vid = (intensity_vid - np.mean(intensity_vid)) / np.std(intensity_vid)
        return self._nni_video(np.concatenate((intensity_vid, depth_vid), axis=3))

    def _read_and_process_one_avi(self, path, type):
        """
        Resize (and apply sobel on) frame when reading video
        :param path:
        :param type:
        :return:
        """
        res = []
        cap = av.open(path)
        for frame in cap.decode(video=0):
            x = np.asarray(frame.to_image())[:, :, 0].astype(np.float32)  # one frame
            x = cv2.resize(x, (self.final_img_shape[1], self.final_img_shape[0]))
            if type == 'intensity':
                x = cv2.Sobel(x, cv2.CV_32F, 1, 1)
            res.append(x)

        return np.array(res)

    def _nni_video(self, video_in):
        """
        Reshape each video to a frame number of <final_depth>
        :param video_in: input video
        :return: reshape video sequence
        """
        curr_depth = video_in.shape[0]
        if curr_depth < self.final_depth:
            video_in = np.repeat(video_in, (self.final_depth - 1) // curr_depth + 1, axis=0)

        curr_depth = video_in.shape[0]
        if curr_depth == self.final_depth:
            return video_in

        while curr_depth != self.final_depth:
            delete_every = (curr_depth - 1) // (curr_depth - self.final_depth) + 1
            mask = np.ones(curr_depth, dtype=bool)
            mask[range(curr_depth // delete_every // 2, curr_depth, delete_every)] = False
            video_in = video_in[mask]
            curr_depth = video_in.shape[0]

        assert(curr_depth == self.final_depth)

        return video_in


class ProcessLabel:
    def __init__(self, depth_path='./data/videos/depth'):
        self.gestures = [1,2,3,4,6,7,8,13,14,15,16,21,23,27,28,29,30,31,32]
        self.label_gesture_mapping = {self.gestures[i]: i for i in range(len(self.gestures))}
        self.label_gesture_mapping[70] = 7
        self.label_gesture_mapping[80] = 8

        subject_id_map = [(1, 1), (2, 1), (3, 2), (6, 2), (4, 3), (5, 3),
                          (7, 4), (8, 4), (9, 5), (12, 5), (10, 6), (11, 6),
                          (13, 7), (16, 7), (14, 8), (15, 8)]
        self.raw_id_subject_id_mapping = {i: j for i, j in subject_id_map}

        self.all_samples_by_subject_id = {i: [] for i in range(1, 9)}
        self.all_labels_by_subject_id = {i: [] for i in range(1, 9)}

        self.get_names_and_labels(depth_path)

    def get_names_and_labels(self, path):
        file_names = os.listdir(path)
        for n in file_names:
            parsed_result = self._parse_label(n)
            if parsed_result is None:
                continue
            self.all_samples_by_subject_id[parsed_result[0]].append(n)
            self.all_labels_by_subject_id[parsed_result[0]].append(parsed_result[1])
        for k in self.all_labels_by_subject_id:
            # Because each sample needs off-line augmentation
            # Repeat each sample four times
            self.all_samples_by_subject_id[k] = np.repeat(self.all_samples_by_subject_id[k], off_line_aug_num)
            self.all_labels_by_subject_id[k] = np.repeat(self.all_labels_by_subject_id[k], off_line_aug_num)
            self.all_labels_by_subject_id[k] = self._one_hot(self.all_labels_by_subject_id[k])

    def _one_hot(self, input_1d_array, label_num=19):
        result = np.zeros((len(input_1d_array), label_num))
        result[range(len(input_1d_array)), input_1d_array] = 1

        return result

    def _parse_label(self, file_name):
        split_names = file_name[:-4].split('_')
        subject_id = int(split_names[0])
        gesture_id = int(split_names[1])
        if gesture_id not in self.label_gesture_mapping:
            return None
        else:
            return self.raw_id_subject_id_mapping[subject_id],\
                   self.label_gesture_mapping[gesture_id]


if __name__ == '__main__':
    hps = HParams(final_depth=32, final_img_shape=(57, 125),
                  angle_range=(-10, 10), scaling_range=(-0.3, 0.3),
                  x_translate_range=(-8, 8), y_translate_range=(-4, 4),
                  sigma=10, alpha=6,
                  ted_sigma=4, t_scaling_range=(-0.2, 0.2), t_translate_range=(-4, 4) )
    pp = PrimaryProcess(hps.final_depth, hps.final_img_shape)
    # agt = Augmentation(hps)
    video_test = pp.read_both_channels('01_13_01.avi')
    aug = Augmentation(hps)
    video_test = aug.offline_aug(video_test, 3)
    aug.online_aug(video_test)
    # # video_test = agt.offline_aug(video_test, 3)
    # video_test = agt.temporal_aug.temporal_augmentation(video_test)
    # print(video_test.shape)
    # # # plt.imshow(pp._spatial_elastic_deform(res[0]))
    # # plt.imshow(res[0])
    # # # plt.imshow(pp._sobel(res[0]))
    plt.figure(figsize=(16,16))
    for i in range(32):
        for j in range(2):
            plt.subplot(8, 8, 2*i + j + 1)
            plt.imshow(video_test[i, :, :, j], cmap='Greys')
    plt.show()
    # #
    # plt.savefig('result/temporal_aug.png')
    # # # plt.show()

    # pl = ProcessLabel()
    # pl.get_names_and_labels()
    # print(pl.all_labels_by_subject_id[1])
    # print(pl.all_samples_by_subject_id[1])
