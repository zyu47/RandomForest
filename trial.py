from forest import Forest

import numpy as np
import random

test_set_feature = np.load('../result/test_set_feature.npy')
no_classes, no_clips = test_set_feature.shape[0], test_set_feature.shape[1]
test_set_feature = test_set_feature.reshape((no_classes*no_clips, 15, 1024))
samples = np.array([(test_set_feature[i*no_clips + j, :, :], i) for i in range(no_classes) for j in range(no_clips)])

segments = np.load('../result/segments.npy')
features = np.load('../result/features.npy')

available_starting_indices = []
for i, s in enumerate(segments):
    available_starting_indices += [(i, j) for j in range(len(s) - 15 + 1)]

result = open('./trial.py.find_nn_method1_result', 'w')

for step, ref_start_ind in enumerate(random.sample(available_starting_indices, 150)):
    result.write('\n----------%d %d----------\n' % ref_start_ind)
    print('step ', step)
    for repeats in range(5):
        print('\t substep', repeats)
        forest = Forest()
        forest.build_forest(samples)

        ref_feature = features[ref_start_ind[0]][ref_start_ind[1]:ref_start_ind[1]+15]
        forest.add_bulk([(np.array(ref_feature), -1)])

        # test accuracy
        error_cnt = 0
        labels = []
        for i, j in available_starting_indices:
            labels.append(forest.find_nn((np.array(features[i][j:j+15]), -2), method=1)[0])
            if labels[-1] != -1:
                error_cnt += 1
        result.write('\t' + str(error_cnt/len(available_starting_indices)))
        print('\t',  str(error_cnt/len(available_starting_indices)))
result.close()

# i, j = available_starting_indices[20]
# test = (np.array(features[i][j:j+15]), -2)
# forest.find_nn(test)

# trainging_sample_inds = random.sample(range(no_classes*no_clips), int(no_classes*no_clips*0.8))
# testing_sample_inds = np.setdiff1d(range(no_classes*no_clips), trainging_sample_inds)

# forest.build_forest(samples=samples[trainging_sample_inds])
# forest.print_forest(0)

# for i in testing_sample_inds:
#     found = forest.find_nn(samples[i])
#     if found != samples[i][1]:
#         print('Error: ', found, samples[i][1])

# forest.add_bulk(samples[testing_sample_inds])
