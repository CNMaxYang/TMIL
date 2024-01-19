import numpy as np
import cv2
import os
import random

means, stdevs = [], []
img_list = []
path = '../data'

imgs_path_list = os.listdir(path)
imgs_path_list = random.sample(imgs_path_list, 50000)
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(path, item), 0)
    img = cv2.resize(img, (384, 384))
    img = img[:, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i, '/', len_)

imgs = np.concatenate(img_list, axis=2)
imgs = imgs.astype(np.float32) / 255.

pixels = imgs[:, :].ravel()
means.append(np.mean(pixels))
stdevs.append(np.std(pixels))

print('normMean = {}'.format(means))
print('normStd = {}'.format(stdevs))
