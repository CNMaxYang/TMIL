import pandas as pd
import os
from shutil import copyfile
import cv2
import numpy as np

f = open('C:/temp/test_list.txt', 'r')
f_str = f.read()
f.close()
file_list = f_str.split('\n')
count = 0
for i in range(1, 13):
    if i < 10:
        path = 'C:/temp/images_00' + str(i) + '/images'
    else:
        path = 'C:/temp/images_0' + str(i) + '/images'
    for j in os.listdir(path):
        file_path = os.path.join(path, j)
        img = cv2.imread(file_path, 0)
        img = cv2.resize(img, (512, 512))
        if j in file_list:
            save_path = '../test_set'
        else:
            save_path = '../data'
        cv2.imwrite(os.path.join(save_path, j), img)
        count += 1
        print(count)


