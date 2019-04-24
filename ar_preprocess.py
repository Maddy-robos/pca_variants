import numpy as np
import os
from matplotlib.image import imread

path = "../Datasets/AR/Images/"
path_save = "../Datasets/AR/"
dimensions = (165, 120, 3)
num_of_images = 2600
X = np.zeros((num_of_images, np.prod(dimensions)))
Y = np.zeros((num_of_images,))

cnt = 0
class_label = -1
previous_name = 'NONE'
for filename in os.listdir(path):
    if previous_name not in filename:
        class_label = class_label + 1
        previous_name = filename[:-6]
    npimg = imread(os.path.join(path, filename))
    X[cnt, :] = npimg.flatten()
    Y[cnt] = str(int(class_label))
    cnt = cnt + 1

np.save(os.path.join(path_save, 'X.npy'), X)
np.save(os.path.join(path_save, 'Y.npy'), Y)
