import numpy as np
import os
from matplotlib.image import imread
from matplotlib import pyplot as plt
from PIL import Image

# Parameters
path = "../Datasets/CACD/Images/"
path_save = "../Datasets/CACD/"
dimensions = (50, 50, 3)
num_of_images = 163446
image_file_extention = '0001.jpg'

remove_file_extension = -1 * len(image_file_extention)

X = np.zeros((num_of_images, np.prod(dimensions)))
Y = np.zeros((num_of_images,))

cnt = 0
class_label = -1
previous_name = 'NONE'
for filename in os.listdir(path):
    if previous_name not in filename:
        class_label = class_label + 1
        previous_name = filename[:remove_file_extension]
    npimg_pil = Image.open(os.path.join(path, filename))
    if cnt == 0:
        plt.imshow(npimg_pil)
        plt.show()
    npimg_pil = npimg_pil.resize((dimensions[0], dimensions[1]), Image.ANTIALIAS)
    if cnt == 0:
        plt.imshow(npimg_pil)
        plt.show()
    npimg = np.asarray(npimg_pil)
    X[cnt, :] = npimg.flatten()
    Y[cnt] = str(int(class_label))
    cnt = cnt + 1

np.save(os.path.join(path_save, 'X.npy'), X)
np.save(os.path.join(path_save, 'Y.npy'), Y)
