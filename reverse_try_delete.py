import numpy as np
import os
from matplotlib import pylab as plt
from matplotlib.image import imread

path = "../Datasets/AR/"
path_2 = "../Datasets/AR/Images/"
dimensions = (165, 120, 3)
num_of_images = 2600
X = np.load(os.path.join(path, 'X.npy'))
X = X.astype(np.int)
print(X.shape)

current_row = X[:, 0]
np.savetxt(os.path.join(path, 'reverse_image_flatten.txt'), current_row, fmt='%d')

current_row = np.reshape(current_row, dimensions)

plt.imshow(current_row)
plt.axis('off')
plt.show()

