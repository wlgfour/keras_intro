import numpy as np


# load
print('loading data')
f = np.load('data/face_images.npz')
data = f['face_images']
f.close()
