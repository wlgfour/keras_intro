import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# globals
print('defining globals')
VISUALIZE = True


# load
print('loading data')
f = np.load('data/face_images.npz')
data = f['face_images']
f.close()
new_data = list()
for i in range(np.shape(data)[2]):
    new_data.append(data[:, :, i])
data = np.array(new_data)  # change from (height, width, images) to (images, height, width)
df = pd.read_csv('data/facial_keypoints.csv')
df = df.reset_index()  # makes index df[image][0]
# df.fill_na(0)  # fill nan with 0
# TODO: fill rather than drop nan. do not increase loss for points with true values of 0 -- custom loss
df = df.dropna()  # easy way out. lots of data lost
pnts = df.values  # df to np.ndarray shape=(imgs, 30)
data = data[pnts[:, 0].astype('int32')]  # get all images that haven't been dropped
if VISUALIZE:
    for i in range(10):
        plt.imshow(data[i])
        x = pnts[i][1::2]
        y = pnts[i][2::2]
        plt.scatter(x, y)
        plt.show()

