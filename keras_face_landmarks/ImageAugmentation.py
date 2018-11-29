import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

"""
Goal:
    1) preprocess a random selection of images to be augmented
    2) operations 
        a) scaling
        b) rotations of 90 degrees (if image is square)
      **c) shift left and right -- avoid noise by scaling and cropping 
Method: rather than make the tensorflow layers directly, use tensorflow augmentation
    1) convert keypoints to values on the image (one for each keypoint) with value of points.index(point)
    2) build augmentation model. add necessary augmentations
    3) feed images through all augmentations and points through augmentations that transform
    4) get indices of points and convert to coordinates
"""


if __name__ == '__main__':
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
    train_data = data * (1 / data.max())  # scale to [0, 1]
    train_data = train_data.reshape((-1, 96, 96, 1))
    train_labels = pnts[:, 1:]
    train_labels = train_labels * (1 / np.shape(data)[1])  # scale so scaled * image_width = original position

    img = tf.placeholder('float', (None, 96, 96, 1))
    labels = tf.placeholder('float', (None, 15 * 2,))
    with tf.Session() as sess:
        pass
