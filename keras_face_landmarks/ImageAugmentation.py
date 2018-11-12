import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras as k
import tensorflow as tf

"""
Goal:
    1) create a layer that will preform image transformations on the fly on the images and 
    2) operations 
        a) scaling
        b) rotations of 90 degrees (if image is square)
      **c) shift left and right -- avoid noise by scaling and cropping 
Challenges:
    1) labeling the correct points. If the face is flipped horizontally, the left eye will correspond to the right eye
    2) matrix operations
        a) what is the shape of the batch passed to the layer
        b) how to preform the same operation on x-y points as well as entire images
"""

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


def rot_90(img_tensor, labels_tensor, num_pairs=15):
    """
    rotates a batch of square images with one channel 90' counter clockwise
    :param labels_tensor: tensor for labels batch input
    :param img_tensor: tensor for images batch input
    :param width: width of image
    :param num_pairs: number of x-y pairs
    :return: tensors: img: shape=(96, 96, 1), labels: [x1, y1, ..., xn, yn]
    """
    # image_flip
    flip_img = tf.image.rot90(img_tensor)
    # labels_flip
    batch_size = tf.shape(labels_tensor)[0]
    rot_2d_base = tf.constant([[0., -1.], [1., 0.]])
    rot_2d_ = tf.expand_dims(rot_2d_base, axis=0)
    rot_2d = tf.tile(rot_2d_, multiples=[batch_size, 1, 1])
    labels_down = tf.map_fn(lambda z: z - 0.5, labels_tensor)
    labels_ = tf.reshape(labels_down, (-1, num_pairs, 2))
    labels_rot_ = labels_ @ rot_2d
    labels_rot = tf.reshape(labels_rot_, (-1, num_pairs * 2,))
    labels_rot_up = tf.map_fn(lambda z: z + 0.5, labels_rot)

    return flip_img, labels_rot_up


if __name__ == '__main__':
    img = tf.placeholder('float', (None, 96, 96, 1))
    labels = tf.placeholder('float', (None, 15 * 2,))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        flipped_img_op = rot_90(img, labels)
        flipped_img, rot_points = sess.run(flipped_img_op, {img: train_data[0:5], labels: train_labels[0:5]})
        plt.imshow(flipped_img[0][:, :, 0])
        x = rot_points[0][0::2] * 96
        y = rot_points[0][1::2] * 96
        plt.scatter(x, y, c='r')
        plt.show()
