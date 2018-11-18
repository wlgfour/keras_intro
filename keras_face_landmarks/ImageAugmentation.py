import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def rot_90(img_tensor, labels_tensor, num_pairs=15):
    """
    rotates a batch of square images with one channel 90' counter clockwise
    :param labels_tensor: tensor for labels batch input
    :param img_tensor: tensor for images batch input
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


def scale(img_tensor, labels_tensor, scale_factor, num_pairs=15):
    """

    :param img_tensor:
    :param labels_tensor:
    :param scale_factor:
    :param num_pairs:
    """
    pass


"""
sess = tf.Session()
filter = tf.constant([[1., 1, 1], [1, 1, 1], [1, 1, 1]])
filter = tf.reshape(filter, (3, 3, 1))
a = tf.nn.dilation2d(img, filter, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
sess.run(a, {img: train_data[0:5]})
"""


def pad_0_rows(img_tensor_batch, n=1):
    """
    adds n rows of 0s to image_tensor
    :param img_tensor_batch: tensor to add rows to. should have shape (batch, rows, cols, channels)
    :param n: number of rows to add
    :return: <tensor shape=(-1, img_tensor_batch.shape[1] * (n + 1), img_tensor_batch.shape[2], 1)>
    precondition: channels=1

    >>> sess = tf.Session()
    >>> img_padded = pad_0_rows(img, 3)
    >>> out = sess.run(img_padded, {img: train_data[0:5]})
    >>> plt.imshow(out[0][:, :, 0])
    >>> plt.show()
    """
    def pad(img_tensor):
        """
        stack
        :param img_tensor: should have shape (rows, cols)
        :return: <tensor shape=(img_tensor.shape[0] * (n + 1), img_tensor.shape[1])>
        """
        # tensor is a row. shape: (rows, cols)
        shape = tf.shape(img_tensor)
        zeros = tf.zeros((shape[1],))  # zero tensor with same size of length col
        return tf.map_fn(lambda row: tf.stack([row] + [zeros] * n), img_tensor)

    img_shape = tf.shape(img_tensor_batch)  # (batch, rows, cols, channels)
    rows = img_shape[1]
    cols = img_shape[2]
    img_rows = tf.reshape(img_tensor_batch, (-1, rows, cols))
    return tf.reshape(tf.map_fn(pad, img_rows), (-1, rows * (n + 1), cols, 1))  # nested tf.map_fn call


def pad_0_cols(img_tensor_batch, n=1):
    """
    pads an image with n columns of 0
    :param img_tensor_batch: tensor to pad. should have shape (batch, rows, cols, channels)
    :param n: number of 0s to add after every pixel
    :return: <tensor shape=(-1, img_tensor_batch.shape[1], img_tensor_batch.shape[2] * (n + 1), 1)>
    precondition: channels=1

    >>> sess = tf.Session()
    >>> img_padded = pad_0_cols(img, 3)
    >>> out = sess.run(img_padded, {img: train_data[0:5]})
    >>> plt.imshow(out[0][:, :, 0])
    >>> plt.show()
    """
    img_shape = tf.shape(img_tensor_batch)
    paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, n]])  # only pad on last dim
    padded = tf.pad(img_tensor_batch, paddings)
    return tf.reshape(padded, (-1, img_shape[1], img_shape[2] * (n + 1), 1))


def pad_0(img_tensor_batch, n=1):
    """
    pads an image with n rows and columns of 0s
    :param img_tensor_batch: batch of images with shape (-1, rows, columns, channels)
    :param n: number of rows and columns to pad
    :return: <tensor shape=(-1, img_tensor_batch.shape[1] * (n + 1), img_tensor_batch.shape[2] * (n + 1), 1)>
    precondition: channels=1

    >>> sess = tf.Session()
    >>> img_padded = pad_0(img, 3)
    >>> out = sess.run(img_padded, {img: train_data[0:5]})
    >>> plt.imshow(out[0][:, :, 0])
    >>> plt.show()
    """
    return pad_0_rows(pad_0_cols(img_tensor_batch, n), n)


def upscale(img_tensor_batch, n=1):
    """
    pads an image with n 0s in each direction to increase size and fills in 0s with convolution
    scales an image up by a factor of n
    :param img_tensor_batch: batch of greyscale images shape=(-1, rows, cols, channels)
    :param n: int to magnify the image by
    :return: <tensor shape=(-1, img_tensor_batch.shape[1] * (n + 1), img_tensor_batch.shape[2] * (n + 1), 1)>

    """
    fltr = tf.ones((n, n, 1, 1))
    padded = pad_0(img_tensor_batch, n - 1)
    return tf.nn.conv2d(padded, fltr, [1, 1, 1, 1], 'SAME')


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
        sess.run(tf.global_variables_initializer())
        # rotate image
        flipped_img_op = rot_90(img, labels)
        flipped_img, rot_points = sess.run(flipped_img_op, {img: train_data[0:5], labels: train_labels[0:5]})
        plt.imshow(flipped_img[0][:, :, 0])
        x = rot_points[0][0::2] * 96
        y = rot_points[0][1::2] * 96
        plt.scatter(x, y, c='r')
        plt.show()
        print(f'rotated img: {np.shape(flipped_img)}, rotated points: {np.shape(rot_points)}')
        # pad image
        img_padded = pad_0(img, 2)
        out = sess.run(img_padded, {img: train_data[0:5]})
        plt.imshow(out[0][:, :, 0])
        plt.show()
        print(f'padded img (2): {np.shape(out)}')
        # upscale padded img
        img_upscaled = upscale(img, 2)
        out = sess.run(img_upscaled, {img: train_data[0:5]})
        plt.imshow(out[0][:, :, 0])
        plt.show()
        print(f'upscale img (2): {np.shape(out)}')
