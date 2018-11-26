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


# tf.py_func helpers. Wrappers for numpy functions
def tf_repeat(arr, repeats):
    """
    wraps a numpy functions in a tf graph operation
    :param arr: tensor to repeat. should have shape (batch, x) to repeat x
    :param repeats: list of multiples for each dimension
    :return: tf.py_func
    """
    return tf.py_func(np.repeat, [arr, repeats, 0], tf.float32)


def replace_out_of_bound(outer_arr, lower=0, upper=1, rep=-1.):
    """
    for every element x in arr, if not (lower <= x <= upper), replace with rep
    :param outer_arr: changes values contained in this tensor
    :param rep: value to replace out of range values with
    :param lower: lower bound of range to keep values
    :param upper: upper bound of range to keep values
    :return: tf.py_func
    """
    def func(arr):
        arr_ = np.array(arr)
        np.place(arr_, arr < lower, [rep])
        np.place(arr_, arr > upper, [rep])
        return arr_
    return tf.py_func(func, [outer_arr], [tf.float32])


# image augmenter files
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


def zoom(img_tensor_batch, labels_tensor, n, num_pairs=15, permutations_per_image=2):
    """
    zooms by a factor of n where n is an int.
    :param img_tensor_batch: tensor of shape (batch, rows, cols, channels=1)
    :param labels_tensor: tensor of shape (batch, (num_pairs) * 2) with (x1, y1, ..., x(num_pairs), y(num_pairs))
                            labels must be in [0, 1]
    :param n: integer for scale factor
    TODO: make scale factor support fractional values. Currently only supports zooming in
    :param num_pairs: number of point pairs. labels with shape (batch, 30) means 15 pairs
    :param permutations_per_image: number of images to generate for each image
    :return: <tensor shape=(-1, img_tensor_batch.shape[1], img_tensor_batch.shape[2], 1)>,
                                                       <tensor shape=(batch_size, num_pairs * 2)>

    1) upscale image so that it has larger dimensions
    2) find top left corner coordinates. i.e. random int in range (0, edge of scaled image - original side length)
    3) get image crop with corners at left coordinate, right_coordinate
    4) tile label to be [[x1.1, y1.1], [x1.2, y1.2], ..., [xn.1, yn.1], [xn.m, yn.m]]
        where each label is repeated m = permutations_per_image times
    5) subtract label -= ( new image corner / large_image dimension )
    6) if label not in [0, 1], replace with -1
    """

    # image
    upscale_img = upscale(img_tensor_batch, n)
    upscale_shape = tf.cast(tf.shape(img_upscaled), tf.float32)  # (batch, rows, cols, channels
    img_shape = tf.cast(tf.shape(img_tensor_batch), tf.float32)
    x_max = tf.cast(upscale_shape[2] - img_shape[2], 'int32')
    y_max = tf.cast(upscale_shape[1] - img_shape[1], 'int32')
    # lists of top left corners of  crops
    x_values = tf.random_uniform((permutations_per_image,), 0, tf.cast(x_max, tf.int32), tf.int32)
    y_values = tf.random_uniform((permutations_per_image,), 0, tf.cast(y_max, tf.int32), tf.int32)
    x_y_pairs = tf.stack([x_values, y_values], axis=1)  # shape is (permutations_per_image, 2)
    cropped = tf.map_fn(lambda x_y: tf.image.crop_to_bounding_box(
                        upscale_img, x_y[0], x_y[1], tf.cast(img_shape[1], tf.int32), tf.cast(img_shape[2], tf.int32)),
                        tf.cast(x_y_pairs, tf.int32))
    # keypoints
    x_y_pairs_repeat = tf_repeat(tf.reshape(x_y_pairs, (-1, permutations_per_image, 2)), num_pairs)
    x_y_pairs_repeat = tf.reshape(x_y_pairs_repeat, (-1, permutations_per_image * num_pairs, 2))  # set shape
    label_pairs = tf.reshape(labels_tensor, (-1, num_pairs, 2))
    labels_repeated = tf_repeat(label_pairs, permutations_per_image + 1)
    labels_repeated = tf.reshape(labels_repeated, (-1, permutations_per_image * num_pairs, 2))  # make sure shape is set
    labels_shifted = labels_repeated - (x_y_pairs_repeat / upscale_shape[0])
    # TODO: WARNING: above operation assumes that the image is a square when it divides all x_y points by large image
    #       dimension
    labels_replaced = replace_out_of_bound(labels_shifted)
    return_labels = tf.reshape(labels_replaced, (-1, num_pairs * 2))

    return cropped, return_labels




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

    >>> sess = tf.Session()
    >>> img_upscaled = upscale(img, 2)
    >>> out = sess.run(img_upscaled, {img: train_data[0:5]})
    >>> plt.imshow(out[0][:, :, 0])
    >>> plt.show()
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
        # zoom image
        img_zoomed, labels_zoomed = zoom(img, labels, 2)
        out_img, out_labels = sess.run([img_zoomed, labels_zoomed], {img: train_data, labels: train_labels})
        plt.imshow(out_img[0][:, :, 0])
        # plt.scatter
