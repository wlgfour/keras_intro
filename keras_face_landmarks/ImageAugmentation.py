import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.engine.topology import Layer

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
Defines:
    1) define model to connect augmentation layers
    2) define keras layers that augment the images and point maps
    3) define class to handle the images and convert to and from point maps. run the model
        - hold 2 models, one for point maps and images and images and one for just images
Challenges / Unresolved Problems:
    1) what shape do images and points go in?
        - how are images and points ordered?
        - points masks will be added to image as an appended channel
"""


class RandomCrop(Layer):
    """
    layer that inherits keras layer and randomly crops and resize to output_size size a batch of images. points will be
        cropped the same as images
    supports:
        - list of ratios in (0, 1) - 1/2 will be half the original image resize to original (i.e. 2x zoom)
        - output size
    Usage:
        - pass ratio as a single element s.t. el * [height, width] = output_shape
            will act as aa regular random crop
        - pass list of ratios and output_size s.t. output_size = original_dims
            will randomly zoom random sections of the image by a factor of 1 / ratio
        - pass ratio [0.5, 0.5, 0.2]
            will have 2:1 images cropped to 0.5:0.2
        - pass output_shape > input_dim and ratio = 1
            will make the image larger and scale to new size
    """
    def __init__(self, ratios, output_shape, **kwargs):
        # output shape should have shape [rows, cols]. i.e. [output_height, output_width]
        self.output_dim = output_shape
        super(RandomCrop, self).__init__(**kwargs)
        self.ratios = tf.reshape(tf.constant(ratios), [-1,])
        self.crop = None
        self.new_shapes = None
        self.resize = None
        self.crop_resize = None

    def build(self, input_shape):
        # trainable things are initialized here
        # no trainable values to be initialized here
        super(RandomCrop, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # operations to call
        # inputs must be of shape (..., rows, cols, channels)
        in_shape = tf.shape(inputs)
        w_h = in_shape[-3:-1]
        new_w_h = tf.map_fn(lambda r: tf.scalar_mul(r, w_h), self.ratios)
        self.new_shapes = tf.map_fn(lambda wh: tf.concat([in_shape[0:-3], wh, in_shape[-1:]], axis=0), new_w_h)

        def random_crop_resize(s):
            # s is a shape to crop to before resizing
            # crops image down and then resize to output_shape
            self.crop = tf.random_crop(inputs, s)
            self.resize = tf.image.resize_image_with_pad(self.crop, self.output_dim[0], self.output_dim[1])
            return self.resize

        self.crop_resize = tf.map_fn(lambda s: random_crop_resize(s), self.new_shapes)
        # ^^ maps each crop shape to the input making output shape: [in[0] * len(ratios),... , in[-3:]
        return self.crop_resize

    def compute_output_shape(self, input_shape):
        output_shape_tensor = tf.reshape(tf.constant(self.output_dim), [-1,])
        return tf.concat([input_shape[0:-3], self.output_dim, input_shape[-1]], axis=0)


class ImAug(k.Model):
    """
    class that inherits keras.model and creates an image augmentation model that outputs
    """
    def __init__(self):
        super(ImAug, self).__init__()
        # define a bunch of self.layers

    def call(self, inputs, training=None, mask=None):
        # https://www.tensorflow.org/api_docs/python/tf/keras/models/Model
        # define forward pass of the network
        pass


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
