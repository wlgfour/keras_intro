import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer

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
    !!! - will output shape (batch, len(ratios), height, width, channels)
          because model.predict only supports batch size equal to input batch size
    !!! - interesting bug when passed only one image rather than an array of images
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
        super(RandomCrop, self).__init__(**kwargs)
        self.output_dim = output_shape
        self.ratios = tf.reshape(tf.constant(ratios, tf.float32), [-1, ])
        self.perms = len(ratios)
        self.crop = None
        self.crop_shapes = None
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
        new_w_h = tf.map_fn(lambda r: tf.scalar_mul(r, tf.cast(w_h, tf.float32)), self.ratios, name='new_w_h')
        new_w_h = tf.cast(new_w_h, tf.int32, name='round_new_w_h')
        self.crop_shapes = tf.map_fn(lambda wh: tf.concat([in_shape[0:-3], wh, in_shape[-1:]], axis=0),
                                     new_w_h, dtype=tf.int32, name='concat_new_shapes')  # list of [batch, h, w, c]

        # function necessary for crop_resize to have uniform image height and width
        def random_crop_resize(s):
            # s is a shape to crop to before resizing
            # crops image down and then resize to output_shape
            self.crop = tf.random_crop(inputs, s)
            self.resize = tf.image.resize_image_with_pad(self.crop, self.output_dim[0], self.output_dim[1])
            return self.resize

        self.crop_resize = tf.map_fn(lambda s: random_crop_resize(s), self.crop_shapes,
                                     dtype=tf.float32, name='map_crop_fn')
        # ^^ maps each crop shape to the input making output shape: [in[0] * len(ratios),... , in[-3:]
        self.crop_resize = tf.reshape(self.crop_resize, tf.concat([tf.constant([-1]), tf.constant([self.perms]),
                                                                   tf.constant(self.output_dim), in_shape[-1:]],
                                                                  axis=0, name='output_shape'))
        return self.crop_resize

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([tf.constant(-1), tf.constant(self.perms), tf.constant(self.output_dim[0]),
                               tf.constant(self.output_dim[1]), input_shape[-1]])  # (batch, perms, height, ...)


class ZoomOut(Layer):
    """
    zooms out of an image with various padding options
    steps:
        inputs: output_shape(height, width), shrink_factor, padding('CONSTANT', 'REFLECT', 'SYMMETRIC'), constant=0.
        -) calculate new image shapes by shrink_factor * dimension: shape=(len(shrink_factor), 2(height, width))
        -) resize to new dimensions. changes shape to (batch, len(shrink_factor), height, ...)
        -) find number of items to pad on top and bottom. output_dimension - new_dimension
        -) find random split for top vs. bottom pad. splits = random(0, num_pads)
        -) pad images s.t. side-1-pad = split, side-2-pad = number-of-elems-to-pad - split
    """

    def __init__(self, output_shape, shrink_factors, padding='SYMMETRIC', constant=0, **kwargs):
        super(ZoomOut, self).__init__(**kwargs)
        self.out_shape = output_shape  # [height, width]
        self.out_shape_tensor = tf.constant(output_shape)
        self.shrink_factors = shrink_factors
        self.shrink_factors_tensor = tf.constant(shrink_factors)
        self.perms = len(shrink_factors)
        self.padding = padding
        self.constant = constant
        self.in_shape = None

    def build(self, input_shape):
        super(ZoomOut, self).build(input_shape)
        self.in_shape = input_shape

    def call(self, inputs, **kwargs):
        in_shape = tf.shape(inputs)  # has to have shape (batch, height, width, channels)
        w_h = in_shape[-3:-1]
        new_w_h = tf.map_fn(lambda r: tf.scalar_mul(r, tf.cast(w_h, tf.float32)),
                            self.shrink_factors_tensor, name='new_w_h')
        new_w_h = tf.cast(new_w_h, tf.int32)  # shape=(self.perms, 2)
        # imgs_small = tf.map_fn(lambda wh: tf.image.resize_images(inputs, tf.cast(wh, tf.int32), align_corners=True),
        #                        tf.cast(new_w_h, tf.float32), name='small_images')
        # shape=(perms, batch, height, ...). can't reshape because some images got cropped to different sizes
        total_pad_elems = tf.map_fn(lambda wh: self.out_shape_tensor - wh, new_w_h, name='total_pad_elems')
        # shape should = (self.perms, 2)
        # !!! - WARNING: really only good for square images
        splits = tf.map_fn(lambda tot_pad: tf.cond(tf.logical_or(tf.equal(tot_pad[0], tf.constant(0)),
                                                                 tf.equal(tot_pad[1], tf.constant(0))),
                                                   lambda: tf.constant([0, 0]),
                                                   lambda: tf.concat(
                                                       [tf.random_uniform((1,), tf.constant(0), tot_pad[0], tf.int32),
                                                        tf.random_uniform((1,), tf.constant(0), tot_pad[1], tf.int32)],
                                                       axis=0)
                                                   ), total_pad_elems, name='pad_split')
        # generates a tensor of shape=(perms, 2) with random values of where to split the pads for top and bottom
        split_and_tot = tf.stack([splits, total_pad_elems], axis=1)
        pad_lens = tf.map_fn(lambda spl_tot: tf.stack([spl_tot[0], spl_tot[1] - spl_tot[0]], axis=0),
                             split_and_tot, name='width_height_l_r_pads')
        # creates a list of [[pad_elems_vertical_left, "horizontal__left], ["_right, "_right]]
        # below: while loop to pad images to return size
        k = tf.constant(0)
        condition = lambda _i, _: tf.less(_i, tf.constant(self.perms), name='pad_while_cond')  # i < perms
        # padded0 = tf.zeros(tf.TensorShape([1, self.in_shape[0].value, self.out_shape[0],
        #                                    self.out_shape[1], self.in_shape[-1].value]),
        #                    dtype=tf.int32, name='padded_imgs')
        padded0 = tf.zeros([tf.constant(1), in_shape[0], self.out_shape_tensor[0],
                            self.out_shape_tensor[1], self.in_shape[-1]])

        # tf.expand_dims(tf.reshape(tf.pad(imgs_small[0], [[0, 0], [pad_lens[0][0][0], pad_lens[0][1][0]],
        #                                                 [pad_lens[0][0][1], pad_lens[0][1][1]], [0, 0]],
        #                                 mode=self.padding, constant_values=self.constant,
        #                                 name='pad_imgs_to_out_shape0'),
        #                          [in_shape[0], self.out_shape_tensor[0],
        #                           self.out_shape_tensor[1], self.in_shape[-1]]
        #                          ), 0)

        def body(_i, _padded):
            img_small = tf.image.resize_images(inputs, new_w_h[_i], align_corners=True)
            __padded = tf.expand_dims(tf.pad(img_small, [[0, 0], [pad_lens[_i][0][0], pad_lens[_i][1][0]],
                                                         [pad_lens[_i][0][1], pad_lens[_i][1][1]], [0, 0]],
                                             mode=self.padding, constant_values=self.constant,
                                             name='pad_imgs_to_out_shape'), axis=0)
            ret = tf.concat([__padded, _padded], axis=0, name='padded_img')
            return tf.add(_i, 1), ret

        padded = tf.while_loop(condition, body, [k, padded0],
                               shape_invariants=[k.get_shape(), tf.TensorShape([None,
                                                                                self.in_shape[0].value,
                                                                                self.out_shape[0],
                                                                                self.out_shape[1],
                                                                                self.in_shape[-1].value])],
                               name='pad_while_loop')[1]
        # padded = tf.map_fn(lambda im_pad: tf.pad(im_pad[0], [[0, 0], [im_pad[1][0][0], im_pad[1][1][0]],
        #                                                [im_pad[1][0][1], im_pad[1][1][1]], [0, 0]],
        #                                           mode=self.padding, constant_values=self.constant,
        #                                           name='pad_imgs_to_out_shape0'), (imgs_small, pad_lens))
        output_shape_cast = [self.in_shape[0].value, self.perms, self.out_shape[0],
                             self.out_shape[1], self.in_shape[-1].value]
        output_shape_cast = [-1 if _el is None else _el for _el in output_shape_cast]  # None -> -1 for tensor
        return tf.reshape(padded[1:], output_shape_cast)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([tf.constant(-1), tf.constant(self.perms), tf.constant(self.out_shape_tensor[0]),
                               tf.constant(self.out_shape_tensor[1]), input_shape[-1]])


class ImAug(keras.Model):
    """class that inherits keras.model and creates an image augmentation model that outputs"""

    def __init__(self, output_shape, ratios=None, shrink_factors=None):
        """

        :param output_shape: has shape [output_height, output_width]
        :param ratios: list of ratios to crop to (ratio=0.5 = 2x zoom). will expand dataset by factor of len(ratios)
        """
        super(ImAug, self).__init__()
        self.point_layers = []
        if ratios is not None:
            lyr = RandomCrop(ratios, output_shape)
            self.point_layers.append(lyr)
        if shrink_factors is not None:
            lyr = ZoomOut(output_shape, shrink_factors)
            self.point_layers.append(lyr)

    def call(self, inputs, training=None, mask=None):
        # https://www.tensorflow.org/api_docs/python/tf/keras/models/Model
        # define forward pass of the network
        outputs = []
        p_layer = inputs
        for layer in self.point_layers:
            outputs.append(layer(p_layer))
            # p_layer = layer
        return tf.concat(outputs, axis=1)


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

    plt.imshow(train_data[0, :, :, 0])
    plt.show()

    model = ImAug([96, 96], ratios=[0.5, 0.75, 1.], shrink_factors=[0.5, 0.75, 1.])
    # (96, 96)
    out = model.predict(train_data[0:3])
    for i in range(len(out[0])):
        plt.imshow(out[0, i, :, :, 0])
        plt.show()
    model.summary()
