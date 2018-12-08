import graphviz as graphviz
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


class MyReshape(Layer):
    """
    wraps tf.reshape in a keras layer so that output is not fixed to [batch_size] + output_shape
    """

    def __init__(self, target_shape, **kwargs):
        super(MyReshape, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs, **kwargs):
        return tf.reshape(inputs, self.target_shape)


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
           where dimension is the output dimension
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
        w_h = self.out_shape_tensor  # in_shape[-3:-1]
        new_w_h = tf.map_fn(lambda r: tf.scalar_mul(r, tf.cast(w_h, tf.float32)),
                            self.shrink_factors_tensor, name='new_w_h')
        new_w_h = tf.cast(new_w_h, tf.int32)  # shape=(self.perms, 2)
        total_pad_elems = tf.map_fn(lambda wh: self.out_shape_tensor - wh, new_w_h, name='total_pad_elems')
        # shape should = (self.perms, 2)
        # !!! - WARNING: really only good for square images
        splits = tf.map_fn(lambda tot_pad: tf.cond(tf.logical_or(tf.equal(tot_pad[0], tf.constant(0)),
                                                                 tf.equal(tot_pad[1], tf.constant(0))),
                                                   lambda: tf.constant([0, 0]),  # if the number of px to pad is 0
                                                   lambda: tf.concat(  # if there are mode than 0 px to pad, rand split
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
        k = tf.constant(1)
        img_small0 = tf.image.resize_images(inputs, new_w_h[0], align_corners=True)  # initial small image to be padded
        padded0 = tf.expand_dims(tf.reshape(tf.pad(img_small0, [[0, 0], [pad_lens[0][0][0], pad_lens[0][1][0]],
                                                                [pad_lens[0][0][1], pad_lens[0][1][1]], [0, 0]],
                                                   mode=self.padding, constant_values=self.constant,
                                                   name='pad_imgs_to_out_shape'),
                                            [-1, self.out_shape[0], self.out_shape[1], self.in_shape[-1].value]),
                                 axis=0)

        # creates initial image for the rest to be appended to. shape needs to be very specific to match shape_invariant

        def body(_i, _padded):
            # operation to preform on each batch of images per ratio
            img_small = tf.image.resize_images(inputs, new_w_h[_i], align_corners=True)
            __padded = tf.expand_dims(tf.pad(img_small, [[0, 0], [pad_lens[_i][0][0], pad_lens[_i][1][0]],
                                                         [pad_lens[_i][0][1], pad_lens[_i][1][1]], [0, 0]],
                                             mode=self.padding, constant_values=self.constant,
                                             name='pad_imgs_to_out_shape_in_loop'), axis=0)
            ret = tf.concat([__padded, _padded], axis=0, name='padded_img')
            return tf.add(_i, 1), ret

        padded = tf.while_loop(lambda _i, _: tf.less(_i, tf.constant(self.perms), name='pad_while_cond'),  # i < perms
                               body, [k, padded0],
                               shape_invariants=[k.get_shape(), tf.TensorShape([None,
                                                                                self.in_shape[0].value,
                                                                                self.out_shape[0],
                                                                                self.out_shape[1],
                                                                                self.in_shape[-1].value])],
                               name='pad_while_loop')[1]
        output_shape_cast = [self.in_shape[0].value, self.perms, self.out_shape[0],
                             self.out_shape[1], self.in_shape[-1].value]
        output_shape_cast = [-1 if _el is None else _el for _el in output_shape_cast]  # None -> -1 for tensor
        return tf.reshape(padded, output_shape_cast)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([tf.constant(-1), tf.constant(self.perms), tf.constant(self.out_shape_tensor[0]),
                               tf.constant(self.out_shape_tensor[1]), input_shape[-1]])


class ImAug:
    """
    - class that makes a keras model containing image augmentation layers specified above
    - class acts as a wrapper for tf.keras.Model
    - calls all methods that keep points constant on inputs directly and then concatenates outputs and flattens to
      the form (batch, height, width, channels)
    TODO: make it so that images with ratio 1. skip augmentation to speed up process
    TODO: does not support operations that do not keep points constant in a 2nd/4th channel
    TODO: shrinking might get rid of keypoints, expanding might add values around keypoints
    """

    def __init__(self, input_shape, output_shape, inputs=None, model_input=None, ratios=None, shrink_factors=None):
        """
        initialized layers and model
        :param input_shape: should be of the form (batch, HEIGHT, WIDTH, CHANNELS)[1:]
        :param output_shape: should be of the form (batch, HEIGHT, WIDTH, CHANNELS)[1:]
        :param ratios: floats in the range (0, 1.]. effective zoom of 1/ratio
        :param shrink_factors: floats in the range (0., 1.]. effective zoom of ratio
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        # define model
        if model_input is None:
            # self.Input is the model input
            self.Input = keras.Input(shape=input_shape)
        else:
            self.Input = model_input
        if inputs is None:
            # first input to model layers
            self.inputs = self.Input
        else:
            self.inputs = inputs
        self.point_layers = []
        # append layers that do not change pixel values (much) to output
        if ratios is not None:
            lyr = RandomCrop(ratios, output_shape[0:-1])(self.inputs)
            self.point_layers.append(lyr)
        if shrink_factors is not None:
            lyr = ZoomOut(output_shape[0:-1], shrink_factors)(self.inputs)
            self.point_layers.append(lyr)

        if len(self.point_layers) == 1:
            self.point_concat = self.point_layers[0]
        else:
            self.point_concat = keras.layers.concatenate(self.point_layers, axis=1)
        self.point_model = keras.Model(inputs=self.Input, outputs=self.point_concat)

    def __call__(self, inputs):
        """
        - something like ImAug(init_vars)(numpy_array)
        - wraps model.predict()
        :param inputs: numpy array of input images. shape(batch, ...)
        :return: numpy array
        """
        _out = self.point_model.predict(inputs)
        return np.reshape(_out, [-1] + self.output_shape)

    def get_output_tensor(self):
        """
        - returns a tensor containing the reshaped output of the model
        :return: tensor from model.outputs reshaped
        """
        return tf.reshape(self.point_model.outputs, [-1] + self.output_shape)

    def get_output_layer(self):
        """
        - returns a keras layer containing the reshaped output of the model
        - used for linking one model to another because this way does not actually export the defined model with a graph
        :return: tf.keras.layers.Reshape, self.Input
        """
        # return keras.layers.Reshape(self.output_shape)(self.point_concat), self.Input
        return MyReshape([-1] + self.output_shape)(self.point_concat), self.Input

    def summary(self):
        """
        - wraps self.model.summary
        """
        self.point_model.summary()


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

    model = ImAug([96, 96, 1], [48, 48, 1], ratios=[0.5, 0.75, 1.], shrink_factors=[0.5, 0.75, 1.])
    out_layer, model_in = model.get_output_layer()
    model2 = ImAug([48, 48, 1], [72, 72, 1], inputs=out_layer, model_input=model_in,
                   ratios=[0.5, 0.75, 1.], shrink_factors=[0.1])
    # (96, 96)
    keras.utils.plot_model(model2.point_model, 'ImAug_chain.png', show_shapes=True)
    model2.summary()
    out = model(train_data[0:5])
    for i, img in enumerate(out[-1::-1]):
        plt.imshow(img[:, :, 0])
        plt.show()
        if i > 10:
            break
