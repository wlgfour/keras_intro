import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dropout, Flatten, Dense
# from training_arrays
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.generic_utils import make_batches
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.generic_utils import slice_arrays

try:
    from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
    issparse = None

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
        return tf.TensorShape([tf.constant(-1), tf.constant(self.output_dim[0]),
                               tf.constant(self.output_dim[1]), input_shape[-1]])


class ImAug(keras.Model):
    """class that inherits keras.model and creates an image augmentation model that outputs"""

    def __init__(self):
        super(ImAug, self).__init__()
        # define a bunch of self.layers

    def call(self, inputs, training=None, mask=None):
        # https://www.tensorflow.org/api_docs/python/tf/keras/models/Model
        # define forward pass of the network
        pass


"""
def custom_predict_loop(model, inputs, batch_size=32, verbose=0, steps=None):
    
    #mostly copied from training_arrays.py. want to be able to output larger batch than input
    
    model._make_predict_function()
    f = model.predict_function

    if model.uses_learning_phase and not isinstance(tf.python.keras.backend.learning_phase(), int):
        ins = inputs + [0]
    else:
        ins = inputs

    num_samples = training_utils.check_num_samples(
        inputs, batch_size, steps, 'steps')
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    indices_for_conversion_to_dense = []
    for i in range(len(model._feed_inputs)):
        if (issparse is not None and issparse(inputs[i]) and
                not tf.python.keras.backend.is_sparse(model._feed_inputs[i])):
            indices_for_conversion_to_dense.append(i)
    # Sample-based predictions.
    outs = []
    batches = make_batches(num_samples, 32)  # changed last param from batch_size to 32
    index_array = np.arange(num_samples)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        if ins and isinstance(ins[-1], int):
            # Do not slice the training phase flag.
            ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
        else:
            ins_batch = slice_arrays(ins, batch_ids)
        for i in indices_for_conversion_to_dense:
            ins_batch[i] = ins_batch[i].toarray()

        batch_outs = f(ins_batch)
        if not isinstance(batch_outs, list):
            batch_outs = [batch_outs]
        if batch_index == 0:
            # Pre-allocate the results arrays.
            for batch_out in batch_outs:
                shape = batch_out.shape  # edited from: (num_samples,) + batch_out.shape[1:]
                outs.append(np.zeros(shape, dtype=batch_out.dtype))
        for i, batch_out in enumerate(batch_outs):
            outs[i][batch_start:batch_end] = batch_out
        if verbose == 1:
            progbar.update(batch_end)
    if len(outs) == 1:
        return outs[0]
    return outs


def predict(model, x=None, batch_size=None, verbose=0,
            steps=None, max_queue_size=10, workers=1, use_multiprocessing=False):
    # - custom predict loop for model
    #https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/engine/training.py
    #- line: 1778
    
    x, _, _ = model._standardize_user_data(x, check_steps=True, steps_name='steps', steps=steps)
    return custom_predict_loop(model, x, batch_size=batch_size, verbose=verbose, steps=steps)
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

    plt.imshow(train_data[0, :, :, 0])
    plt.show()

    model = keras.Sequential()
    # (96, 96)
    model.add(RandomCrop([0.75, 0.85, 0.95], [96, 96], input_shape=(96, 96, 1)))
    model.summary()
    out = model.predict(train_data[0:1])
    for i in range(3):
        plt.imshow(out[0, i, :, :, 0])
        plt.show()
    print(np.shape(out))
