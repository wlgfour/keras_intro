# imports
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List

print('done importing')


# helpers
print('defining helpers')


class ActivationCollector(k.callbacks.Callback):
    """
    add to callbacks in model.fit to record activation maps of data after every batch in
    self.images['layer_name']: List[image]
    """
    def __init__(self, layer_names: List[str], data: np.ndarray =None, counter_mod: int =32, save_dir: str =None,
                 every_epoch: bool =True):
        """
        :param layer_names: names of layers to get. incorrect names will be left out
        :param data: image to get maps from
            # TODO: make it possible to save multiple images. Make compatible with ImageHandler.animate()
        :param counter_mod: save every counter_mod batches
        """
        super().__init__()
        print('defining image collector for intermediate layers')
        self.data = data
        self.images = dict()
        self.model_names = list()
        self.mid_stages = None
        self.layer_names = layer_names
        self.counter = 0  # only save practice images on every counter_mod batches
        self.counter_mod = counter_mod
        self.every_epoch = every_epoch
        self.save_dir = save_dir

    def set_data(self, data: np.ndarray) -> None:
        """
        alternative to declaring data data at init
        """
        self.data = data

    def set_save(self, f: str, every_epoch: bool =True) -> None:
        """
        alternative to initializing with these values. If every_epoch, save every epoch.
        .npy files will be saved in save_dir
        """
        self.every_epoch = every_epoch
        self.save_dir = f

    def save(self, f: str=None) -> None:
        """
        if f: save in f; else: save in save_dir
        save .npy files in dir as {key}.npy
        """
        if f is None:
            f = self.save_dir
        for key in self.images:
            np.save(f'{f}/{key}', self.images[key])

    def load(self, f: str) -> None:
        """
        loads .npy files in a directory into self.images[key] to be appended to
            key is most likely a layer name
        """
        save_files = [lyr for lyr in os.listdir(f) if os.path.isfile(f'{f}/{lyr}')]
        for s in save_files:
            self.images[s.replace('.npy', '')] = np.load(f'{f}/{s}')

    # --------------- Begin overrides that will be called during training ---------------------------------
    def on_train_begin(self, logs=None):
        # input layer is hard coded in and will always be saved
        self.model_names = [layer.name for layer in self.model.layers]
        for name in self.layer_names:
            if name in self.model_names:
                # add layer names to images if layer from model as np.ndarray with shape layer.output_shape
                layer_shape = self.model.get_layer(name).output_shape
                self.images[name] = np.ndarray((1,) + layer_shape[1:])
            else:
                print(f'    could not find layer with name {name}')
        # define model to pass data through
        self.mid_stages = k.Model(self.model.inputs,
                                  [self.model.get_layer(name).output for name in list(self.images.keys())])
        # reshape self.data to be in batch format
        self.data = np.reshape(self.data, (-1,) + self.model.layers[0].input_shape[1:])

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.counter_mod == 0:
            preds = self.mid_stages.predict(self.data)  # [layers, batch, width, height, channels]
            for i, key in enumerate(self.images.keys()):
                self.images[key] = np.concatenate((self.images[key], preds[i]))

    def on_epoch_end(self, epoch, logs=None):
        if self.every_epoch:
            self.save()


class Debug(object):
    """
    class that says whether various parts of program are debugging
    add flags to debug to toggle off
    DEBUG and 'str' means that str is off
    DEBUG and 'str' returns whether to execute block
    *args is a list of blacklisted blocks
    """
    def __init__(self, debug: bool, *args: str):
        super(Debug, self).__init__()
        self._flags = list(args)
        if debug:
            self._flags.append('debug')  # if DEBUG and 'debug' means debugging
            print('----------DEBUGGING----------')

    def __add__(self, other: str):
        # returns false if debug is on and  flag in flags
        # true if debug is off or other not in flags
        debugging = 'debug' in self._flags
        flag = other in self._flags
        return not(debugging and flag)


def show_imgs(imgs, labels, rows, cols):
    ax_imgs = [(plt.subplot(rows, cols, i),
                imgs[i],
                labels[i])
               for i in range(1, rows * cols + 1)]
    plt.tight_layout(pad=0.1)
    for a, img, label in ax_imgs:
        a.axis('off')
        a.set_title(label)
        a.imshow(img)
    plt.show()


# globals
print('defining globals')
DEBUG = Debug(False, 'load_data', 'train')
VISUALIZE = False
SAVE = True
LOAD = False
MODEL_NUMBER = 'v2.1'
BASE_DIR = f'./log_dir/{MODEL_NUMBER}'
SAVE_FILE = f'{BASE_DIR}/volcano_classifier_{MODEL_NUMBER}.h5'
ACT_SAVE = f'{BASE_DIR}/act_maps'
ACT_LOAD = False
act_collector = ActivationCollector(['m1.0', 'm1.1', 'm2.0', 'm2.1', 'm3.0', 'm3.1'])
act_collector.set_save(ACT_SAVE, True)
# callback instance of k.callbacks.Callback

# tensorboard and file management
print('--starting file management--')
TBOARD_DIR = f'{BASE_DIR}/tboard_{MODEL_NUMBER}'
TBOARD_CUR_DIR = f'{TBOARD_DIR}/current'
TBOARD_HIST_DIR = f'{TBOARD_DIR}/history'
if not os.path.isdir(BASE_DIR):
    print(f'    making model {MODEL_NUMBER} log directory')
    os.mkdir(BASE_DIR)
if not os.path.isdir(TBOARD_DIR):
    print('    making tensorboard log directory')
    os.mkdir(TBOARD_DIR)
# move past logfiles to history directory
if not os.path.isdir(TBOARD_HIST_DIR):
    print('    making tensorboard history directory')
    os.mkdir(TBOARD_HIST_DIR)
if not os.path.isdir(TBOARD_CUR_DIR):
    print('    making tensorboard current directory')
    os.mkdir(TBOARD_CUR_DIR)
print('    moving tensorboard log files from current directory to history')
log_files = [f for f in os.listdir(TBOARD_CUR_DIR) if os.path.isfile(f'{TBOARD_CUR_DIR}/{f}')]
for file in log_files:
    os.rename(f'{TBOARD_CUR_DIR}/{file}', f'{TBOARD_HIST_DIR}/{file}')

# check for save file
if os.path.exists(SAVE_FILE):
    print('    found save file. setting LOAD to True')
    LOAD = True
if os.path.isdir(ACT_SAVE):
    print('    found act map save file. setting ACT_LOAD to true')
    act_collector.load(ACT_SAVE)
else:
    os.mkdir(ACT_SAVE)
print('--done with file management--')


# helpers
print('defining helpers')


def show_imgs(imgs, labels, rows, cols):
    ax_imgs = [(plt.subplot(rows, cols, i),
                imgs[i],
                labels[i])
               for i in range(1, rows * cols + 1)]
    plt.tight_layout(pad=0.1)
    for a, img, label in ax_imgs:
        a.axis('off')
        a.set_title(label)
        a.imshow(img)
    plt.show()


# import data

if DEBUG + 'load_data':
    print('reading data')

    pd_train_images = pd.read_csv('data/train/train_images.csv', header=None)
    pd_train_labels = pd.read_csv('data/train/train_labels.csv')
    train_imgs = pd_train_images.values.reshape((-1, 110, 110))
    train_imgs = np.divide(train_imgs, 225)
    train_labels = pd_train_labels['Volcano?'].values

    pd_test_images = pd.read_csv('data/test/test_images.csv', header=None)
    pd_test_labels = pd.read_csv('data/test/test_labels.csv')
    test_imgs = pd_test_images.values.reshape((-1, 110, 110))
    test_imgs = np.divide(test_imgs, 225)
    test_labels = pd_test_labels['Volcano?'].values

    # visualize
    if VISUALIZE:
        print('visualizing')
        show_imgs(train_imgs, train_labels, 3, 3)

    act_collector.set_data(test_imgs[1:2])
    train_imgs = train_imgs.reshape((-1, 110, 110, 1))
    train_labels = to_categorical(train_labels, 2).astype(int)
    test_imgs = test_imgs.reshape((-1, 110, 110, 1))
    test_labels = to_categorical(test_labels, 2).astype(int)
else:
    print('skipping data load because of debug')
    test_labels = test_imgs = train_labels = train_imgs = None

# model
print('initializing model')
if LOAD:
    print('    loading model and act imgs')
    model = k.models.load_model(SAVE_FILE)
else:
    print('    building model')
    model = k.Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(110, 110, 1), name='m1.0'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m1.1'))

    model.add(MaxPool2D(pool_size=(2, 2)))  # (110, 110) -> (55, 55)
    model.add(Dropout(0.15))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m2.0'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m2.1'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # (55, 55) -> (?, ?)
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m3.0'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m3.1'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # (55, 55) -> (?, ?)
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
model.summary()


# callbacks
print('defining callbacks')
callbacks = list()
callbacks.append(k.callbacks.TensorBoard(log_dir=TBOARD_CUR_DIR, histogram_freq=1, batch_size=32,
                                         write_graph=True, write_grads=True, write_images=True))
callbacks.append(act_collector)

if SAVE:
    # save at checkpoints
    callbacks.append(k.callbacks.ModelCheckpoint(SAVE_FILE, monitor='val_loss', save_best_only=True, mode='min'))


# train
print('training model')

if DEBUG + 'train':
    model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), epochs=8, batch_size=32,
              shuffle=True,
              callbacks=callbacks)

    # save
    if SAVE:
        print('saving model and activation maps')
        act_collector.save(ACT_SAVE)
        model.save(SAVE_FILE)
else:
    print('not training because of debug setting')
