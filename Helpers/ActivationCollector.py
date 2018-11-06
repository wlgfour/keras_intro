import tensorflow.keras as k
import numpy as np
import os
from typing import List


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
