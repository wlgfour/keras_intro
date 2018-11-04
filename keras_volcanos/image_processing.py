import os
import numpy as np


class ImageHandler:
    def __init__(self):
        self.images = dict()

    def load(self, f):
        save_files = [file for file in os.listdir(f) if os.path.isfile(f'{f}/{file}')]
        for s in save_files:
            self.images[s.replace('.npy', '')] = np.load(s)
