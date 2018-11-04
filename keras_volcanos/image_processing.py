import os
import numpy as np
from typing import Tuple
import math
import matplotlib.pyplot as plt


class ImageHandler:
    def __init__(self):
        self.images = dict()
        self.imgs_as_frames = dict()

    def load(self, f: str) -> None:
        """
        loads .npy files from f into a dict with key 'filename' - '.npy'
        """
        save_files = [file for file in os.listdir(f) if os.path.isfile(f'{f}/{file}')]
        for s in save_files:
            self.images[s.replace('.npy', '')] = np.load(f'{f}/{s}')

    def animate(self, key: str, save: str) -> None:
        """
        animates key from images_as_frames and saves to save directory
        """
        pass  # TODO: animate self.imgs_as_frames

    def show(self, key):
        # TODO: change show to get_fig
        # TODO: annotate fig with frame number and key as title. remove axises
        for img in self.imgs_as_frames[key]:
            plt.imshow(img)
            plt.show()

    def imgs_to_frames(self, aspect_ratio: Tuple[int, int], filler: int =1) -> None:
        """
        convert images to frames with aspect_ratio in (images-x, images-y)
        filler is the empty space value
        """
        # TODO: very inefficient to copy the np array every time it is change. init as large as it will be
        for key in self.images:
            # many images in each key
            self.imgs_as_frames[key] = list()
            for img in self.images[key]:
                subframe_shape = np.shape(img)
                empty_frame = np.ndarray((subframe_shape[0], subframe_shape[1]))  # will be filled and padded
                empty_frame.fill(filler)
                empty_frame = np.pad(empty_frame, (1, 1), mode='constant', constant_values=filler)
                row_length = math.floor(subframe_shape[2] * (aspect_ratio[0] / (aspect_ratio[0] * aspect_ratio[1])))
                col_length = math.floor(subframe_shape[2] / row_length)
                # number_of_images * (row_aspect / total_aspect)
                # make rows and append rows to each other
                counter = 0
                cols = None
                for k in range(col_length):  # every column
                    if counter < subframe_shape[2]:
                        row = img[:, :, counter]  # first item in row
                        row = np.pad(row, (1, 1), mode='constant', constant_values=filler)  # pad edges
                    else:
                        row = empty_frame
                    counter += 1  # next channel
                    for i in range(1, row_length):  # for the rest of the row
                        if counter < subframe_shape[2]:
                            channel_frame = img[:, :, counter]  # removes channel dimension
                            channel_frame = np.pad(channel_frame, (1, 1), mode='constant', constant_values=filler)
                            row = np.append(row, channel_frame, axis=1)  # append on the right of the row
                        else:
                            row = np.append(row, empty_frame, axis=1)
                        counter += 1  # next channel
                    if k == 0:  # first row
                        cols = row
                    else:
                        cols = np.append(cols, row, axis=0)
                self.imgs_as_frames[key].append(cols)


if __name__ == '__main__':
    print('running image_processing main')
    img_processor = ImageHandler()
    img_processor.load('./log_dir/v2.2_images/act_maps_old')
    img_processor.imgs_to_frames((3, 3))
    img_processor.show('m2.0')
