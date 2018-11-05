import os
import numpy as np
from typing import Tuple
import math
import matplotlib.pyplot as plt
import matplotlib.animation as a
import imageio


class ImageHandler:
    """
    in order to function fully, call:
        -load
        -set_save
        -imgs_to_frames
    """
    def __init__(self):
        self.images = dict()
        self.imgs_as_frames = dict()
        self.save_dir = None
        self.project_path = None
        self.srcs_file = '../src/srcs.txt'
        self.project = 'keras_volcanos'

    def load(self, f: str) -> None:
        """
        loads .npy files from f into a dict with key 'filename' - '.npy'
        """
        save_files = [file for file in os.listdir(f) if os.path.isfile(f'{f}/{file}')]
        for s in save_files:
            self.images[s.replace('.npy', '')] = np.load(f'{f}/{s}')

    def set_save(self, directory: str, project: str =None) -> None:
        """
        sets self.project if project is provided
        sets save directory and creates an animations subdirectory
        sets project root directory in order to find srcs/src.txt file
        """
        if project is not None:
            self.project = project
        self.save_dir = directory
        if not os.path.isdir(f'{self.save_dir}/animations'):
            os.mkdir(f'{self.save_dir}/animations')
        self.save_dir = self.save_dir + '/animations'
        self.project_path = self.project + self.save_dir[1:]  # removes initial '.' and provides a root project dir

    def animate(self, key: str) -> None:
        """
        animates key from images_as_frames and saves to save directory
        """
        imgs = self.imgs_as_frames[key]
        for i, img in enumerate(imgs):
            img *= 255.0/img.max()
            img = img.astype('uint8')
            imgs[i] = img
        imageio.mimsave(f'{self.save_dir}/{key}.gif', imgs, duration=0.2)
        f = open(self.srcs_file, 'r')
        lines = f.readlines()
        f.close()
        # update srcs/srcs.txt
        f = open(self.srcs_file, 'w')
        start = -1
        end = -1
        for i, line in enumerate(lines):
            if f'START::{self.project_path}\n' in line:
                start = i
            elif f'END::{self.project_path}\n' in line:
                end = i
        if (start != -1 and end != -1) and f'{self.project_path}/{key}.gif\n' not in lines[start:end]:
            lines = lines[:start+1] + [f'{self.project_path}/{key}.gif\n'] + lines[start+1:]
        elif start == -1 and end == -1:
            lines.append(f'START::{self.project_path}\n')
            lines.append(f'{self.project_path}/{key}.gif\n')
            lines.append(f'END::{self.project_path}\n')
        else:
            f.writelines(lines)
            f.close()
            return
        f.writelines(lines)
        f.close()

    def show(self, key):
        for img in self.imgs_as_frames[key]:
            plt.imshow(img)
            plt.show()

    def imgs_to_frames(self, aspect_ratio: Tuple[int, int], filler: int = -1) -> None:
        """
        convert images to frames with aspect_ratio in (images-x, images-y)
        filler is the empty space value
        if filler == -1, use max as the filler
        """
        # TODO: very inefficient to copy the np array every time it is change. init as large as it will be
        for key in self.images:
            # many images in each key
            self.imgs_as_frames[key] = list()
            for img in self.images[key]:
                if filler == -1:  # set filler to the max of the image
                    filler = img.max()
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
    VERSION = 'v2.2_images'
    print('running image_processing main')
    img_processor = ImageHandler()
    img_processor.load(f'./log_dir/{VERSION}/act_maps')
    img_processor.imgs_to_frames((3, 3))
    # img_processor.show('m2.0')
    img_processor.set_save(f'./log_dir/{VERSION}')
    for img_key in img_processor.imgs_as_frames.keys():
        img_processor.animate(img_key)
