import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ImageAugmentation import *


# importing and handling data
IMPORT_DATA = True
SLICE_DATA = True
SLICE = 10  # minimum = PLOT_NUMBER

# plot information
PLOT_NUMBER = 1

# modules to test
TEST_PointsToHeatmap = True
TEST_ImAug = True


if IMPORT_DATA:
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
    train_labels = np.reshape(pnts[:, 1:], (-1, 15, 2))
    train_labels = train_labels * (1 / np.shape(data)[1])  # scale so scaled * image_width = original position
else:
    # create test data with batch_size = 2
    train_labels = np.random.randint(0, 97, (2, 15, 2)) * (1 / 96)
    train_data = np.random.randint(0, 256, (2, 96, 96, 1)) * (1 / 255)

if SLICE_DATA:
    train_labels = train_labels[0:SLICE]
    train_data = train_data[0:SLICE]

# plot original images
for i in range(PLOT_NUMBER):
    plt.imshow(train_data[i, :, :, 0])
    plt.scatter(train_labels[i, :, 0] * 96, train_labels[i, :, 1] * 96, c='r')
    plt.show()

if TEST_PointsToHeatmap:
    print('TESTING: PointsToHeatmap layer')
    inputs = keras.Input(shape=(15, 2))
    layer = PointsToHeatmap((96, 96), 3)(inputs)
    model = keras.Model(inputs=inputs, outputs=layer)
    # keras.utils.plot_model(model, 'test.png', show_shapes=True)
    print(f'    input  shape: {np.shape(train_labels)}')
    out = model.predict(train_labels)
    print(f'    output shape: {np.shape(out)}')
    for i in range(PLOT_NUMBER):
        plt.imshow(np.sum(out[i, :, :, :], axis=-1))
        plt.show()

if TEST_ImAug:
    print('TESTING: ImAug models')
    print('    model1 has halves the input image')
    model1 = ImAug([96, 96, 1], [48, 48, 1], ratios=[0.85, 0.9, 1.], shrink_factors=[0.85, 0.9, 1.],
                   pad_mode='CONSTANT')
    print('    model2 increases the input_image and takes model1 as input')
    model2 = ImAug([48, 48, 1], [72, 72, 1], p_im_aug=model1,
                   ratios=[0.7, 0.9, 1.], shrink_factors=[0.85])
    # (96, 96)

    out_layer, model_in = model2.get_output_layer()
    # keras.utils.plot_model(model, 'ImAug_chain.png', show_shapes=True)
    print(f'    input  shape: {np.shape(train_data)}')
    out = model2(train_data)
    print(f'    output shape: {np.shape(out)}')
    for i, img in enumerate(out[-1::-1]):
        if i == PLOT_NUMBER:
            break
        plt.imshow(img[:, :, 0])
        plt.show()
