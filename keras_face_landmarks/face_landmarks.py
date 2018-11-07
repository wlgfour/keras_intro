import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as k
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
import tensorflowjs as tfjs
from Helpers import ActivationCollector, FileArchitecture, ImageHandler


# helpers
print('defining helpers')


def visualize(images, labels, number):
    for inner_i in range(number):
        plt.imshow(images[inner_i])
        x = labels[inner_i][0::2]
        y = labels[inner_i][1::2]
        plt.scatter(x, y, c='r')
        plt.show()


# globals
print('defining globals')
VISUALIZE = False
VISUALIZE_TRAIN = True
VISUALIZE_PREDICTIONS = True
# execute blocks
TRAIN = False
CHKPT_SAVE = True
SAVE = True
PREDICT = False

model_number = 'v1.1'
files = FileArchitecture(model_number, f'./log_dir/{model_number}', 'face_keypoints')
files.construct_file_tree()
act_collector = ActivationCollector(['m1.0', 'm1.1', 'm2.0', 'm2.1', 'm3.0', 'm3.1'],
                                    save_dir=files.act_save, counter_mod=16)

# load
print('loading data')
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

if VISUALIZE:
    visualize(data, pnts, 10)

train_data = data * (1 / data.max())  # scale to [0, 1]
train_data = train_data.reshape((-1, 96, 96, 1))
train_labels = pnts[:, 1:]
train_labels = train_labels * (1 / np.shape(data)[1])  # scale so scaled * image_width = original position
act_collector.set_data(train_data[1:2])
print(f'    images -- shape: {np.shape(train_data)}  max: {train_data.max()}  min: {train_data.min()}')
print(f'    labels -- shape: {np.shape(train_labels)}  max: {train_labels.max()}  min: {train_labels.min()}')

if VISUALIZE_TRAIN:  # visualize training data to verify that scaling works
    visualize(train_data[:, :, :, 0], train_labels * np.shape(data)[1], 2)


# model
print('initializing model')
if files.load:
    print('    found save file: loading model')
    model = k.models.load_model(files.save_file)
else:
    # TODO: data augmentation layer
    print('    building model')
    model = k.Sequential()
    # (96, 96)
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(96, 96, 1), name='m1.0'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', name='m1.1'))
    model.add(MaxPool2D((2, 2)))  # (96, 96) -> (48, 48)
    model.add(Dropout(0.15))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='m2.0'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='m2.1'))
    model.add(MaxPool2D((2, 2)))  # (48, 48) -> (24, 24)

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='m3.0'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', name='m3.1'))
    model.add(MaxPool2D((2, 2)))  # (24, 24) -> (12, 12)

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(30, activation=None))

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=[]
                  )
model.summary()

# callbacks
print('defining callbacks')
callbacks = list()
callbacks.append(k.callbacks.TensorBoard(log_dir=files.tboard_cur_dir, histogram_freq=1, batch_size=32,
                                         write_graph=True, write_grads=True, write_images=True))
callbacks.append(act_collector)
if CHKPT_SAVE:
    # checkpoint saves
    callbacks.append(k.callbacks.ModelCheckpoint(files.save_file, monitor='val_loss', save_best_only=True, mode='min'))

# train model
if TRAIN:
    print('entering training phase')
    model.fit(train_data, train_labels, validation_split=0.2, epochs=5, batch_size=16,
              shuffle=True, callbacks=callbacks)
# predictions
if PREDICT:
    print('predicting')
    predict_labels = model.predict(train_data, verbose=1)
    predict_labels = predict_labels * (np.shape(data)[1])
    if VISUALIZE_PREDICTIONS:
        visualize(train_data[:, :, :, 0], predict_labels, 2)

if SAVE:
    print('saving model, activation maps, and javascript model')
    if TRAIN:
        act_collector.save(files.act_save)
        model.save(files.save_file)
    ImageHandler(files.act_save, (3, 3), -1, files.base_dir, 'keras_face_landmarks')
    tfjs.converters.save_keras_model(model, files.js_dir)
