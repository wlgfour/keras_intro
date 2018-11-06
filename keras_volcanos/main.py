# imports
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Helpers import Debug, ActivationCollector, FileArchitecture, ImageHandler

print('done importing')


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


# globals
print('defining globals')
DEBUG = Debug(False, 'load_data', 'train')
VISUALIZE = False
model_number = 'm2.3_helpers_test'
files = FileArchitecture(model_number, f'./log_dir/{model_number}')
SAVE = True
act_collector = ActivationCollector(['m1.0', 'm1.1', 'm2.0', 'm2.1', 'm3.0', 'm3.1'])
act_collector.set_save(files.act_save, True)

files.construct_file_tree()

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
print('initializing model and/or loading')
if files.act_load:
    print('    found activation maps. loading into activation_collector')
    act_collector.load(files.act_save)
if files.load:
    print('    loading model and act imgs')
    model = k.models.load_model(files.save_file)
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
callbacks.append(k.callbacks.TensorBoard(log_dir=files.tboard_cur_dir, histogram_freq=1, batch_size=32,
                                         write_graph=True, write_grads=True, write_images=True))
callbacks.append(act_collector)

if SAVE:
    # save at checkpoints
    callbacks.append(k.callbacks.ModelCheckpoint(files.save_file, monitor='val_loss', save_best_only=True, mode='min'))


# train
print('training model')

if DEBUG + 'train':
    model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), epochs=10, batch_size=32,
              shuffle=True,
              callbacks=callbacks)

    # save
    if SAVE:
        print('saving model and activation maps')
        act_collector.save(files.act_save)
        model.save(files.save_file)
        ImageHandler(files.act_save, (3, 3), -1, files.base_dir)


else:
    print('not training because of debug setting')

