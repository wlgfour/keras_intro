# imports
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

print('done importing')

# globals
print('defining globals')

VISUALIZE = False
SAVE = True
LOAD = False
MODEL_NUMBER = 'v2.1'
BASE_DIR = f'./log_dir/{MODEL_NUMBER}'
SAVE_FILE = f'{BASE_DIR}/volcano_classifier_{MODEL_NUMBER}.h5'

# tensorboard
TBOARD_DIR = f'{BASE_DIR}/tboard_{MODEL_NUMBER}'
TBOARD_CUR_DIR = f'{TBOARD_DIR}/current'
TBOARD_HIST_DIR = f'{TBOARD_DIR}/history'
if os.path.isdir(BASE_DIR):
    if os.path.exists(SAVE_FILE):
        LOAD = True
    if os.path.isdir(TBOARD_DIR):
        # move past logfiles to history directory
        if not os.path.isdir(TBOARD_HIST_DIR):
            os.mkdir(TBOARD_HIST_DIR)
        if not os.path.isdir(TBOARD_CUR_DIR):
            os.mkdir(TBOARD_CUR_DIR)
            log_files = [f for f in os.listdir(TBOARD_CUR_DIR) if os.path.isfile(f'{TBOARD_CUR_DIR}/{f}')]
            for file in log_files:
                os.rename(f'{TBOARD_CUR_DIR}/{file}', f'{TBOARD_HIST_DIR}/{file}')


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

# model
print('initializing model')

train_imgs = train_imgs.reshape((-1, 110, 110, 1))
train_labels = to_categorical(train_labels, 2).astype(int)
test_imgs = test_imgs.reshape((-1, 110, 110, 1))
test_labels = to_categorical(test_labels, 2).astype(int)

if LOAD:
    print('    loading model')
    model = k.models.load_model(SAVE_FILE)
else:
    print('    building model')
    model = k.Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(110, 110, 1)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # (110, 110) -> (55, 55)
    model.add(Dropout(0.15))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))  # (55, 55) -> (?, ?)
    # model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
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

callbacks = list()
callbacks.append(k.callbacks.TensorBoard(log_dir=TBOARD_CUR_DIR, histogram_freq=1, batch_size=32,
                                         write_graph=True, write_grads=True, write_images=True))
if SAVE:
    callbacks.append(k.callbacks.ModelCheckpoint(SAVE_FILE, monitor='val_loss', save_best_only=True, mode='min'))


# train
print('training model')

model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), epochs=20, batch_size=32, shuffle=True,
          callbacks=callbacks)

# save
if SAVE:
    print('saving model')
    model.save(SAVE_FILE)
