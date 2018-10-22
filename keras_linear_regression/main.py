# imports
from tensorflow.keras.layers import Dense
import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
print('done importing')

SAVE = True
LOAD = False
SAVE_DIR = './save'
SAVE_FILE = f'{SAVE_DIR}/x^2.h5'
TBOARD_SAVE = f'{SAVE_DIR}/tboard'

# data generation
print('building data')
x = np.array(range(1, 2001))
y = np.array(list(map(lambda j: j ** 2, x)))
x_max = max(x)
y_max = max(y)
x = np.array([i/x_max for i in x])
y = np.array([i/y_max for i in y])
x_axis = np.array(x)
f_x = np.array(y)
x_y = np.array(list(zip(x, y)))
np.random.shuffle(x_y)
x = x_y[:, 0]
y = x_y[:, 1]
data = np.reshape(x, (-1, 1))
labels = np.reshape(y, (-1, 1))

# model
if LOAD:
    print('loading model')
    model = k.models.load_model(SAVE_FILE)
else:
    print('building model')
    model = k.models.Sequential()
    model.add(Dense(64, activation='relu', input_shape=(1,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error'
                  )

# callbacks
tboard = k.callbacks.TensorBoard(log_dir=TBOARD_SAVE, histogram_freq=1, batch_size=32,
                                 write_graph=True, write_grads=True, write_images=True)

# train and visualize
plt.plot(x_axis, f_x)
# train
print('training model')
model.fit(data, labels, epochs=5, batch_size=32, callbacks=[tboard], validation_split=0.2)

# visualize
print('plotting predictions')
labels_pred = model.predict(data)
plt.scatter(x, np.reshape(labels_pred, (-1)))
plt.show()

# save
if SAVE:
    print('saving model')
    model.save(SAVE_FILE)
