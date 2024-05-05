import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

layer = Dense(units=1, input_shape=[1])
model = Sequential([layer])
model.compile(optimizer='sgd', loss='mean_squared_error')
#Y = 3X - 5
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
model.fit(xs, ys, epochs=1000)
print("w and b are ", format(layer.get_weights()))
print(model.predict(np.array([6.0])))