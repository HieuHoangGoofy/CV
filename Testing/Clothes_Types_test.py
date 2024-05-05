import tensorflow as tf

class Desire_Accuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.96):
            self.model.stop_training = True
callback = Desire_Accuracy()
data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
layer_input = tf.keras.layers.Flatten(input_shape=(28, 28))
layer_hidden = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
layer_output = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
model = tf.keras.models.Sequential([layer_input, layer_hidden, layer_output])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=100, callbacks = [callback])
model.save("fashion_mnist_model_Acc96%.h5")
model.evaluate(test_images, test_labels)
predictions = model.predict(test_images)
print(predictions[0])
print(test_labels[0])