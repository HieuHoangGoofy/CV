import tensorflow as tf
from keras.src.saving import load_model

# Load the saved model
model=load_model('fashion_mnist_model_Acc96%.h5')

data = tf.keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = data.load_data()
test_images = test_images / 255.0

# Make predictions using the loaded model
predictions = model.predict(test_images)

# Display the first prediction and true label
print(predictions[0])
print(test_labels[0])
