import tensorflow as tf
import numpy as np

#Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.85):
            print("\n Reached 85% accuracy so cancelling training!\n")
            self.model.stop_training = True
callbacks = myCallback()

#Load Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Plot Image
import matplotlib.pyplot as plt
print(np.shape(training_images[0]))
plt.imshow(training_images[0])
plt.show()
print(training_labels[0])

#Normalizing
np.shape(training_images)
np.shape(training_images[0])
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images/255
test_images=test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28,28,1), filters=64, kernel_size=(3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics='accuracy')

model.fit(x=training_images, y=training_labels, callbacks=[callbacks], epochs=10)

results = model.evaluate(x=test_images, y=test_labels)

#Visualising
print(test_labels[:100])

import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

plt.show()