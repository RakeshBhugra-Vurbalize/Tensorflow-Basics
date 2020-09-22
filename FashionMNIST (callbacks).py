    import tensorflow as tf
print(tf.__version__)

#Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.8):
            print("\n Reached 80% accuracy so cancelling training!\n")
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
plt.imshow(training_images[5])
plt.show()
print(training_labels[5])
print(training_images[5])

#Normalize the data
training_images = training_images/255.0
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

model.evaluate(x=test_images, y=test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])

print(classifications[2])
print(test_labels[2])

