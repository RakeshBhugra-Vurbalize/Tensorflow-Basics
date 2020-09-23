import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

#Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\n Reached 80% accuracy so cancelling training!\n")
            self.model.stop_training = True

callbacks = myCallback()

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(training_images[0])
plt.show()
print(training_labels[0])

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                         tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.fit(x=training_images, y=training_labels  ,epochs=50, callbacks=[callbacks])

classification = model.predict(test_images)
print(np.argmax(classification[0]))
print(test_labels[0])

print(np.argmax(classification[5]))
print(test_labels[5])
