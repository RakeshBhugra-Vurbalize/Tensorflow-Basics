import  tensorflow as tf



#Downloading Dataset

import wget
url="https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"
def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))
wget.download(url, 'HappySadClassifier/data', bar=bar_custom)

#Unzipping
import os
import zipfile

local_zip = 'HappySadClassifier/data/happy-or-sad.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('HappySadClassifier/data/happy-or-sad')
zip_ref.close()

#Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\n Reached 99% accuracy so cancelling training!\n")
            self.model.stop_training = True

callbacks = myCallback()


#Image Visualization
#using matplotlib
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg
import numpy as np
img = mpimg.imread('HappySadClassifier/data/happy-or-sad/happy/happy1-00.png')
plt.imshow(img)
plt.show()
print(np.shape(img))

#using opencv / cv2
import cv2
img = cv2.imread('HappySadClassifier/data/happy-or-sad/happy/happy1-00.png')
cv2.imshow("img", img); cv2.waitKey(0); cv2.destroyAllWindows()
print(np.shape(img))
print(img)


#Model Architecture
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation=tf.keras.activations.relu, input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),

    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.summary()

#Compile
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

#Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Rescaling Images
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

#Flow training images using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                'HappySadClassifier/data/happy-or-sad',
                target_size=(150,150),
                #batch_size=128,
                class_mode='binary')
history = model.fit(
    train_generator,
    epochs = 50,
    verbose = 1,
    callbacks=[callbacks]
)

#Check
from tensorflow.keras.preprocessing import image
path = 'HappySadClassifier/data/happy-or-sad/sad/sad1-00.png'
img = image.load_img(path, target_size=(150,150))
plt.imshow(img)
plt.show()
x=image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)
images = np.vstack([x])
classes = model.predict(images)
print(classes)
if classes[0]>0.5:
    print('Sad')
else:
    print('Happy')