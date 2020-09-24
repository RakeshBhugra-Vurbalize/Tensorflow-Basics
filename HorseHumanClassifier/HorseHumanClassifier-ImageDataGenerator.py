#Downloading Dataset

#Dataset
import wget
url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip'
def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))
wget.download(url, 'HorseHumanClassifier/data/horse-human-data.zip', bar=bar_custom)

#Validation Set
import wget
url='https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip'
wget.download(url, 'HorseHumanClassifier/data/validation-horse-human-data.zip', bar=bar_custom)

#Unzipping
import os
import zipfile

local_zip = 'HorseHumanClassifier/data/horse-human-data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('HorseHumanClassifier/data/horse-human-data')
local_zip = 'HorseHumanClassifier/data/validation-horse-human-data.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('HorseHumanClassifier/data/validation-horse-human-data')
zip_ref.close()

#Path
train_horse_dir = os.path.join('HorseHumanClassifier/data/horse-human-data/horses')
train_human_dir = os.path.join('HorseHumanClassifier/data/horse-human-data/humans')
validation_horse_dir = os.path.join('HorseHumanClassifier/data/validation-horse-human-data/horses')
validation_human_dir = os.path.join('HorseHumanClassifier/data/validation-horse-human-data/humans')

print('Total training horse images :', len(os.listdir(train_horse_dir)))
print('Total training human images :', len(os.listdir(train_human_dir)))
print('Total validation horse images :', len(os.listdir(validation_horse_dir)))
print('Total validation human images :', len(os.listdir(validation_human_dir)))

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

#Images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

#Random
img = mpimg.imread(os.path.join(train_horse_dir, train_horse_names[0]))
plt.imshow(img)
plt.show()

#Model Architecture
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation=tf.keras.activations.relu, input_shape=(300, 300, 3)),
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

#Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                'HorseHumanClassifier/data/horse-human-data',
                target_size=(300,300),
                batch_size=128,
                class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                    'HorseHumanClassifier/data/validation-horse-human-data', # This is the source directory for training images
                    target_size=(300,300), # All images will be resized to 300x300
                    batch_size=32,
                    # Since we use binary_crossentropy loss, we need binary labels
                    class_mode='binary')


history = model.fit(
    train_generator,
    steps_per_epoch = 8, #Num of images / batch size = (500+527)/128
    epochs = 15,
    verbose = 1,
    validation_data = validation_generator,
    validation_steps = 8 #Num of Valid Images/batch size of valid datagen = (128+128)/32
)


# Running the model

import numpy as np
from tensorflow.keras.preprocessing import image

path = 'HorseHumanClassifier/data/test/horse3.jpg'
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x, axis=0)
print(x.shape)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes)
if classes[0]>0.5:
    print("is a human")
else:
    print("is a horse")