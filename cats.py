import warnings
warnings.filterwarnings('ignore')
import os, shutil
from pyspark import SparkConf, SparkContext
from PIL import Image
from pyspark import SparkFiles

conf = SparkConf().setAppName("catsanddogs").setMaster("yarn")
sc = SparkContext(conf=conf)
sc.addPyFile("hdfs:///user/hduser/Python.zip")
sc.addFile("hdfs:///user/hduser/pet.zip")
import zipfile
zip_ref = zipfile.ZipFile(SparkFiles.get('pet.zip'), 'r')
zip_ref.extractall('image')
zip_ref.close()

base_dir = 'image'
original_dir='image/Petimages'
train_dir = os.path.join(base_dir, 'trainS') 
os.mkdir(train_dir)
val_dir = os.path.join(base_dir, 'valS') 
os.mkdir(val_dir)
test_dir = os.path.join(base_dir, 'testS')
os.mkdir(test_dir)

train_cats_dir=os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir=os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
test_cats_dir=os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir=os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
val_cats_dir=os.path.join(val_dir, 'cats')
os.mkdir(val_cats_dir)
val_dogs_dir=os.path.join(val_dir, 'dogs')
os.mkdir(val_dogs_dir)

#the following script copy the first 1000 cat images to the training directoy. 
fnames = ['{}.jpg'.format(i) for i in range(1000,3000)]
cat_dir=os.path.join(original_dir, 'Cat')
dog_dir=os.path.join(original_dir, 'Dog')
for fname in fnames:
    src = os.path.join(cat_dir, fname)
    dst = os.path.join(train_cats_dir, 'cat.'+fname)
    shutil.copyfile(src, dst)
    
fnames = ['{}.jpg'.format(i) for i in range(3000,3500)]
for fname in fnames:
    src = os.path.join(cat_dir, fname)
    dst = os.path.join(test_cats_dir, 'cat.'+fname)
    shutil.copyfile(src, dst)
    
fnames = ['{}.jpg'.format(i) for i in range(3500,4000)]
for fname in fnames:
    src = os.path.join(cat_dir, fname)
    dst = os.path.join(val_cats_dir, 'cat.'+fname)
    shutil.copyfile(src, dst)
    
fnames = ['{}.jpg'.format(i) for i in range(1000,3000)]
for fname in fnames:
    src = os.path.join(dog_dir, fname)
    dst = os.path.join(train_dogs_dir, 'dog.'+fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(3000,3500)]
for fname in fnames:
    src = os.path.join(dog_dir, fname)
    dst = os.path.join(test_dogs_dir, 'dog.'+fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(3500,4000)]
for fname in fnames:
    src = os.path.join(dog_dir, fname)
    dst = os.path.join(val_dogs_dir, 'dog.'+fname)
    shutil.copyfile(src, dst)

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for
     fname in os.listdir(train_cats_dir)]

img_path = fnames[3] 
img = image.load_img(img_path, target_size=(150, 150))

#this code shows the distorted images
x = image.img_to_array(img) 
x = x.reshape((1,) + x.shape) 

i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255) #we do not augment the test set

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)

