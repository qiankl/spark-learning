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
original_dir='image/PetImages'
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

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False, #we are going to remove the top layer, VGG was trained for 1000 classes, here we only have two
                  input_shape=(150, 150, 3))

conv_base.summary()
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
base_dir = 'image' 
train_dir = os.path.join(base_dir, 'trainS') 
val_dir = os.path.join(base_dir, 'valS') 
test_dir = os.path.join(base_dir, 'testS')

datagen = ImageDataGenerator(rescale=1./255) 
batch_size = 20

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(150, 150, 3))

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512)) # we use the last layer of VGG
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory, 
        target_size=(150, 150), 
        batch_size=batch_size, 
        class_mode='binary')
    i=0
    print ("before for loop")
    
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch 
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000) 
validation_features, validation_labels = extract_features(val_dir, 1000) 
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

#we use the features and labels we got from VGG which are given in input 
#to fit instead of the input.
history = model.fit(train_features, train_labels, 
                    epochs=10,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))