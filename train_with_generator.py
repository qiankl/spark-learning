#!/usr/bin/env python
# coding: utf-8

# # Food Classification with Deep Learning in Keras / Tensorflow
# ## *Computer, what am I eating anyway?*

# ## Experiment

# ### Loading and Preprocessing Dataset

# Let's import all of the packages needed for the rest of the notebook:

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

get_ipython().run_line_magic('matplotlib', 'inline')

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model


# In[2]:


sc.stop()


# In[3]:


from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

# Create Spark context
conf = SparkConf().setAppName('Spark_MLP').setMaster('yarn')
sc = SparkContext(conf=conf)


# Download the dataset and extract it within the notebook folder. It may be easier to do this in a separate terminal window.

# A `multiprocessing.Pool` will be used to accelerate image augmentation during training.

# In[ ]:


# import multiprocessing as mp

# num_processes = 6
# pool = mp.Pool(processes=num_processes)


# We need maps from class to index and vice versa, for proper label encoding and pretty printing.

# In[4]:


class_to_ix = {}
ix_to_class = {}
with open('../food-101/meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))


# The Food-101 dataset has a provided train/test split. We want to use this in order to compare our classifcation performance with other implementations.

# In[5]:


# Only split files if haven't already

if not os.path.isdir('../food-101/test') and not os.path.isdir('../food-101/train'):

    def copytree(src, dst, symlinks = False, ignore = None):
        if not os.path.exists(dst):
            os.makedirs(dst)
            shutil.copystat(src, dst)
        lst = os.listdir(src)
        if ignore:
            excl = ignore(src, lst)
            lst = [x for x in lst if x not in excl]
        for item in lst:
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if symlinks and os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass # lchmod not available
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def generate_dir_file_map(path):
        dir_files = defaultdict(list)
        with open(path, 'r') as txt:
            files = [l.strip() for l in txt.readlines()]
            for f in files:
                dir_name, id = f.split('/')
                dir_files[dir_name].append(id + '.jpg')
        return dir_files

    train_dir_files = generate_dir_file_map('food-101/meta/train.txt')
    test_dir_files = generate_dir_file_map('food-101/meta/test.txt')


    def ignore_train(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = train_dir_files[subdir]
        return to_ignore

    def ignore_test(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = test_dir_files[subdir]
        return to_ignore

    copytree('../food-101/images', '../food-101/test', ignore=ignore_train)
    copytree('../food-101/images', '../food-101/train', ignore=ignore_test)
    
else:
    print('Train/Test files already copied into separate folders.')


# We are now ready to load the training and testing images into memory. After everything is loaded, about 80 GB of memory will be allocated.
# 
# Any images that have a width or length smaller than `min_size` will be resized. This is so that we can take proper-sized crops during image augmentation.

# In[6]:


from PIL import Image
x=np.array(Image.open('/home/hduser/food/food-101/train/apple_pie/208041.jpg').resize((150,150)))
x.shape


# In[7]:


from PIL import Image
def load_images(root，num=20):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        n=0
        for img_name in imgs:
            if n < num:
                img_arr = np.array(Image.open(join(root, subdir, img_name)).resize((200,200)))
                img_arr_rs = img_arr
                try:
                    if img_arr_rs.shape!=(200,200,3):
                        continue
                    all_imgs.append(img_arr_rs)
                    all_classes.append(class_ix)
                    n+=1
                except:
                    invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)
    


# In[8]:


X_test, y_test = load_images('../food-101/test',50)
X_train, y_train = load_images('../food-101/train',20)


# In[9]:


print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)


# In[10]:


from keras.utils.np_utils import to_categorical

n_classes = 101
y_train_cat = to_categorical(y_train, num_classes=n_classes,dtype='int')
y_test_cat = to_categorical(y_test, num_classes=n_classes,dtype='int')


# In[11]:


# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, y_train_cat)


# In[17]:


from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='SAME',
                        input_shape=(200, 200, 3))) #we need to specify the size of images, 150 x 150 in our case
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='SAME'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu',padding='SAME'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(101, activation='softmax'))


# In[18]:


model.summary()


# In[19]:


from keras import optimizers
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])


# In[20]:


spark_model = SparkModel(model, mode='synchronous')


# In[21]:


spark_model.fit(rdd, epochs=2, batch_size=2, verbose=1, validation_split=0.1)


# In[22]:


score = spark_model.master_network.evaluate(X_test, y_test_cat, verbose=1)


# In[23]:


print(score)

