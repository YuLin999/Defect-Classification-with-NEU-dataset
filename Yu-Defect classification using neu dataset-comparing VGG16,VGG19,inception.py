
# coding: utf-8

# In[1]:


import os
import glob

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#from scipy.misc import imresize
from skimage import io
from skimage.color import gray2rgb
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.set(style='white')

import keras.backend as K
from keras.models import Model
from keras.applications import vgg16, vgg19, inception_v3, resnet50

#from keras.applications.inception_v3 import InceptionV3, preprocess_input,decode_predictions

from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras import optimizers

neu_root = 'D:\Python\CV Project\defect'
#!cat {neu_root}/README.md


# In[2]:


# load training and test set filenames
def load_paths(filepath):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f]
            
train_files = load_paths(os.path.join(neu_root, 'train.txt'))
test_files = load_paths(os.path.join(neu_root, 'test.txt'))


# In[3]:


# load images and labels
def load_neu(image_files):
    """ load images into numpy array """

    images = [io.imread(os.path.join(neu_root, 'data', p)) for p in image_files]
    images = np.array(images)

    get_image_class = lambda path: path.split('_')[0]
    labels = list(map(get_image_class, image_files))
    return np.array(images), labels


train_images, train_labels = load_neu(train_files)
test_images, test_labels = load_neu(test_files)


label_dict = {label: idx
              for idx, label in enumerate(sorted(set(train_labels)))}
y_train = np.array([label_dict[label] for label in train_labels])
y_test = np.array([label_dict[label] for label in test_labels])


# In[4]:


#model = vgg16.VGG16(weights=None)
#model = vgg19.VGG19(weights=None)
#model = inception_v3.InceptionV3(weights=None)
#model = resnet50.ResNet50(weights=None)
#print("model structure: ", model.summary())
#print("model weights: ", model.get_weights())


# In[5]:


def preprocess_imagenet(images):
    """ expect a 3D array containing grayscale uint8 images """
    
    # resize images with scipy
    I = np.array([imresize(image, (224,224)) for image in images])
    
    # convert to RGB and apply imagenet preprocessing
    I = gray2rgb(I).astype(np.float32)

    return vgg16.preprocess_input(I)


def vgg16_layer(output_name='fc1'):
    # Note: currently hippolyta compute nodes cannot access user home directories,
    # and do not have direct internet access.
    # use full paths to NFS filesystem endpoints (not symlinks)
    # keras pretrained model weight files are here:
    KERAS_ROOT = 'D:\Python\CV Project\defect'
    weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = os.path.join(KERAS_ROOT, 'models', weights_file)

    # initialize network with random weights, and load from hdf5
    cnn = vgg16.VGG16(include_top=True, weights='Imagenet')
    cnn.load_weights(filepath=weights_path)

    model = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer(output_name).output
    )

    return model


def cached_forward_pass_1(images, phase='train', layername='fc1', datadir='../data'):
        
    datafile = os.path.join(datadir, 'neu-vgg16-{}-{}.npy'.format(layername, phase))
    try:
        features = np.load(datafile)
    except FileNotFoundError:
        print('forward pass for NEU {}ing images'.format(phase))
        model = vgg_layer(layername)
        features = model.predict(preprocess_imagenet(images), verbose=True)
        np.save(datafile, features)

    return features


# In[6]:


datadir='D:\Python\CV Project\defect\data'

fc1_train = cached_forward_pass_1(train_images, phase='train', layername='fc1', datadir=datadir)
fc1_test = cached_forward_pass_1(test_images, phase='test', layername='fc1', datadir=datadir)

fc2_train = cached_forward_pass_1(train_images, phase='train', layername='fc2', datadir=datadir)
fc2_test = cached_forward_pass_1(test_images, phase='test', layername='fc2', datadir=datadir)


# In[7]:


def linear_softmax_classifier(n_classes=6, input_dim=2048):
    
    input_layer = Input(shape=(input_dim,))
    output_layer = Dense(n_classes, input_shape=(input_dim,), activation='softmax')(input_layer)

    return Model(inputs=input_layer, outputs=output_layer)


# In[8]:


lm_fc2 = linear_softmax_classifier(input_dim=4096)
opt = optimizers.Adam(lr=0.001)
lm_fc2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

target = np_utils.to_categorical(y_train)

# shuffle training data to get shuffled validation data
ntrain = train_images.shape[0]
initial_shuf = np.random.permutation(ntrain)
X = fc2_train[initial_shuf]
y = target[initial_shuf]

hist = lm_fc2.fit(X, y, validation_split=0.2, epochs=20)


# In[9]:


metric = 'acc' # 'loss' | 'acc'
plt.plot(hist.epoch, hist.history[metric], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric)
plt.legend()


# In[10]:


def evaluate_confusion_matrix(model, testdata, y_test):

    p = model.predict(testdata)
    p = np.squeeze(p) # remove spatial dimensions... (they're 1 anyways for these inputs)
    pred = np.argmax(p, axis=-1)

    conf = confusion_matrix(y_test, pred)
    sns.heatmap(
        conf, annot=True, square=True, 
        mask=(conf == 0), linewidths=.5, linecolor='k',
        xticklabels=sorted(set(train_labels)),
        yticklabels=sorted(set(train_labels))
    )


# In[11]:


evaluate_confusion_matrix(lm_fc2, fc2_test, y_test)


# In[12]:


def preprocess_imagenet(images):
    """ expect a 3D array containing grayscale uint8 images """
    
    # resize images with scipy
    I = np.array([imresize(image, (224,224)) for image in images])
    
    # convert to RGB and apply imagenet preprocessing
    I = gray2rgb(I).astype(np.float32)

    return vgg19.preprocess_input(I)


def vgg_layer(output_name='fc1'):
    # Note: currently hippolyta compute nodes cannot access user home directories,
    # and do not have direct internet access.
    # use full paths to NFS filesystem endpoints (not symlinks)
    # keras pretrained model weight files are here:
    #KERAS_ROOT = 'D:\Python\CV Project\defect'
    #weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #weights_path = os.path.join(KERAS_ROOT, 'models', weights_file)

    # initialize network with random weights, and load from hdf5
    cnn = vgg19.VGG19(include_top=True, weights='imagenet')
   # cnn.load_weights(filepath=weights_path)

    model = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer(output_name).output
    )

    return model


def cached_forward_pass(images, phase='train', layername='fc1', datadir='../data'):
        
    datafile = os.path.join(datadir, 'neu-vgg19-{}-{}.npy'.format(layername, phase))
    try:
        features = np.load(datafile)
    except FileNotFoundError:
        print('forward pass for NEU {}ing images'.format(phase))
        model = vgg_layer(layername)
        features = model.predict(preprocess_imagenet(images), verbose=True)
        np.save(datafile, features)

    return features


# In[13]:


datadir='D:\Python\CV Project\defect\data'

v19fc1_train = cached_forward_pass(train_images, phase='train', layername='fc1', datadir=datadir)
v19fc1_test = cached_forward_pass(test_images, phase='test', layername='fc1', datadir=datadir)

v19fc2_train = cached_forward_pass(train_images, phase='train', layername='fc2', datadir=datadir)
v19fc2_test = cached_forward_pass(test_images, phase='test', layername='fc2', datadir=datadir)


# In[14]:


lm_v19fc2 = linear_softmax_classifier(input_dim=4096)
opt = optimizers.Adam(lr=0.001)
lm_v19fc2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

target = np_utils.to_categorical(y_train)

# shuffle training data to get shuffled validation data
ntrain = train_images.shape[0]
initial_shuf = np.random.permutation(ntrain)
X = v19fc2_train[initial_shuf]
y = target[initial_shuf]

hist = lm_v19fc2.fit(X, y, validation_split=0.2, epochs=20)


# In[15]:


metric = 'acc' # 'loss' | 'acc'
plt.plot(hist.epoch, hist.history[metric], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric)
plt.legend()


# In[16]:


evaluate_confusion_matrix(lm_v19fc2, v19fc2_test, y_test)


# In[17]:


def preprocess_imagenet(images):
    """ expect a 3D array containing grayscale uint8 images """

    # resize images with scipy
    I = np.array([resize(image, (224,224)) for image in images])

    # convert to RGB and apply imagenet preprocessing
    I = gray2rgb(I).astype(np.float32)

    return inception_v3.preprocess_input(I)

def inception_layer(output_name='avg_pool'):
    # Note: currently hippolyta compute nodes cannot access user home directories,
    # and do not have direct internet access.
    # use full paths to NFS filesystem endpoints (not symlinks)
    # keras pretrained model weight files are here:
    #KERAS_ROOT = 'D:\Python\CV Project\defect'
    #weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #weights_path = os.path.join(KERAS_ROOT, 'models', weights_file)

    # initialize network with random weights, and load from hdf5
    cnn = inception_v3.InceptionV3(include_top=True, weights='imagenet', classes=1000)
    #cnn.load_weights(filepath=weights_path)

    model = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer(output_name).output
    )

    return model


def cached_forward_pass(images, phase='train', layername='avg_pool', datadir='../data'):
        
    datafile = os.path.join(datadir, 'neu-inception-{}-{}.npy'.format(layername, phase))
    try:
        features = np.load(datafile)
    except FileNotFoundError:
        print('forward pass for NEU {}ing images'.format(phase))
        model = inception_layer(layername)
        features = model.predict(preprocess_imagenet(images), verbose=True)
        np.save(datafile, features)

    return features


# In[18]:


datadir='D:\Python\CV Project\defect\data'

ap_train = cached_forward_pass(train_images, phase='train', layername='avg_pool', datadir=datadir)
ap_test = cached_forward_pass(test_images, phase='test', layername='avg_pool', datadir=datadir)


# In[19]:


ntrain = train_images.shape[0]

lm_ap = linear_softmax_classifier(input_dim=2048)
opt = optimizers.Adam(lr=0.03)
lm_ap.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

target = np_utils.to_categorical(y_train)

# shuffle training data to get shuffled validation data
initial_shuf = np.random.permutation(ntrain)
X = ap_train[initial_shuf]
y = target[initial_shuf]

hist = lm_ap.fit(X, y, validation_split=0.2, epochs=20)


# In[20]:


metric = 'acc' # 'loss' | 'acc'
plt.plot(hist.epoch, hist.history[metric], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric)
plt.legend()


# In[21]:


evaluate_confusion_matrix(lm_ap, ap_test, y_test)


# In[22]:


def preprocess_imagenet(images):
    """ expect a 3D array containing grayscale uint8 images """

    # resize images with scipy
    I = np.array([resize(image, (224,224)) for image in images])

    # convert to RGB and apply imagenet preprocessing
    I = gray2rgb(I).astype(np.float32)

    return resnet50.preprocess_input(I)

def resnet_layer(output_name='avg_pool'):
    # Note: currently hippolyta compute nodes cannot access user home directories,
    # and do not have direct internet access.
    # use full paths to NFS filesystem endpoints (not symlinks)
    # keras pretrained model weight files are here:
    #KERAS_ROOT = 'D:\Python\CV Project\defect'
    #weights_file = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    #weights_path = os.path.join(KERAS_ROOT, 'models', weights_file)

    # initialize network with random weights, and load from hdf5
    cnn = resnet50.ResNet50(include_top=True, weights='imagenet', classes=1000)
    #cnn.load_weights(filepath=weights_path)

    model = Model(
        inputs=cnn.input,
        outputs=cnn.get_layer(output_name).output
    )

    return model


def cached_forward_pass_resnet(images, phase='train', layername='avg_pool', datadir='../data'):
        
    datafile = os.path.join(datadir, 'neu-resnet-{}-{}.npy'.format(layername, phase))
    try:
        features = np.load(datafile)
    except FileNotFoundError:
        print('forward pass for NEU {}ing images'.format(phase))
        model = resnet_layer(layername)
        features = model.predict(preprocess_imagenet(images), verbose=True)
        np.save(datafile, features)

    return features


# In[23]:


datadir='D:\Python\CV Project\defect\data'

rnap_train = cached_forward_pass_resnet(train_images, phase='train', layername='avg_pool', datadir=datadir)
rnap_test = cached_forward_pass_resnet(test_images, phase='test', layername='avg_pool', datadir=datadir)


# In[24]:


ntrain = train_images.shape[0]

lm_rnap = linear_softmax_classifier(input_dim=2048)
opt = optimizers.Adam(lr=0.04)
lm_rnap.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

target = np_utils.to_categorical(y_train)

# shuffle training data to get shuffled validation data
initial_shuf = np.random.permutation(ntrain)
X = rnap_train[initial_shuf]
y = target[initial_shuf]

hist = lm_rnap.fit(X, y, validation_split=0.2, epochs=20)


# In[25]:


metric = 'acc' # 'loss' | 'acc'
plt.plot(hist.epoch, hist.history[metric], label='train')
plt.plot(hist.epoch, hist.history['val_{}'.format(metric)], label='val')
plt.xlabel('epoch')
plt.ylabel(metric)
plt.legend()


# In[26]:


evaluate_confusion_matrix(lm_rnap, rnap_test, y_test)

