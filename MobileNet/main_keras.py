# coding: utf-8

# # ECE 6141 Final Project
# ## Resnet
# ### Skin cancer Data
# 
# We classify skin cancer with using [Dataset: Skin Cancer MNIST:HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) using [ResNet (Microsoft)](https://arxiv.org/abs/1512.03385)
# 
# ##### Class
# akiec, bcc, bkl, df, mel, nv, vasc
# 
# 
# ##### Features
# 1. image
# 2. dx_type (histo, follow_up, consensus, and confocal)
# 3. localization (abdomen, back, chest, ear, face, foot, genital, hand, lower extremity, neck, scalp, trunk, upper extremity, and unknown)
# 4. sex
# 5. age
# 
# 
# ###### dx_type details
# - histo: Histopathology (microscopic examination of tissue)
# - follow_up: follow-up examination
# - consensus: expert consensus
# - confocal: in-vivo confocal microscopy

# In[1]:


import pandas as pd
import numpy as np
import keras
import scipy
import scipy.ndimage
import itertools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.core import Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
# from keras.applications.mobilenet import MobileNetV2
# from keras_applications import mobilenet_v2
# from utils import MobileNetv2, fine_tune
from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping
# from models import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

# ### Set network parameters

# In[2]:


batch_size = 32  # 5
num_class = 7
epochs = 100
K = 5
resize_width = 224
resize_height = 224
# root = '/Users/Kong/Documents/ECE6141_ResNet'


# ### Set input data

# In[3]:


meta_file_path = './input/HAM10000_metadata.csv'
print(meta_file_path)
meta_file = pd.read_csv(meta_file_path)
meta_file['noise'] = False
new_meta_file = pd.DataFrame(columns=meta_file.columns)

n_each_class = {}

n_each_class = {Class: meta_file[meta_file.dx == Class].count()["dx"] for Class in pd.unique(meta_file["dx"])}

# n_each_class = [meta_file[meta_file.dx==Class].count()["dx"] for Class in pd.unique(meta_file["dx"])]
# class_name = [Class for Class in pd.unique(meta_file["dx"])]
avg_n_class = np.ceil(np.mean(list(n_each_class.values()))).astype(int)

# For Balance
# Permutate and Copy with Gaussian Noise
for Class in pd.unique(meta_file["dx"]):

    if avg_n_class > n_each_class[Class]:
        # copy
        n_copy = avg_n_class - n_each_class[Class]

        new_meta_file = new_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class]), ignore_index=True)

        copy_meta_file = pd.DataFrame(columns=meta_file.columns)
        while meta_file[meta_file.dx == Class].shape[0] <= n_copy:
            copy_meta_file = copy_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class]), ignore_index=True)
            n_copy = n_copy - meta_file[meta_file.dx == Class].shape[0]
        copy_meta_file = copy_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class].sample(n=n_copy)),
                                               ignore_index=True)
        copy_meta_file['noise'] = True
        new_meta_file = new_meta_file.append(copy_meta_file, ignore_index=True)
    elif avg_n_class < n_each_class[Class]:
        # permutate
        pass
        new_meta_file = new_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class]), ignore_index=True)
        #new_meta_file = new_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class].sample(n=avg_n_class)),
        #                                     ignore_index=True)
    else:
        new_meta_file = new_meta_file.append(pd.DataFrame(meta_file[meta_file.dx == Class]), ignore_index=True)
new_meta_file = new_meta_file.sample(frac=1).reset_index(drop=True)

# In[4]:

n_sample = new_meta_file.shape[0]
X = np.zeros([n_sample, resize_height, resize_width, 3])
Y = np.zeros([n_sample])
Y_name = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
for idx, row in new_meta_file.iterrows():
    filename = row['image_id']
    try:
        x = img_to_array(load_img('./input/HAM10000_images_part_1/' + filename + '.jpg'))
    except FileNotFoundError:
        x = img_to_array(load_img('./input/HAM10000_images_part_2/' + filename + '.jpg'))
    # trim width to have same as height and resize
    x_width_start = int(((x.shape[1] - x.shape[0]) / 2))
    x_width_len = x.shape[0]
    x = x[:, x_width_start:x_width_start + x_width_len, :].astype('uint8')
    # resample
    x = scipy.misc.imresize(x, [resize_height, resize_width, 3], 'cubic')
    # gaussian noise
    if row['noise'] == True:
        mu, sigma = 0, 1
        x = x + np.random.normal(mu, sigma, x.shape)
    else:
        x = x.astype('float')
    X[idx, :, :, :] = x.reshape((1,) + x.shape)
    Y[idx] = Y_name[row['dx']]
print('Finished read files.')
Y = to_categorical(Y)


# ### Resnet Block Function

# In[5]:


def ResBlock(x, n_output, n_kernel, n_strides1, n_strides2, activation):
    y1 = Conv2D(n_output, kernel_size=n_kernel, strides=n_strides1, activation=activation, padding="same")(
        x)  # y = Conv2D(n_output, (n_kernel, n_kernel), padding='same')(x)
    y1_b = BatchNormalization()(y1)
    y2 = Conv2D(n_output, kernel_size=n_kernel, strides=n_strides2, activation=activation, padding="same")(
        y1_b)  # y = Conv2D(n_output, (n_kernel, n_kernel), padding='same')(x)
    y2_b = BatchNormalization()(y2)
    # this returns x + y.
    if x.shape != y2_b.shape:
        x = Conv2D(n_output, kernel_size=n_kernel, strides=n_strides1, activation=activation, padding="same")(x)
        x = BatchNormalization()(x)  # necessary?
    z = keras.layers.add([x, y2_b])
    return z


# ### CNN

# In[6]:


def CreateResNet():
    x = Input(shape=(resize_height, resize_width, 3))
    x_b = BatchNormalization()(x)

    h1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', padding="same")(x_b)
    h1_b = BatchNormalization()(h1)
    h1_mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h1_b)
    h1_mp_b = BatchNormalization()(h1_mp)

    h2a = ResBlock(h1_mp_b, 64, (3, 3), (1, 1), (1, 1), 'relu')
    h2b = ResBlock(h2a, 64, (3, 3), (1, 1), (1, 1), 'relu')

    h3a = ResBlock(h2b, 128, (3, 3), (2, 2), (1, 1), 'relu')
    h3b = ResBlock(h3a, 128, (3, 3), (1, 1), (1, 1), 'relu')

    h4a = ResBlock(h3b, 256, (3, 3), (2, 2), (1, 1), 'relu')
    h4b = ResBlock(h4a, 256, (3, 3), (1, 1), (1, 1), 'relu')

    h5a = ResBlock(h4b, 512, (3, 3), (2, 2), (1, 1), 'relu')
    h5b = ResBlock(h5a, 512, (3, 3), (1, 1), (1, 1), 'relu')

    ya = GlobalAveragePooling2D(data_format=None)(h5b)
    ya_b = BatchNormalization()(ya)
    yb = Dense(num_class, activation='softmax')(ya_b)

    model = Model(inputs=[x], outputs=[yb])
    sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])  # Loss function? , loss_weights=[1., 0.2] mean_squared_error
    return model


# regularizers?


# ### Cross Validation

# In[7]:


n_each_fold = np.ones([K]) * int(np.ceil(n_sample / K))
n_each_fold = n_each_fold.astype('int')
idx = 0
while np.sum(n_each_fold) - n_sample > 0:
    n_each_fold[idx] = n_each_fold[idx] - 1
    idx = idx + 1 % K
cumsum_n_each_fold = np.cumsum(n_each_fold)
idx_fold = np.insert(cumsum_n_each_fold, 0, 0).astype('int')

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=180,
    zca_whitening=False,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=180,
    zca_whitening=False,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.001)
# earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
# train_datagen = ImageDataGenerator(rescale=1./255)
# model.fit(x=None, y=None, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[reduce_lr], validation_split=0.1, shuffle=True)

y_pred = np.array([])
y_true = np.array([])
model_list = []
history_list = []
for idx, value in enumerate(n_each_fold):
    # model = mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1,
    #                  include_top=False,weights='imagenet',input_tensor=None, pooling=None,
    #                                 classes=7, backend=keras.backend,
    #                                 layers = keras.layers, models = keras.models, utils = keras.utils)
    base_model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1,
                             include_top=False, weights='imagenet', input_tensor=None, pooling=None,
                             classes=7)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(7, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    # model = MobileNetv2((224, 224, 3), 7)
    # model = fine_tune(num_classes = 7, weights = 'imagenet', model = model)
    opt = Adam()

    # model = MobileNet()
    # sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])  # Loss function? , loss_weights=[1., 0.2] mean_squared_error

    model_list.append(model)
    print("Fold: {0}".format(idx + 1))
    testX = X[idx_fold[idx]:idx_fold[idx + 1], :, :, :]
    testY = Y[idx_fold[idx]:idx_fold[idx + 1], :]

    trainX = np.delete(X, range(idx_fold[idx], idx_fold[idx + 1]), axis=0)
    trainY = np.delete(Y, range(idx_fold[idx], idx_fold[idx + 1]), axis=0)

    train_datagen.fit(trainX)
    test_datagen.fit(testX)

    train_generator = train_datagen.flow(trainX, trainY, batch_size=batch_size)
    validation_generator = test_datagen.flow(testX, testY, batch_size=batch_size)
    prediction_generator = test_datagen.flow(testX, y=testY, batch_size=batch_size, shuffle=False)

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  steps_per_epoch=len(trainX) // batch_size, epochs=epochs,
                                  validation_steps=len(testX) // batch_size,
                                  callbacks=[reduce_lr])
    history_list.append(history)

    Y_pred = model.predict_generator(prediction_generator, n_each_fold[idx] // batch_size + 1, verbose=1)
    y_pred = np.append(y_pred, np.argmax(Y_pred, axis=1)[0:n_each_fold[idx]])
    y_true = np.append(y_true, np.argmax(prediction_generator.y, axis=1))

# ### Result

# In[ ]:


print('Confusion Matrix')
conf_mat = confusion_matrix(y_true, y_pred)
print('Classification Report')
target_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
print(classification_report(y_true, y_pred, target_names=target_names))


# ### Plot

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=target_names,
                      title='Confusion matrix, without normalization')
plt.savefig('./result/conf_no_norm.png', bbox_inches='tight')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('./result/conf_norm.png', bbox_inches='tight')
plt.show()

# ### Loss, ACC Plot

# In[ ]:


for idx, history in enumerate(history_list):
    # print(history.history.keys())
    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy cv:{0}'.format(idx + 1))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./result/acc_cv{0}.png'.format(idx + 1), bbox_inches='tight')
    plt.show()
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss cv:{0}'.format(idx + 1))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./result/loss_cv{0}.png'.format(idx + 1), bbox_inches='tight')
    plt.show()

