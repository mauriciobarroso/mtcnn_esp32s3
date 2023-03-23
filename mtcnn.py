import tensorflow as tf
from tensorflow import keras

# P-Net model
def create_pnet(width, height, weights_file):
    input = keras.layers.Input(shape=[width, height, 3])
    x = keras.layers.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = keras.layers.PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = keras.layers.PReLU(shared_axes=[1,2],name='PReLU2')(x)
    x = keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = keras.layers.PReLU(shared_axes=[1,2],name='PReLU3')(x)
    classifier = keras.layers.Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    bbox_regress = keras.layers.Conv2D(4, (1, 1), name='conv4-2')(x)
    model = keras.models.Model([input], [classifier, bbox_regress])
    model.load_weights(weights_file, by_name=True)
    return model

# R-Net model
def create_rnet(weights_file):
    input = keras.layers.Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = keras.layers.Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = keras.layers.PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = keras.layers.MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    x = keras.layers.Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = keras.layers.PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x = keras.layers.Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = keras.layers.PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = keras.layers.Permute((3, 2, 1))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, name='conv4')(x)
    x = keras.layers.PReLU( name='prelu4')(x)
    classifier = keras.layers.Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = keras.layers.Dense(4, name='conv5-2')(x)
    model = keras.models.Model([input], [classifier, bbox_regress])
    model.load_weights(weights_file, by_name=True)
    return model


# O-Net model
def create_onet(weights_file):
    input = keras.layers.Input(shape = [48,48,3])
    x = keras.layers.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = keras.layers.PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = keras.layers.PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = keras.layers.PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = keras.layers.MaxPool2D(pool_size=2)(x)
    x = keras.layers.Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = keras.layers.PReLU(shared_axes=[1,2],name='prelu4')(x)
    x = keras.layers.Permute((3,2,1))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, name='conv5') (x)
    x = keras.layers.PReLU(name='prelu5')(x)

    classifier = keras.layers.Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = keras.layers.Dense(4,name='conv6-2')(x)
    # landmark_regress = keras.layers.Dense(10,name='conv6-3')(x)
    model = keras.models.Model([input], [classifier, bbox_regress])#, landmark_regress])
    model.load_weights(weights_file, by_name=True)

    return model