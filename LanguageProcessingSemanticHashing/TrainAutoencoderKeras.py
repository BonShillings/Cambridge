from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

import numpy as np

def build_autoencoder_stack(layers,num_encoders,x_train):

    input_placeholder = Input(shape=(len(x_train[1]),))
    encoded = Dense(layers[0], activation='relu')(input_placeholder)

    for i in range(1,len(layers)):
        layer_size = layers[i]
        if i < num_encoders -1:
            encoded = Dense(layer_size, activation='relu')(encoded)

        elif i == num_encoders -1:
            encoded = Dense(layer_size, activation='sigmoid')(encoded)


        elif i == num_encoders:
            decoded = Dense(layer_size, activation='relu')(encoded)

        elif i > num_encoders:
            decoded = Dense(layer_size, activation='relu')(decoded)

        elif i == (len(layers) -1):
            decoded = Dense(layer_size, activation='sigmoid')(decoded)

    autoencoder = Model(input_placeholder, decoded)

    encoder = Model(input_placeholder, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder

def extract_encoder(layers,num_encoders,x_train):

    input_placeholder = Input(shape=(len(x_train[1]),))
    encoded = Dense(layers[0], activation='relu')(input_placeholder)

    for i in range(1,len(layers)):
        layer_size = layers[i]
        if i < num_encoders-1:
            encoded = Dense(layer_size, activation='relu')(encoded)
        elif i == num_encoders-1:
            encoded = Dense(layer_size, activation='sigmoid')(encoded)

    encoder = Model(input_placeholder, encoded)

    encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return encoder

def extract_encoder_from_file(filepath,layers,num_encoders,x_train):

    encoder = extract_encoder(layers,num_encoders,x_train)

    encoder.load_weights(filepath, by_name=True)

    return encoder

def example_autoencoder():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print x_train.shape
    print x_test.shape

    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='sigmoid')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

def run_mnist():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print x_train.shape
    print x_test.shape

    network_structure = [128,64,32,64,128,784]

    autoencoder = build_autoencoder_stack(layers=network_structure,num_encoders=3,x_train=x_train)

    validation_data=None
    if x_test is not None:
        validation_data = (x_test, x_test)

    autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

    print(autoencoder.summary())
    '''
    autoencoder.save_weights("kerasweights.txt",overwrite=True)

    encoder = extract_encoder_from_file(filepath="kerasweights.txt",layers=network_structure,num_encoders=3,x_train=x_train,)

    print(encoder.summary())

    predictions = encoder.predict(x_test)

    for prediction in predictions[1:10]:
        print(prediction)
    '''


