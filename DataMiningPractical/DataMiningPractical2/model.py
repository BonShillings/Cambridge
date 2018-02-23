from keras.models import Model
from keras.layers import Input, Dense, Flatten,Dropout,Conv2D,MaxPooling2D

def get_model():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10

    hidden_size = 128

    inp = Input(shape=input_shape)
    conv_1 = Conv2D(64, (3, 3), activation='relu',input_shape=input_shape)(inp)
    conv_2 = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(conv_1)
    conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropped = Dropout(0.2)(conv_3)

    flat = Flatten()(dropped)
    hidden_1 = Dense(hidden_size, activation='relu')(flat)
    hidden_2 = Dense(hidden_size/2, activation='relu')(hidden_1)
    hidden_3 = Dense(hidden_size/4, activation='relu')(hidden_2)
    hidden_4 = Dropout(0.2)(hidden_3)
    hidden_5 = Dense(hidden_size / 8, activation='relu')(hidden_4)
    hidden_6 = Dropout(0.2)(hidden_5)
    hidden_7 = Dense(hidden_size / 4, activation='relu')(hidden_6)
    hidden_8 = Dense(hidden_size/2, activation='relu')(hidden_7)
    hidden_9 = Dense(hidden_size, activation='relu')(hidden_8)
    out = Dense(nb_classes, activation='softmax')(hidden_9)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model()
