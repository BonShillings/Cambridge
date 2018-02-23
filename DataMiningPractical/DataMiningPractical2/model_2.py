from keras.models import Model
from keras.layers import Input, Dense, Flatten,Dropout,Conv2D,MaxPooling2D,BatchNormalization

def get_model_2():

    # Model parameters
    rows, cols = 28, 28
    input_shape = (rows, cols, 1)

    nb_classes = 10

    hidden_size = 128

    inp = Input(shape=input_shape)
    '''
    flat = Flatten()(inp)
    hidden_1 = Dense(hidden_size, activation='relu')(flat)
    hidden_2 = Dense(hidden_size / 2, activation='relu')(hidden_1)
    hidden_3 = Dense(hidden_size, activation='relu')(hidden_2)
    '''
    conv_1 = Conv2D(64, (3, 3), activation='relu',input_shape=input_shape)(inp)
    conv_2 = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(conv_1)
    conv_3 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    dropped = Dropout(0.2)(conv_3)

    batched = BatchNormalization()(dropped)

    conv_4 = Conv2D(128, (4,4), activation='relu', input_shape=input_shape)(batched)
    conv_5 = Conv2D(128, (4, 4), activation='relu', input_shape=input_shape)(conv_4)
    conv_6 = MaxPooling2D(pool_size=(2, 2))(conv_5)
    dropped_2 = Dropout(0.25)(conv_6)

    batched_2 = BatchNormalization()(dropped_2)

    '''
    conv_7 = Conv2D(256, (5, 5), activation='relu', input_shape=input_shape)(batched_2)
    conv_8 = Conv2D(256, (5, 5), activation='relu', input_shape=input_shape)(conv_7)
    conv_9 = MaxPooling2D(pool_size=(2, 2))(conv_8)
    dropped_3 = Dropout(0.25)(conv_9)

    batched_3 = BatchNormalization(batched_2)
    '''

    #flat_2 = Flatten()(batched_2)

    flat_2 = Flatten()(batched_2)
    hidden_4 = Dense(hidden_size, activation='relu')(flat_2)

    batched_4 = BatchNormalization()(hidden_4)
    hidden_5 = Dense(hidden_size, activation='relu')(batched_4)
    hidden_6 = Dense(hidden_size / 2, activation='relu')(hidden_5)

    #hidden_5 = Dense(hidden_size/2, activation='relu')(hidden_4)

    out = Dense(nb_classes, activation='softmax')(hidden_6)

    model = Model(inputs=inp, outputs=out)

    print(model.summary())

    return model


if __name__ == '__main__':

    model = get_model_2()
