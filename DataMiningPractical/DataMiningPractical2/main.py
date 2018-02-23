from keras.optimizers import SGD, Adadelta, Adam

from data import load_data
from model import get_model
from model_2 import get_model_2

batch_size = 128
nb_epoch = 100

# Load data
(X_train, y_train, X_test, y_test) = load_data()

# Load and compile model
model = get_model_2()

model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=1)

print("Accuracy:", score[1])
