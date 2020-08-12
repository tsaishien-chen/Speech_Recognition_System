from util import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Feature dimension
mfcc_max_len = 32
n_mfcc       = 13
mfcc_shape   = (n_mfcc*3, mfcc_max_len)
channel      = 1
epochs       = 50
batch_size   = 64
verbose      = 1
num_classes  = 6

# Save data to array file first
save_data_to_array(max_len=mfcc_max_len, n_mfcc=n_mfcc)

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test(split_ratio=0.8)


# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], mfcc_shape[0], mfcc_shape[1], channel)
X_test = X_test.reshape(X_test.shape[0], mfcc_shape[0], mfcc_shape[1], channel)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)


model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(mfcc_shape[0], mfcc_shape[1], channel)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.35),
              metrics=['accuracy'])
log = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

model.save('model/model.h5')

# list all data in history
print(log.history.keys())
# summarize history for accuracy
plt.plot(log.history['acc'])
plt.plot(log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
