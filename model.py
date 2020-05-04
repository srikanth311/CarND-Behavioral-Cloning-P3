# https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model

from data1 import read_csv_data, get_samples
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers import ELU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
import sys

train_data, valid_data = read_csv_data()
INIT='glorot_uniform'

model = Sequential()
model.add(Lambda (lambda X: X/255-0.5, input_shape=(32, 128, 3)))

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 128, 3) ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))

model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))

model.add(Dense(20, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error')
batch_size = 128

# https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model

model_fit_gen = model.fit_generator(
                                    get_samples(train_data, batch_size),
                                    steps_per_epoch=train_data.shape[0] // batch_size,
                                    epochs = 11,
                                    validation_data=get_samples(valid_data, batch_size),
                                    validation_steps=valid_data.shape[0] // batch_size
                                    )
model.save('model_base_set3_epochs_11.h5')
print("Model saved.")

sys.exit(0)




