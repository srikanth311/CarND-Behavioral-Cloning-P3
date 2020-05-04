# https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model

from additional_models.data3 import read_csv_data, get_samples
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
import sys

train_base = "datasets/sri/srikanth/"
valid_base = "datasets/data_from_udacity_site/"


train_data = read_csv_data(train_base + "driving_log.csv")
valid_data = read_csv_data(valid_base + 'driving_log.csv')

train_images_path = train_base + 'IMG/'
valid_images_path =  valid_base + 'IMG/'

def run_model(run_number, num_epochs):

    model_save_file = "models/model_base_set3_epochs_{}_run_{}.h5".format(num_epochs, run_number)

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
                                        get_samples(train_data, train_images_path, batch_size),
                                        steps_per_epoch=train_data.shape[0] // batch_size,
                                        epochs = num_epochs,
                                        validation_data=get_samples(valid_data, valid_images_path, batch_size),
                                        validation_steps=valid_data.shape[0] // batch_size
                                        )
    model.save(model_save_file)
    print("Model saved {}".format(model_save_file))
    K.clear_session()

if __name__ == "__main__":
    number_of_epochs = 8
    for i in range(2):
        print("Running {}st/th time".format(i+1) )
        run_model(i+1, number_of_epochs)
    print("All runs completed.")
    sys.exit(0)




