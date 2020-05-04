import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
import cv2
import pathlib
import os
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_csv_data(csv_file_location):

    CENTER, LEFT, RIGHT, STEERING, THROTTLE, BRAKE, SPEED = range(7)
    #driving_log = pd.io.parsers.read_csv('datasets/combined-new-data/driving_log_new.csv').to_numpy()
    driving_log = pd.io.parsers.read_csv(csv_file_location).to_numpy()
    count = len(driving_log)

    left = np.random.choice(count, count//2, replace=False)
    right = np.random.choice(count, count//2, replace=False)
    center_data = driving_log[:, [CENTER, STEERING, THROTTLE, BRAKE, SPEED]]

    #
    # Add 0.25 to the left camera angle. # Main idea being left camera has to move right to get to the center.
    left_data = driving_log[:, [LEFT, STEERING, THROTTLE, BRAKE, SPEED]] [left, :]
    #left_data[:, 1] += 0.25

    # Subtracct 0.25 to the right camera steering angle. subtract 0.25 to move left to get to the center.
    right_data = driving_log[:, [RIGHT, STEERING, THROTTLE, BRAKE, SPEED]] [right, :]
    #right_data[:, 1] -= 0.25

    left_data[:, [STEERING]] = left_data[:, [STEERING]] + 0.25
    right_data[:, [STEERING]] = right_data[:, [STEERING]] - 0.25

    data = np.concatenate( (center_data, left_data) )
    data = np.concatenate( (data, right_data) )
    np.random.shuffle(data)
    #train_data, valid_data = model_selection.train_test_split(data, test_size=.2)

    #print(len(left_data))
    #print(len(train))
    #print(len(valid))

    #print(len(left_data))

    return data

def process_input_image(image, file_name_part=""):
    try:
        x = image.shape[2]
        if x == 3:
            #image = image[60:140,:,:]
            image = cv2.resize(image, (128, 32))
    except:
        print("Exception occurred :: {}".format(file_name_part))
        return None
    return image

def crop_image(image):
    return image[60:140,:,:]

def trans_image(image, steer, trans_range=50, file_name_part=""):
    """
    translate image and compensate for the translation on the steering angle
    """

    try:
        x = image.shape[2]
        if x == 3:
            rows, cols, chan = image.shape

            # horizontal translation with 0.008 steering compensation per pixel
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            steer_ang = steer + tr_x / trans_range * .4

            # option to disable vertical translation (vertical translation not necessary)

            tr_y = 0

            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

            return image_tr, steer_ang
    except:
        print("Exception occurred :: {}".format(file_name_part))
        return None, None

def apply_brightness(image):
    """
    apply random brightness on the image
    """
    # Input image from the training data is in BGR format
    # First we convert this into HSV
    # drive.py takes input in RGB format - so converting from HSV to RGB before returning.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()

    # scaling up or down the V channel of HSV
    image[:, :, 2] = image[:, :, 2] * random_bright
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def crop_and_bright_resize_for_autonomous_driving(image):
    crp_img = crop_image(image)
    bri_img = apply_brightness(crp_img)
    final_image = process_input_image(bri_img, "")
    return final_image

def get_batch_data(batch, image_path):
    IMG, STEERING, THROTTLE, BRAKE, SPEED = range(5)
    x, y = [],[]
    trans_range = 50
    for row in batch:
        file_name_part = pathlib.PurePath(row[IMG]).name
        full_file_path = pathlib.Path(image_path + file_name_part)
        if full_file_path.exists():
            if full_file_path.is_file():
                orig_image = plt.imread(image_path + file_name_part)
                steering_angle = row[STEERING]
                transed_image, steer_ang = trans_image(orig_image, steering_angle, trans_range, file_name_part)
                if transed_image is not None:

                    cropped_image = crop_image(transed_image)
                    bright_image = apply_brightness(cropped_image)
                    final_image = process_input_image(bright_image, file_name_part)

                    if final_image is not None:
                        angle = steer_ang
                        x.append( final_image )
                        y.append( angle )
                        # An effective technique for helping with the left turn bias invovles flipping images and taking the oppsite sign
                        # of the steering measurement.
                        # image_flipped = np.fliplr(image)
                        # measurementr_flipped = -measurement
                        # TODO: details
                        # Add a flipped image for every image.
                        x.append( final_image[:, ::-1, :] )
                        # TODO:
                        y.append( -1 * angle )
                else:
                    print("Corrupted image - {}".format(full_file_path))
            else:
                print("Corrupted image - {}".format(full_file_path))

    return np.array(x), np.array(y)

def get_samples(data, image_path, batch_size=128):
    # From stackoverflow
    # in Keras, generators in fit_generator need to be infinitely iterable. The idea being that the function that created the generator
    # needs to be responsible for cycling through your data as many times as needed
    # for cycling through your data as many times as needed
    # If you do not have "while True: - you will get "Stop Iteration" message and the program will be stopped.
    while True:
        for index in range(0, len(data), batch_size):  #batch_size is step size in range function
            next_batch = data[index:index+batch_size]
            X, Y = get_batch_data(next_batch, image_path)
            yield X, Y

#if __name__ == '__main__':
##    train, valid = read_csv_data()
#    get_samples(train, valid)