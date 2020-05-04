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

def read_csv_data():
    CENTER, LEFT, RIGHT, STEERING, THROTTLE, BRAKE, SPEED = range(7)
    #driving_log = pd.io.parsers.read_csv('datasets/combined-new-data/driving_log_new.csv').to_numpy()
    driving_log = pd.io.parsers.read_csv('datasets/srikanth/driving_log.csv').to_numpy()
    count = len(driving_log)

    left = np.random.choice(count, count//2, replace=False)
    right = np.random.choice(count, count//2, replace=False)
    center_data = driving_log[:, [CENTER, STEERING, THROTTLE, BRAKE, SPEED]]

    #
    # Add 0.25 to the left camera angle. # Main idea being left camera has to move right to get to the center.
    left_data = driving_log[:, [LEFT, STEERING, THROTTLE, BRAKE, SPEED]] [left, :]
    left_data[:, 1] += 0.25

    # Subtracct 0.25 to the right camera steering angle. subtract 0.25 to move left to get to the center.
    right_data = driving_log[:, [RIGHT, STEERING, THROTTLE, BRAKE, SPEED]] [right, :]
    right_data[:, 1] -= 0.25

    data = np.concatenate( (center_data, left_data) )
    data = np.concatenate( (data, right_data) )
    np.random.shuffle(data)
    train_data, valid_data = model_selection.train_test_split(data, test_size=.2)

    print(len(left_data))
    #print(len(train))
    #print(len(valid))

    #print(len(left_data))

    return train_data, valid_data

def crop_and_resize_for_autonomous_driving(image):
    img = crop_image(image, 20, 140, 50, 270)
    img = cv2.resize(img, (200, 66))
    return img

def process_input_image(image, file_name_part=""):
    # TODO: How to find the orig image size - does it mention somewhere
    # TODO: resize to 128x32
    #print("File name part is :: {} ".format(file_name_part))
    #image = image[70:140, :, :]
    #image = cv2.resize(image, (128, 32))
    try:
        x = image.shape[2]
        if x == 3:
            #image = image[70:140,:,:] - cropping is part of other function.
            #image = cv2.resize(image, (128, 32))
            image = cv2.resize(image, (200, 66))
    except:
        print("Exception occurred :: {}".format(file_name_part))
        return None
    return image

def crop_image(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]


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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()

    # scaling up or down the V channel of HSV
    image[:, :, 2] = image[:, :, 2] * random_bright
    return image

def get_batch_data(batch):
    IMG, STEERING, THROTTLE, BRAKE, SPEED = range(5)
    x, y = [],[]
    trans_range = 50
    for row in batch:
        #print("Row is ::: {}".format(row[IMG]))
        #file_name_part = split(row[IMG], '/', 5)[1:2]
        file_name_part = pathlib.PurePath(row[IMG]).name
        #print("SSSS {}".format(file_name_part))
        #full_file_path = pathlib.Path("datasets/combined-forward-reverse/combined_old_new/images/" + file_name_part[0])
        #full_file_path = pathlib.Path("datasets/combined-new-data/images/" + file_name_part)
        full_file_path = pathlib.Path("datasets/srikanth/IMG/" + file_name_part)
        #print("File full part is :: {}".format(full_file_path))
        if full_file_path.exists():
            #print("File full part is :: {}".format(full_file_path))
            if full_file_path.is_file():
                orig_image = plt.imread("datasets/srikanth/IMG/" + file_name_part)
                steering_angle = row[STEERING]
                transed_image, steer_ang = trans_image(orig_image, steering_angle, trans_range, file_name_part)
                if transed_image is not None:

                    cropped_image = crop_image(transed_image, 20, 140, 0+trans_range, transed_image.shape[1]-trans_range)
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
                        print("Image is none ***** ")
                else:
                    print("Corrupted image - {}".format(full_file_path))

    return np.array(x), np.array(y)

def get_samples(data, batch_size=128):
    # From stackoverflow
    # in Keras, generators in fit_generator need to be infinitely iterable. The idea being that the function that created the generator
    # needs to be responsible for cycling through your data as many times as needed
    # for cycling through your data as many times as needed
    # If you do not have "while True: - you will get "Stop Iteration" message and the program will be stopped.
    while True:
        for index in range(0, len(data), batch_size):  #batch_size is step size in range function
            next_batch = data[index:index+batch_size]
            X, Y = get_batch_data(next_batch)
            yield X, Y

# Found on stack overflow
def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


#if __name__ == '__main__':
##    train, valid = read_csv_data()
#    get_samples(train, valid)