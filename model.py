import csv
import cv2
import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import math
import random
import tensorflow as tf
import matplotlib.image as mpimg

# Read input data
# Read data csv file
def read_data_csv(filename):
    columns = ('center', 'left', 'right', 'steering', 'throttle',
               'brake', 'speed')
    df = pd.read_csv(filename, skipinitialspace=True)
    df.columns = columns
    return df

# Change the image path as path of images in csv different from actual path
def change_image_path(df, folder):
    def new_filepath(s): return os.path.join(folder, s.split('/')[-1])
    df['center'] = df['center'].map(new_filepath)
    df['left'] = df['left'].map(new_filepath)
    df['right'] = df['right'].map(new_filepath)
    return df


# load image from file in HSV color space
def load_image(path):
    img = cv2.imread(path)
    print(path)
    return img


def get_input_shape(df):
    sample_input_image = df['center'].iloc[1]
    return load_image(sample_input_image).shape



def read_all_csvs_folders(parent_folder):
    data = []
    folders = os.listdir(parent_folder)
    print(folders)
    for folder in folders:
        if '.' in folder:
            print("Found")
            continue
        folder = os.path.join(parent_folder, folder)
        img_folder_path = os.path.join(folder, "IMG")
        csv_filepath = os.path.join(folder, "driving_log.csv")
        print(csv_filepath)
        df = read_data_csv(csv_filepath)
        df = change_image_path(df, img_folder_path)
        data.append(df)
    concatenate_df = pd.DataFrame(
        np.concatenate(data, axis=0), columns=df.columns)
    # concatenate_df = [concatenate_df['speed'] >= 0.1]
    return concatenate_df


flags = tf.app.flags
FLAGS = flags.FLAGS

# DEFINE FLAGS VARIABLES#
flags.DEFINE_integer('epo', 9, "The number of epochs.")
flags.DEFINE_integer('batch', 32, "batch size")

# Samples of the data.
samples = read_all_csvs_folders('data')
image_input_shape = get_input_shape(samples)

print("Total number of samples {}".format(len(samples)))
print ("Image input shape {}".format(image_input_shape))

## Divided Train and validation data using sci-kit split with factor or 0.1
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Function to resize images within the model
# as the NVidia model was designed with 66x200x3 images in mind
def resize(img):
    return ktf.image.resize_images(img, (66, 200))


'''
Added random brightness on HSV layer of the image so that it could 
learn any track with less brightness also.
'''
def random_brightness(img, amount):
    img_copy = img.copy().astype(np.int16)
    img_copy[:,:, 2] = img_copy[:,:, 2] + amount
    img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)
    return img_copy


def convert_image_rgbtohls(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    #hls_img[:,:,2] = 0.5 * hls_img[:,:,2]
    return hls_img

def convert_img_bgrtohsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def flip_img(img):
    return np.fliplr(img.copy())

'''
Generator that will generate the data in parallerl of defined batch size
'''
def generator(samples, batch_size=FLAGS.batch):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset: offset+batch_size]
            center_images = []
            center_measurments = []
            left_images = []
            left_measurments = []
            right_images = []
            right_measurments = []
            for i in range(len(batch_samples)):
                line = batch_samples.iloc[i]
                steering_center = float(line['steering'])
                ## Correction for left and right images so that car doesn't go off the track.
                correction = 0.3
                ## center image 
                amount = int(random.uniform(20, 50))
                #if (abs(steering_center) >= 0.05):
                source_path = line['center']
                img = cv2.imread(source_path)
                img = convert_img_bgrtohsv(img)
                center_images.append(img)
                center_measurments.append(steering_center)

                img = flip_img(img)
                new_steering = steering_center * -1.0
                center_images.append(img)
                center_measurments.append(new_steering)

                ## left image
                source_path = line['left']
                left_correction = steering_center + correction
                img = cv2.imread(source_path)
                img = convert_img_bgrtohsv(img)
                left_images.append(img)
                left_measurments.append(left_correction)

                img = flip_img(img)
                new_steering = left_correction * -1.0
                left_images.append(img)
                left_measurments.append(new_steering)

                ## Right image
                source_path = line['right']
                right_correction = steering_center - correction
                img = cv2.imread(source_path)
                img = convert_img_bgrtohsv(img)
                right_images.append(img)
                right_measurments.append(right_correction)

                img = flip_img(img)
                new_steering = right_correction * -1.0
                right_images.append(img)
                right_measurments.append(new_steering)

            concatenate_images = []
            concatenate_measurments = []
            concatenate_images.extend(center_images)
            concatenate_images.extend(left_images)
            concatenate_images.extend(right_images)
            concatenate_measurments.extend(center_measurments)
            concatenate_measurments.extend(left_measurments)
            concatenate_measurments.extend(right_measurments)

            X_samples = np.array(concatenate_images)
            y_samples = np.array(concatenate_measurments)
            yield sklearn.utils.shuffle(X_samples, y_samples)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

## Model is same as the model mentioned in the NVIDIA paper.
### Added drop out layers extra on FC to avoid overfitting.
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5,subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()
model.fit_generator(train_generator, samples_per_epoch= \
    (len(train_samples)/FLAGS.batch), validation_data= validation_generator, 
    validation_steps=(len(validation_samples)/FLAGS.batch), epochs=FLAGS.epo)
### Save the model 
model.save('model.h5')
