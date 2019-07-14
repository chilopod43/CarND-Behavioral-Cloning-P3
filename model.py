import os
import csv
from random import shuffle

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Input,Lambda,Cropping2D,Conv2D,Flatten,Dense,Dropout
import sklearn
from sklearn.model_selection import train_test_split

def nvidia_model():
    model = Sequential()
    # normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    # cropping
    model.add(Cropping2D(cropping=((70,25),(0, 0))))
    # conv2d
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Dropout(0.5))
    # fullconnect
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def image_generator(image_dir, samples, sample_size=32):
    steer_offsets = [0.2, 0, -0.2]
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for sample_offset in range(0, num_samples, sample_size):
            batch_samples = samples[sample_offset:sample_offset+sample_size]

            images = list()
            measurements = list()
            for sample in batch_samples:
                for i,steer_offset in enumerate(steer_offsets): 
                    source_path = sample[i]
                    source_path = source_path.replace("\\", "/")
                    file_name = os.path.basename(source_path)
                    image_path = os.path.join(image_dir, "IMG", file_name)

                    # append images.
                    image = ndimage.imread(image_path)
                    images.append(image)
                    
                    # append measurements.
                    measurement = float(sample[3]) + steer_offset
                    measurements.append(measurement)
                    
                    # add flip data.
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurement_flipped = -measurement
                    measurements.append(measurement_flipped)
                    
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

def save_losses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig( 'loss.png' )

def train(cvs_path):
    # load samples.
    samples = list()
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)        
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # create a nvidia model.
    model = nvidia_model()
    model.compile(loss="mse", optimizer="adam")
    
    # train drives.
    sample_size=32 # batch size is 32*3*2. 3(left,center,right) x 2(flip) 
    image_dir = os.path.dirname(csv_path)
    train_generator = image_generator(image_dir, train_samples, sample_size=sample_size)
    validation_generator = image_generator(image_dir, validation_samples, sample_size=sample_size)

    history = model.fit_generator(
        train_generator, 
        steps_per_epoch=np.ceil(len(train_samples)/sample_size),
        validation_data=validation_generator, 
        validation_steps=np.ceil(len(validation_samples)/sample_size),
        epochs=5, verbose=1)
    
    # save results. 
    model.save("model.h5")
    save_losses(history)

if __name__=="__main__":
    import sys
    csv_path = sys.argv[1]
    
    train(csv_path)
