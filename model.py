from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D as MaxPool
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import os
import math
import random 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('training_file', '', 'csv file')
flags.DEFINE_string('output', '', 'output model path')
flags.DEFINE_boolean('update', False, 'update weights instead of train from scratch')

def preprocess(X):
    import tensorflow as tf
    # X2 = tf.image.resize_images(X, size= (32, 32))

    #mean, var = tf.nn.moments(X, [1], keep_dims=True)
    #nX = (X - mean)/tf.sqrt(var)
    nX = (X/255.) - 0.5

    nX = tf.image.resize_images(nX, size= (64, 64))
    #nX = tf.image.rgb_to_grayscale(nX)

    return nX

def lenet(image_shape):
    print("Using Lenet")

    model = Sequential()
    model.add(Cropping2D(cropping=((35,25), (0,0)), input_shape = image_shape ))
    #model.add(Lambda(preprocess, input_shape=image_shape))
    model.add(Lambda(preprocess))

    model.add(Conv2D(32, 5,5, activation='relu'))
    model.add(MaxPool())

    model.add(Conv2D(64, 5,5, activation='relu'))
    model.add(MaxPool())

    model.add(Flatten())

    #model.add(Dense(120, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(.5) )
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.5) )
    model.add(Dense(1))

    model.compile('adam', 'mse')

    return model

def my_model(image_shape):
    print("Using noobie1")
    #image_shape = X.shape[1:]

    model = Sequential()
    model.add(Lambda(preprocess, input_shape=image_shape))
    model.add(Flatten())
    #model.add(Flatten(input_shape = image_shape))
    model.add(Dense(1))
    
    model.compile('adam', 'mse')
    return model

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    n_samples = df.shape[0]
    image_paths = df.center.values
    steering_angles = df.steering.values

    l_image_paths = df.left.values
    r_image_paths = df.right.values

    #generate lr samples
    # import pdb
    # pdb.set_trace()
    non_zeros = steering_angles
    #extra = int(0.85 * non_zeros.shape[0])
    #extra_translation = int(0.15 * non_zeros.shape[0])
    
    #print("Generating extra ", extra * 2, "samples")
    #print("Generating extra ", extra_translation, "samples for translation")

    #l_indices = np.random.choice(non_zeros.shape[0], extra, replace = False)
    #r_indices = np.random.choice(non_zeros.shape[0], extra, replace = False)

    correction = 0.20
    l_steering = non_zeros + correction
    r_steering = non_zeros - correction
    l_images = l_image_paths
    r_images = r_image_paths

    #weights = np.ones(n_samples)
    #weights[(steering_angles >= 0.3) | (steering_angles <= -0.3)] = 10.

    # l_steering_angles = steering_angles + 0.27#1.24
    # r_steering_angles = steering_angles - 0.27#1.24

    Y = np.concatenate((steering_angles, l_steering, r_steering))
    X = np.concatenate((image_paths, l_images, r_images))

    weights = np.ones(Y.shape)
    weights[Y == 0] = 0.9#0.3

    # weights = np.ones(n_samples * 3)
    # weights[(steering_angles >= 0.3) | (steering_angles <= -0.3)] = 10.

    """ sanity check """
    print("Before: ", n_samples, " After augmentation: ", X.shape[0])
    #pd.Series(Y).hist(bins = 30, weights=weights)
    #plt.show()

    return X, Y, weights, Y.shape[0]
    #return image_paths, steering_angles, weights, n_samples * 3

def jitter(img, steer):
    s = img.shape
    noise = np.random.randint(0, 50, (s[0], s[1]))
    jitter = np.zeros_like(img)
    jitter[:,:,1] = noise

    jitter_img = cv2.add(img, jitter)

    return jitter_img, steer

def translate(img, steer):
    shift = 0
    while shift  == 0:
        shift = random.randint(-25, 25)
    
    if shift <= 0:
        aug = np.pad(img, ((0,0), (abs(shift),0), (0,0)), 'constant', constant_values=0)[:, :shift, :]
    else: 
        aug = np.pad(img, ((0,0), (0,shift), (0,0)), 'constant', constant_values=0)[:, shift:, :]

    correct_steer = steer #- shift * 0.004
    return aug, correct_steer
        
def translate2(img, steer):
    shift = 0
    while shift  == 0:
        shift = random.randint(-35, 35)
    
    if shift <= 0:
        aug = np.pad(img, ( (abs(shift),0), (0,0), (0,0)), 'constant', constant_values=0)[:shift, :, :]
    else: 
        aug = np.pad(img, ((0,shift), (0,0), (0,0)), 'constant', constant_values=0)[ shift:, :, :]

    return aug, steer

def generate(X, Y, W, train=True, batch_size=50):
    print("Size: ", Y.shape[0])
    multiplier = .7
    
    imgs = []
    steers= []

    while 1:
        X, Y, W = shuffle(X, Y, W)
        for i in range(0, X.shape[0]):
			
            f = X[i]
            steer = Y[i]
            rand = random.random()
            if rand >= 0.20 and  abs(steer) <= 0.20:
                continue
                #idx = random.choice(np.where(np.abs(Y) > 0.1) [0])
            #if rand >= 0.3 and (steer in [-0.25, 0., .25] or abs(steer) < 0.1):
            #    idx = random.choice(np.where((Y != 0.25) & (Y != 0) & (Y != 0.25) | (np.abs(Y) < 0.1)) [0])
                #f = X[idx]
                #steer = Y[idx]

            f = f.strip()
            path = os.path.join('data', f)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if random.choice([0,1]) == 0:
                img = cv2.flip(img, 1)
                steer *= -1
            
            if random.choice([0,1]) == 0:
                img, steer = translate(img, steer)
            #elif random.choice([0,1]) == 0:
            #    img, steer = translate2(img, steer)
            elif random.choice([0,1]) == 0:
                img, steer = jitter(img, steer)

            imgs.append(img)
            steers.append(steer)
            
            if (len(imgs))%50 == 0:
                yield np.array(imgs), np.array(steers)
                steers = []
                imgs = []

def main(_):
    #step 1 load data.
    csv_file = FLAGS.training_file
    update = FLAGS.update

    image_paths, steer, weights, n_records = load_data(csv_file)
    image_paths, steer, weights = shuffle(image_paths, steer, weights)
    X_train, X_valid, Y_train, Y_valid, W_train, W_valid = train_test_split(image_paths, steer, weights, test_size=0.1)

    train_gen = generate(X_train, Y_train, W_train)
    validation_gen = generate(X_valid, Y_valid, W_valid, train=False)

    #step 2 get architecture
    #import pdb
    #pdb.set_trace()
    f = image_paths[0].replace(" ", '')
    image_shape = cv2.imread(os.path.join('data', f)).shape
    #network = my_model(image_shape)    
    network = lenet(image_shape)

    #step 2 train model
    num_batches = math.ceil( (X_train.shape[0] * 2) / 64)

    history_obj = network.fit_generator(train_gen,
        #samples_per_epoch= int(X_train.shape[0] * 2.7),
        samples_per_epoch= 25000,
        nb_epoch=3, 
        validation_data=validation_gen,
        nb_val_samples=3000,
        verbose=1)

    #step 3 output model!
    network.save(FLAGS.output)


if __name__ == '__main__':
    tf.app.run()
