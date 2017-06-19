import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils, generic_utils
import IO

TRAIN_DATA_NUM = 2295
TEST_DATA_NUM = 10000
IMAGE_ROW = 128
IMAGE_COLUMN = 128
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images, train_labels = IO.load_train()
#test_images = np.load('data/test.image.raw.npy')

#train_images = np.reshape(train_images, (TRAIN_DATA_NUM, 3, IMAGE_ROW, IMAGE_COLUMN))
#test_images = np.reshape(test_images, (TEST_DATA_NUM, 1, IMAGE_ROW, IMAGE_COLUMN))

model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(IMAGE_ROW, IMAGE_COLUMN, 3)))
#model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))

print('Compiling model...')
model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics=["accuracy"])

print('Fitting model...')
model.fit(train_images,
          train_labels,
          batch_size=32, nb_epoch=10,    
          shuffle=True)

'''
result = model.predict(test_images,
                       batch_size=100, verbose=1)

result_file = 'result.csv'
content = "id,label"

for i in range(TEST_DATA_NUM):
    best_class = np.argmax(result[i])
    content += "\n" + str(i) + "," + str(best_class)

with open(result_file, "w") as f:
    f.write(content)
    f.close()
'''
