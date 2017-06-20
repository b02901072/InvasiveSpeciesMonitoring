import sys
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils, generic_utils
import IO
from keras import applications

IMAGE_ROW = 150
IMAGE_COLUMN = 200
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images, train_labels = IO.load_train()
test_images = IO.load_test()

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_ROW, IMAGE_COLUMN, 3))

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dropout(0.5))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

optimizer = SGD(lr=1e-4, momentum=0.9)

print('Compiling model...')
model.compile(loss="binary_crossentropy", 
              optimizer=optimizer,
              metrics=["accuracy"])

print('Fitting model...')
model.fit(train_images,
          train_labels,
          batch_size=32, epochs=50,    
          shuffle=True,
          validation_split=0.2)

model.save('model/vgg16_CNN.model')

result = model.predict(test_images,
                       batch_size=100, verbose=1)

IO.write_result(result, 'result/vgg16_CNN.csv')
