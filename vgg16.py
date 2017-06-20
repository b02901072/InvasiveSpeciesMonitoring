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

validation_split = 0.2
validation_index = int(train_images.shape[0] * (1-validation_split))
valid_images = train_images[validation_index:]
valid_labels = train_labels[validation_index:]
train_images = train_images[:validation_index]
train_labels = train_labels[:validation_index]

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

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 20

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
	
train_datagen.fit(train_images)

print('Fitting model...')
model_name = 'model/vgg16_aug.model'
history = model.fit_generator(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=train_images.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(valid_images, valid_labels),
    callbacks=[ModelCheckpoint(model_name, monitor='val_acc', save_best_only=True)]
)

model = load_model(model_name)

result = model.predict(test_images,
                       batch_size=100, verbose=1)

IO.write_result(result, 'result/vgg16_aug.csv')
