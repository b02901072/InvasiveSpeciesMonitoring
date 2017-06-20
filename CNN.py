import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adagrad
from keras.utils import np_utils, generic_utils
import IO

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



model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(IMAGE_ROW, IMAGE_COLUMN, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))



print('Compiling model...')
model.compile(loss="binary_crossentropy", 
              optimizer="adam",
              metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
        rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)
	train_datagen.fit(train_images)

print('Fitting model...')
history = model.fit_generator(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=train_images.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(valid_images, valid_labels),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
)

model.save('model/CNN_aug.model')

result = model.predict(test_images,
                       batch_size=100, verbose=1)

IO.write_result(result, 'result/CNN_aug.csv')
