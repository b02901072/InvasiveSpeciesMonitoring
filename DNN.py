import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad, SGD
from keras.regularizers import l1
import IO

IMAGE_ROW = 150
IMAGE_COLUMN = 200
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images, train_labels = IO.load_train()
test_images = IO.load_test()

train_images = np.reshape(train_images, (-1, IMAGE_SIZE*3))
test_images = np.reshape(test_images, (-1, IMAGE_SIZE*3))


input_layer = Input(shape=(IMAGE_SIZE*3,))
x = Dense(500, activation='relu')(input_layer)
x = Dropout(0.25)(x)
x = Dense(500, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(500, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input=input_layer, output=x)

optimizer = SGD(lr=1e-3, momentum=0.9)

print('Compiling model...')
model.compile(loss="binary_crossentropy", 
			  optimizer=optimizer,
			  metrics=["accuracy"])

print('Fitting model...')
model.fit(train_images,
		  train_labels,
		  batch_size=100, nb_epoch=50,    
		  shuffle=True,
		  validation_split=0.2)


model.save('model/DNN.model')

result = model.predict(test_images,
                       batch_size=100, verbose=1)

IO.write_result(result, 'result/DNN.csv')
