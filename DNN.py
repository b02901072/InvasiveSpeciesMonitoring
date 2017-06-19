import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad
from keras.regularizers import l1


IMAGE_ROW = 128
IMAGE_COLUMN = 128
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images, train_labels = IO.load_train()
test_images = IO.load_test()

train_images = np.reshape(train_images, (-1, IMAGE_SIZE*3))
test_images = np.reshape(test_images, (-1, IMAGE_SIZE*3))


input_layer = Input(shape=(IMAGE_SIZE*3,))
x = Dense(500, activation='relu')(input_layer)
#x = Dropout(0.25)(x)
x = Dense(500, activation='relu')(x)
#x = Dropout(0.25)(x)
#x = Dense(500, activation='relu')(x)
#x = Dropout(0.25)(x)
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.25)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(input=input_layer, output=x)

print('Compiling model...')
model.compile(loss="categorical_crossentropy", 
			  optimizer="adam",
			  metrics=["accuracy"])

print('Fitting model...')
model.fit(train_images,
		  train_labels,
		  batch_size=100, nb_epoch=50,    
		  shuffle=True,
		  validation_split=0.1)


model.save('model/CNN.model')

result = model.predict(test_images,
                       batch_size=100, verbose=1)

IO.write_result(result, 'result/DNN.csv')
