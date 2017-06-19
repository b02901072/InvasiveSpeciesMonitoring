import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad
from keras.regularizers import l1

TRAIN_DATA_NUM = 60000
TEST_DATA_NUM = 10000
IMAGE_ROW = 28
IMAGE_COLUMN = 28
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images = np.load('data/train.image.raw.npy')
train_labels = np.load('data/train.label.npy')
test_images = np.load('data/test.image.raw.npy')

input_layer = Input(shape=(28*28,))
x = Dense(500, activation='relu')(input_layer)
#x = Dropout(0.25)(x)
x = Dense(500, activation='relu')(x)
#x = Dropout(0.25)(x)
#x = Dense(500, activation='relu')(x)
#x = Dropout(0.25)(x)
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.25)(x)
x = Dense(10, activation='softmax')(x)

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

result = model.predict(test_images,
					   batch_size=1000, verbose=1)

result_file = sys.argv[4]
content = "id,label"

for i in range(TEST_DATA_NUM):
	best_class = np.argmax(result[i])
	content += "\n" + str(i) + "," + str(best_class)

with open(result_file, "w") as f:
	f.write(content)
	f.close()
