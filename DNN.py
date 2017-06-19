import sys
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adagrad
from keras.regularizers import l1


TEST_DATA_NUM = 10000
IMAGE_ROW = 128
IMAGE_COLUMN = 128
IMAGE_SIZE = IMAGE_ROW * IMAGE_COLUMN

train_images, train_labels = IO.load_train()
train_images = np.reshape(train_images, (-1, IMAGE_SIZE*3))

input_layer = Input(shape=(IMAGE_SIZE*3,))
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

'''
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
'''