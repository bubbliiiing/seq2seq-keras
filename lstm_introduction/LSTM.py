import numpy as np
from keras.layers import Input,Activation,Dense
from keras.models import Model
from keras.datasets import mnist
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.optimizers import Adam

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
index_start = 0
OUTPUT_SIZE = 10
CELL_SIZE = 75
LR = 1e-3

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
 
X_train = X_train.reshape(-1,28,28)/255
X_test = X_test.reshape(-1,28,28)/255

Y_train = np_utils.to_categorical(Y_train,num_classes= 10)
Y_test = np_utils.to_categorical(Y_test,num_classes= 10)

inputs = Input(shape=[TIME_STEPS,INPUT_SIZE])

x = LSTM(CELL_SIZE, input_shape = (TIME_STEPS,INPUT_SIZE), return_sequences=False)(inputs)

x = Dense(OUTPUT_SIZE)(x)
x = Activation("softmax")(x)

model = Model(inputs,x)

adam = Adam(LR)
model.summary()
model.compile(loss = 'categorical_crossentropy',optimizer = adam,metrics = ['accuracy'])

for i in range(50000):
    X_batch = X_train[index_start:index_start + BATCH_SIZE,:,:]
    Y_batch = Y_train[index_start:index_start + BATCH_SIZE,:]
    index_start += BATCH_SIZE
    cost = model.train_on_batch(X_batch,Y_batch)
    if index_start >= X_train.shape[0]:
        index_start = 0
    if i%100 == 0:
        cost,accuracy = model.evaluate(X_test,Y_test,batch_size=50)
        print("accuracy:",accuracy)
