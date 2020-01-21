from keras.layers import Input,Activation,Dense
from keras.layers.recurrent import LSTM
from keras.models import Model

TIME_STEPS = 5
INPUT_SIZE = 10
CELL_SIZE = 3

inputs = Input(shape=[TIME_STEPS,INPUT_SIZE])

x = LSTM(CELL_SIZE, input_shape = (TIME_STEPS,INPUT_SIZE), return_sequences=True)(inputs)

model = Model(inputs,x)

model.summary()