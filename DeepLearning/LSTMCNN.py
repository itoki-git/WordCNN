import numpy as np
from keras.utils import np_utils
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras import backend
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

with open('../alcon2019/dataset/ValiableLength/XTrain.pkl', "rb") as f:
    Xtrain= pickle.load(f)
#with open('../alcon2019/dataset/ValiableLength/XTest.pkl', "rb") as f:
 #   XTest= pickle.load(f)
#with open('../alcon2019/dataset/ValiableLength/YTrain.pkl', "rb") as f:
 #   Ytrain= pickle.load(f)
#with open('../alcon2019/dataset/ValiableLength/YTest.pkl', "rb") as f:
 #   YTest= pickle.load(f)
FileName = '../alcon2019/dataset/ValiableLength/'
Name = 'model_SGD_250_1'

n_in = 469
n_time = 1414
n_hidden = 128
n_out = 48*47*46
print(len(Xtrain))
print(Xtrain[-1][:][:].shape)
"""
model = Sequential()
model.add(Bidirectional(LSTM(n_hidden), input_shape=(n_time, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

model.summary()

epochs = 300
batch_size = 250

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
modelCheckpoint = ModelCheckpoint(filepath = FileName + Name + '.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  period=1)

hist = model.fit(Xtrain, Ytrain,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(Xtest, Ytest),
                 callbacks=[early_stopping, CSVLogger(FileName + Name + '.csv'), modelCheckpoint])
                """