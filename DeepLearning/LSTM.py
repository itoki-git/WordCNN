from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import InputLayer, Flatten, Dense
from keras.layers.recurrent import LSTM
from keras import optimizers
import pickle
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import BatchNormalization
import numpy as np

def plot_history(history, Name):
    #%matplotlib inline
    plt.plot(history.history['acc'], "_", label = "accuracy")
    plt.plot(history.history['val_acc'], "_", label = "val_acc")
    plt.title('model acuuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig(Name + 'Accuracy.png')

    plt.plot(history.history['loss'], "_", label = "loss")
    plt.plot(history.history['val_loss'], "_", label = "val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.savefig(Name + 'Loss.png')

def lr_schedule(epoch):
    lrate = 0.01
    if epoch > 10:
        lrate = 0.005
    elif epoch > 30:
        lrate = 0.001
    elif epoch > 50:
        lrate = 0.0001
    return lrate

with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f) 

classes = categories #分類するクラス
nb_classes = len(classes)
img_width, img_height = 64, 64
batch_size = 128
FileName = '../alcon2019/dataset/ValiableLength/'
Name = 'LSTM'

with open('../alcon2019/dataset/ValiableLength/Dataset64.pkl', "rb") as f:
    X_train, Y_train, X_test, Y_test= pickle.load(f)
X_train = np.reshape(X_train, (-1, 64, 64))
Y_train = np.reshape(Y_train, (-1, 64, 64))
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test[1])
model = Sequential()

model.add(
    InputLayer(input_shape=(64,64))
)

weight_decay = 1e-4

model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
#model.add(BatchNormalization())
model.add(LSTM(units=256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)))
#model.add(BatchNormalization())
model.add(LSTM(units=256, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(units=48, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
modelCheckpoint = ModelCheckpoint(filepath = FileName + Name + '.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  period=1)
es_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

history = model.fit(X_train ,X_test,
                    epochs = 1,
                    verbose = 1,
                    batch_size=128,
                    validation_data = (Y_train, Y_test),
                    callbacks = [CSVLogger(FileName + Name + '.csv'), LearningRateScheduler(lr_schedule), modelCheckpoint, es_cb])
                    #, LearningRateScheduler(lr_schedule), es_cb        
model.save('test.h5')
plot_history(history, FileName)
