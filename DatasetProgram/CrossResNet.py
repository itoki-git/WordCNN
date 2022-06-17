from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers
import pickle
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from CrossDataset import main
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
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
    lrate = 0.001
    if epoch > 10:
        lrate = 0.0008
    elif epoch > 30:
        lrate = 0.0005
    elif epoch > 50:
        lrate = 0.0001
    return lrate

with open('Dataset/Data/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f) 

classes = categories #分類するクラス
nb_classes = len(classes)
train_data_dir = './Dataset/Data/Normal/train/'
validation_data_dir = './Dataset/Data/Normal/test/'
img_width, img_height = 64, 64
batch_size = 32
fold_num = 5
FileName = 'Dataset/Data/CrossNormal300/Result/'

X_train, Y_train = main()
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train)
y_test = to_categorical(y_test, nb_classes)



input_tensor = Input(shape=(img_width, img_height, 3))

ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True)
cvscores = []

for train, test in kfold.split(x_train, y_train):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
    top_model.add(Dense(nb_classes, activation='softmax'))
    
    model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
    
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                metrics=['accuracy'])
    model.summary()
    modelCheckpoint = ModelCheckpoint(filepath = FileName + 'model_SGD_1.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='min',
                                    period=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    history = model.fit_generator(x_train[test], to_categorical(y_train[test], nb_classes),
                                epochs = 1000,
                                verbose = 1,
                                callbacks = [CSVLogger(FileName + 'model_SGD_1.csv'), modelCheckpoint])
                                #, LearningRateScheduler(lr_schedule), es_cb

    scores = model.evaluate(x_train[test], to_categorical(y_train[test], nb_classes), verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
