from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers
import pickle
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

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
    elif epoch > 20:
        lrate = 0.0005
    elif epoch > 25:
        lrate = 0.0001
    return lrate

with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
    categories = pickle.load(f) 

classes = categories #分類するクラス
nb_classes = len(classes)
print(nb_classes)
train_data_dir = '../alcon2019/dataset/CrossNormal/train/'
#validation_data_dir = '../Dataset/Data/Normal/test/'
img_width, img_height = 64, 64
batch_size = 64
FileName = '../alcon2019/dataset/CrossNormal/Result/'
Name = "model_Adam_Cross128-1"
fold_num = 5 
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True)
cvscores = []
lossscore = []
accAll = []
lossAll = []
X = []
Y = []
for itr in range(len(classes)):
    ImageFileName = os.listdir(train_data_dir + classes[itr])
    print(classes[itr]+"の画像枚数は"+str(len(ImageFileName)))
    for j in range(0, len(ImageFileName)):
        n = os.path.join(train_data_dir + classes[itr] + '/', ImageFileName[j])
        Image = cv2.imread(n)
        X.append(Image)
        Y.append(itr)
Y = np.array(Y)
ImageCount = len(X)

input_tensor = Input(shape=(img_width, img_height, 3))
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
count = 0
for train, test in kfold.split(X, Y):
    train_count = len(train)
    X_array = copy.deepcopy(X)
    Y = np.array(Y[0:ImageCount]).tolist()
    train = np.concatenate([train,np.arange(ImageCount,len(Y))], axis=0)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
    top_model.add(Dense(nb_classes, activation='softmax'))
    
    model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['categorical_accuracy'])

    Y = np.array(Y)
    X_array = np.array(X_array)
    Y_train = Y[train].tolist()
    Y_train = to_categorical(Y_train)
    Y_test = Y[test].tolist()
    Y_test = to_categorical(Y_test)

    #model.summary()
    modelCheckpoint = ModelCheckpoint(filepath = FileName + Name + '-' + str(count) +'.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',
                                    period=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

    history = model.fit(X_array[train], Y_train,
                                batch_size = batch_size,
                                epochs = 30,
                                verbose = 1,
                                validation_data=(X_array[test], Y_test),
                                callbacks = [CSVLogger(FileName + Name + '-' + str(count) + '.csv'), modelCheckpoint, es_cb])
                                #, LearningRateScheduler(lr_schedule), es_cb
    scores = model.evaluate(X_array[test], Y_test, verbose=0)
    print(scores[0])
    print(scores[1])
    lossAll.append(scores[0])
    accAll.append(scores[1])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    count+=1
    #acc, val_accのプロット
print(lossAll)
print(accAll)
print(np.mean(lossAll))
print(np.mean(cvscores))
plt.subplot(5,1,len(cvscores))
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.title("accuracy"+str(len(cvscores)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
plt.show()

plt.subplot(5,1,len(cvscores))
plt.plot(history.history["loss"], label="loss", ls="-")
plt.plot(history.history["val_loss"], label="val_loss", ls="-")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.title("loss"+str(len(lossAll)))
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(lossAll), np.std(lossAll)))
#plt.show()