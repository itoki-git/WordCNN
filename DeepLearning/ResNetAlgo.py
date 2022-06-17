
import pandas as pd
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import pickle
import os
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from keras.callbacks import EarlyStopping
import gc

def csv():
    csvName = '../alcon2019/dataset/train/annotations.csv'
    one = pd.read_csv(csvName, usecols=[1])
    two = pd.read_csv(csvName, usecols=[2])
    three = pd.read_csv(csvName, usecols=[3])
    ID = pd.read_csv(csvName, usecols=[0])
    Unicode1Label = []
    Unicode2Label = []
    Unicode3Label = []
    Label = np.zeros(46, dtype=np.float32)
    length = len(one)
    print(length)
    ImageData = []
    #print(ID)
    #Unicode1Label = one.Unicode1[count]
    #Unicode2Label = two.Unicode2[count]
    #Unicode3Label = three.Unicode3[count]
    for count in range(length):
        #print(count)
        Name = '../alcon2019/dataset/train/Normal64/'+ str(count)+ '.jpg'
        Unicode1Label = one.Unicode1[count]
        Unicode2Label = two.Unicode2[count]
        Unicode3Label = three.Unicode3[count]
        ImageData.append([Name, Unicode1Label, Unicode2Label, Unicode3Label])
    #print(ImageData[0][1])
    return ImageData

def LabelDeside(categories):
    if categories == "U+304A":
        label = 0 
    elif categories == "U+304B":
        label = 1
    elif categories == "U+304D":
        label = 2
    elif categories == "U+304F":
        label = 3
    elif categories == "U+305B":
        label = 4
    elif categories == "U+305D":
        label = 5
    elif categories == "U+305F":
        label = 6
    elif categories == "U+306A":
        label = 7
    elif categories == "U+306B":
        label = 8
    elif categories == "U+306C":
        label = 9
    elif categories == "U+306D":
        label = 10
    elif categories == "U+306E":
        label = 11
    elif categories == "U+306F":
        label = 12
    elif categories == "U+307B":
        label = 13
    elif categories == "U+307E":
        label = 14
    elif categories == "U+307F":
        label = 15
    elif categories == "U+308A":
        label = 16
    elif categories == "U+308B":
        label = 17
    elif categories == "U+308C":
        label = 18
    elif categories == "U+308D":
        label = 19
    elif categories == "U+308F":
        label = 20
    elif categories == "U+3042":
        label = 21
    elif categories == "U+3044":
        label = 22
    elif categories == "U+3046":
        label = 23
    elif categories == "U+3048":
        label = 24
    elif categories == "U+3051":
        label = 25
    elif categories == "U+3053":
        label = 26
    elif categories == "U+3055":
        label = 27
    elif categories == "U+3057":
        label = 28
    elif categories == "U+3059":
        label = 29
    elif categories == "U+3061":
        label = 30
    elif categories == "U+3064":
        label = 31
    elif categories == "U+3066":
        label = 32
    elif categories == "U+3068":
        label = 33
    elif categories == "U+3072":
        label = 34
    elif categories == "U+3075":
        label = 35
    elif categories == "U+3078":
        label = 36
    elif categories == "U+3080":
        label = 37
    elif categories == "U+3081":
        label = 38
    elif categories == "U+3082":
        label = 39
    elif categories == "U+3084":
        label = 40
    elif categories == "U+3086":
        label = 41
    elif categories == "U+3088":
        label = 42
    elif categories == "U+3089":
        label = 43
    elif categories == "U+3090":
        label = 44
    elif categories == "U+3091":
        label = 45
    elif categories == "U+3092":
        label = 46
    elif categories == "U+3093":
        label = 47
    return label

def DataSplit():
    Data = csv()
    print(type(Data))
    with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f)
    Label = []
    SavePredictImage = "../alcon2019/dataset/ValiableLength/PredictImages64_150/"
    df = pd.DataFrame(columns=["Image", "Label"])
    for count, label in enumerate(categories):
        Label.append(count)
    print(Label)
    TestPicture = random.sample(Data, 150)
    print(type(TestPicture))
    #Data = list(set(Data) - set(TestPicture))
    Data=[x for x in Data if x not in TestPicture]
    #TrainData, TestData = train_test_split(Data, test_size = 0.2, shuffle = False)
    #print(TrainData[1][1])
    for PredictPicture in (TestPicture):
        print(os.path.basename(PredictPicture[0]))
        Predict = cv2.imread(PredictPicture[0], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(SavePredictImage + "1_" + os.path.basename(PredictPicture[0]), Predict)
    return Data #TrainData, TestData
def CreateData(ImageData):
    X = []
    Y = []
    Label = []
    OutLabel = np.zeros(48, dtype=np.float32)
    df = pd.DataFrame(columns=["Image", "Label"])
    print("ラベル付け")
    for count, Image in enumerate(ImageData):
        if count % 1000 == 0:
            print(count)
        #print(Image[0])
        OutLabel = np.zeros(48, dtype=np.float32)
        OutImage = []
        OutImage = img_to_array(load_img(Image[0], grayscale=True))
        #OutImage = cv2.imread(Image[0])
        for i in range(3):
            if  i == 0:
                category = LabelDeside(Image[1])
                #print(category)
            elif i == 1:
                category = LabelDeside(Image[2])
                #print(category)
            else:
                category = LabelDeside(Image[3])
                #print(category)
            Label.append(category)
            OutLabel[category] = 1.0
        FileName = os.path.basename((Image[0]))
        Name, a = os.path.splitext(FileName)
        save_img('../alcon2019/dataset/ValiableLength/Data150/'+Name+'_'+os.path.basename((Image[0])), OutImage)
        File = open('../alcon2019/dataset/ValiableLength/FileName3Word_150.txt', 'a')
        File.write(str(Name) + ' ' + str(Image[1]+Image[2]+Image[3]) + '\n')
        #X.append(OutImage)
        #Y.append(OutLabel)
        #print(OutLabel)
        #df.loc[count] = [OutImage, OutLabel]
        #print(OutLabel)
        #print(df)
    return X,Y #df
def CreateDataset():
    Data = DataSplit()
    Image, Label = CreateData(Data)
    """
    X = np.asarray(Image)
    X = X / 255.0
    X_train, Y_train, X_test, Y_test = train_test_split(X, Label, test_size=0.2, random_state=111)
    print("Dump!")

    XY = (X_train, Y_train, X_test, Y_test)
    with open('../alcon2019/dataset/ValiableLength/Dataset100.pkl', "wb") as f:
        pickle.dump(XY, f, protocol=-1)
    """
    """
    np.savez('../alcon2019/dataset/ValiableLength/DatasetX', X_train, X_test)
    del X_train
    del X_test
    gc.collect()
    np.savez('../alcon2019/dataset/ValiableLength/DatasetY', Y_train, Y_test)
    del Y_train
    del Y_test
    gc.collect()
    """


    #train, test = DataSplit()
    #DfTrain = CreateData(train)
    #DfTest = CreateData(test)
    #DfTrain.to_msgpack("../alcon2019/dataset/train/multilabel_train.pd", compress="zlib")
    #DfTest.to_msgpack("../alcon2019/dataset/train/multilabel_test.pd", compress="zlib")
CreateDataset()
"""
with open('../alcon2019/dataset/ValiableLength/Dataset.pkl', "rb") as f:
    a,b,c,d = pickle.load(f)
print(len(a))
print(len(b))
print(len(c))
print(len(d))
"""
#df_read = pd.read_msgpack("../alcon2019/dataset/train/multilabel_train.pd")
#print(df_read)
from keras.layers import Conv2D, BatchNormalization, Activation, Input, AveragePooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import History
from keras.optimizers import SGD
import keras.backend as K

import pandas as pd
import numpy as np
import pickle

# モデル
def create_block(input, ch, reps):
    x = input
    for i in range(reps):
        x = Conv2D(ch, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def create_network():
    input = Input((64,64,3))
    x = create_block(input, 32, 3)
    x = AveragePooling2D(2)(x)
    x = create_block(x, 64, 3)
    x = AveragePooling2D(2)(x)
    x = create_block(x, 128, 3)
    x = AveragePooling2D(2)(x)
    x = create_block(x, 256, 3)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="sigmoid")(x)

    return Model(input, x)

# ジェネレーター
def generator(df_path, batch_size):
    df = pd.read_msgpack(df_path)
    df = df.values
    while True:
        img_cahce, label_cache = [], []
        indices = np.arange(df.shape[0])
        np.random.shuffle(indices)
        for i in indices:
            img_cahce.append(df[i, 0])
            label_cache.append(df[i, 1])
            if len(img_cahce) == batch_size:
                X_batch = np.asarray(img_cahce, np.float32) / 255.0
                y_batch = np.asarray(label_cache, np.float32)
                img_cahce, label_cache = [], []
                yield X_batch, y_batch

# 損失関数
def categorical_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def binary_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.sum(bce, axis=-1)

# 評価関数
def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

def train():
    FileName = "../alcon2019/dataset/train/model.h5"
    model = create_network()
    
    model.compile(SGD(0.01, 0.9), loss=binary_loss, metrics=[total_acc, binary_acc])
    

    hist = History()
    batch_size = 64
    model.summary()

    modelCheckpoint = ModelCheckpoint(filepath = FileName,
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  period=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

    model.fit_generator(generator("../alcon2019/dataset/train/multilabel_train.pd", batch_size), steps_per_epoch=95996//batch_size,
                        validation_data=generator("../alcon2019/dataset/train/multilabel_test.pd", batch_size), validation_steps=24000//batch_size,
                         epochs=1)
 # callbacks=[hist],
    history = hist.history


#train()