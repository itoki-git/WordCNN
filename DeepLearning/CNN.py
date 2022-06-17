# coding:utf-8
import datetime
import keras
from keras.utils import np_utils
import random
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.optimizers import Adam
import pickle
from keras.callbacks import CSVLogger
import os
from matplotlib.pyplot import plot as plt
from keras.callbacks import ModelCheckpoint
import cv2
import glob
import copy

def ModelPredict():
    with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f)

    model = load_model('../alcon2019/dataset/ValiableLength/LSTM_Adam_64_2.h5')
    label = np.array(categories)
    print(label)
    imagecount = 0
    ImageChoise = []
    image = []
    FalseImage =[]
    FalseLabel = []
    FalseData = []
    FolderPath = '../alcon2019/dataset/ValiableLength/PredictImages64/'
    F = 0
    T = 0
    path = glob.glob(FolderPath + '**')
    for index, folder in enumerate(path):
        #if index == 11:
        FolderName = os.path.basename(folder)
        categories.append(FolderName)
        ImagePath = glob.glob(folder + '/*.jpg')

        for ImageFile in ImagePath:
            #ImageChoise.append(ImageFile)
            image.append(ImageFile)
            imagecount = 0

        #ImageChoise = []

        #else:
           # continue
    image =  random.sample(image, 40)
    print(image)
    for count, PredictImage in enumerate(image):
        
        img = Image.open(PredictImage)
        img = img.convert("RGB")
        temp_img_array = np.asarray(img)
        temp_img_array = np.array(temp_img_array)
        temp_img_array = temp_img_array.astype('float32') / 255.0
        temp_img_array =  temp_img_array.reshape((1, 64, 64, 3))
        ImagePred = model.predict(temp_img_array)
        TopScore = np.max(ImagePred) * 100
        TopLabel = categories[np.argmax(ImagePred)]
        Guess = str(count) + '___' + os.path.basename(os.path.splitext(PredictImage)[0])
        print(Guess)
        print(TopScore)
        print(TopLabel)
        print('****************************************************')
        #img.save('Dataset/Data/Normal/PredictResultPicture/' + str(count) + '___' + os.path.basename(os.path.splitext(PredictImage)[0]) + '.jpg')
        if TopLabel == os.path.basename(os.path.dirname(PredictImage)):
            T+=1
        else:
            FalseImage.append(Guess)
            FalseLabel.append(TopLabel)
            F+=1
    print(F)
    print(FalseLabel)
    print(FalseImage)
    #print(len(image) / F)
    #print(F / T)
    
def ROI(Xmin, Xmax, Ymin, Ymax, PredictImage, scan):
    setsize = 64, 64
    size = 64, 64
    RoiImage = PredictImage[Ymin:Ymax, Xmin:Xmax]
    #print(RoiImage.shape)
    RoiImage = Image.fromarray(RoiImage)
    #print(RoiImage.size)
    color = 255, 255, 255 # 白
    width, height = RoiImage.size[0], RoiImage.size[1]

    if (width > height) and (width > setsize[1]):
        setsize = width, width
        setWidth, setHeight = setsize
    elif (height > width) and (height > setsize[1]):
        setsize = height, height
        setWidth, setHeight = setsize
    elif height == width:
        setsize = height, height
        setWidth, setHeight = setsize
    else:
        setWidth, setHeight = setsize

    setWidth, setHeight = setsize
    setWidthA = setWidth / 2
    setHeightA = setHeight / 2
    setHalfWidth = setWidthA - (width / 2)
    setHalfHeight = setHeightA - (height / 2)
    new_width = width + setHalfWidth * 2
    new_height = height + setHalfHeight * 2
    result = Image.new('RGB', (int(new_width), int(new_height)), color)
    result.paste(RoiImage, (int(setHalfWidth), int(setHalfHeight)))
    result = result.resize(size, Image.LANCZOS)
    #print(result.size)
    scan = scan+1
    return result, scan, Ymax
def Processing(PredictImage, model):
    with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f)
    label = np.array(categories)
    img = PredictImage
    img = img.convert('RGB')
    temp_img_array = np.asarray(img)
    temp_img_array = np.array(temp_img_array)
    #temp_img_array = temp_img_array.astype('float32') / 255.0
    temp_img_array =  temp_img_array.reshape((1, 64, 64, 3))
    ImagePred = model.predict(temp_img_array)
    TopScore = np.max(ImagePred) * 100
    print(TopScore)

    TopLabel = categories[np.argmax(ImagePred)]
    return TopScore, TopLabel
def single():
    model = load_model('../Dataset/Data/Normal64/Result/model_SGD_64_2.h5')
    TrainData = cv2.imread("../Dataset/Data/Level2/Data.jpg", 0)
    #print(TrainData.shape)
    # 注目領域ROI初期値
    Xmin = 0
    Xmax = TrainData.shape[1]
    Ymin = 0
    Ymax = 0
    ScanSize = TrainData.shape[0]
    scan = 0
    piccount = 0
    while True:
        if ScanSize >= 0:
            count = 10
            PredictImage, scan, Yaxis = ROI(Xmin, Xmax, Ymin, Ymax+count, TrainData, scan)
            #print(PredictImage.size)
            cv2.imshow("image",np.asarray(PredictImage))
            cv2.waitKey(0)
            Score, Word = Processing(PredictImage, model)
            #print("scan"+str(scan))
            #print("score"+ str(Score))

            if Score > 80:
                piccount+=1
                print(Word)
                print(Score)
                Ymin = Yaxis
                PredictImage.save("../Dataset/Data/Normal64/SavePicture/"+str(piccount)+"_"+Word+".jpg")
            else:
                pass
            Ymax +=count
            ScanSize = TrainData.shape[0] - Yaxis
            print(ScanSize)
        else:
            break
def Predict():
    with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f)
    model = load_model('test.h5')
    print(model)
    PredictImage = "../alcon2019/dataset/ValiableLength/PredictImages64/1473.jpg"
    print(categories)
    label = np.array(categories)
    img = Image.open(PredictImage)
    img = img.convert("RGB")
    temp_img_array = np.asarray(img)
    temp_img_array = np.array(temp_img_array)
    temp_img_array = temp_img_array.astype('float32') / 255.0
    #temp_img_array =  temp_img_array.reshape((1, 64, 64, 3))
    ImagePred = model.predict(temp_img_array)
    TopScore = np.max(ImagePred) * 100
    TopLabel = categories[np.argmax(ImagePred)]
    #Guess = str(count) + '___' + os.path.basename(os.path.splitext(PredictImage)[0])
    #print(Guess)
    print(TopScore)
    print(TopLabel)
#main()
#ModelPredict()
#single()
Predict()