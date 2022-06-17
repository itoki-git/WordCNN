import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import glob, os
from PIL import Image
import cv2
import random

TrainData = []
TestData = []
TestPicture = []
Data = []

FolderPath = '../alcon2019/dataset/train/Normal64'
Trecount = 0
Testcount = 0
path = glob.glob(FolderPath + '**')
for folder in path:
    FolderName = os.path.basename(folder)
    ImagePath = glob.glob(folder + '/*.jpg')
    SaveTrainPath = "../alcon2019/dataset/3Normal/train/" + os.path.basename(folder) + '/'
    SaveTestPath = "../alcon2019/dataset/3Normal/test/" + os.path.basename(folder) + '/'
    SaveTestPicture = "../alcon2019/dataset/3Normal/PredictImages/" + os.path.basename(folder) + '/'
    print(os.path.basename(folder))
    os.mkdir(SaveTrainPath)
    os.mkdir(SaveTestPath)
    os.mkdir(SaveTestPicture)
    for ImageFile in ImagePath:
        Data.append(ImageFile)
    TestPicture = random.sample(Data, 10)
    Data = list(set(Data) - set(TestPicture))
    TrainData, TestData = train_test_split(Data, train_size = 0.8, shuffle = True)
    #print(TrainData)
    print(os.path.basename(folder) + "__Train")
    for train in Data:
        #print(SaveTrainPath + os.path.basename(train))
        TrainImages = cv2.imread(train, 0)
        cv2.imwrite(SaveTrainPath + os.path.basename(train), TrainImages)
        Trecount+=1
    print(os.path.basename(folder) + "__Test")
    
    for test in TestData:
        TestImages = cv2.imread(test, 0)
        cv2.imwrite(SaveTestPath + os.path.basename(test), TestImages)
        Testcount+=1
    
    print(os.path.basename(folder) + "__Predict")
    for PredictPicture in TestPicture:
        Predict = cv2.imread(PredictPicture, 0)
        cv2.imwrite(SaveTestPicture + os.path.basename(PredictPicture), Predict)
    TrainData = []
    TestData = []
    TestPicture = []
    Data = []
print(Trecount)
print(Testcount)