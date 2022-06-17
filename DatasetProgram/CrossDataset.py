import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import glob, os
from PIL import Image
import cv2
import random
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
import pickle
from keras.preprocessing.image import load_img,img_to_array 
from keras.utils import np_utils
def main():
    TrainData = []
    TestPicture = []
    Data = []
    temp_img_array_list = [] 
    temp_label_array_list = [] 
    FolderPath = '../alcon2019/dataset/Normal64/'
    Trecount = 0
    Testcount = 0
    path = glob.glob(FolderPath + '**')

    with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f) 


    for folder in path:
        FolderName = os.path.basename(folder)
        ImagePath = glob.glob(folder + '/*.jpg')
        SaveTrainPath = "../alcon2019/dataset/CrossNormal/train/" + os.path.basename(folder) + '/'
        SaveTestPicture = "../alcon2019/dataset/CrossNormal/PredictImages/" + os.path.basename(folder) + '/'
        #print(os.path.basename(folder))
        os.mkdir(SaveTrainPath)
        os.mkdir(SaveTestPicture)
        for ImageFile in ImagePath:
            Data.append(ImageFile)
        TestPicture = random.sample(Data, 20)
        Data = list(set(Data) - set(TestPicture))
        
        print(os.path.basename(folder) + "__Train")
        for train in Data:
            TrainImages = cv2.imread(train)
            temp_img_array = np.asarray(TrainImages)
            temp_img_array_list.append(temp_img_array)
            temp_label_array_list.append(os.path.basename(folder))
            cv2.imwrite(SaveTrainPath + os.path.basename(train), TrainImages)
            Trecount+=1
        print(os.path.basename(folder) + "__Predict")
        for PredictPicture in TestPicture:
            print(PredictPicture)
            Predict = cv2.imread(PredictPicture)
            cv2.imwrite(SaveTestPicture + os.path.basename(PredictPicture), Predict)
        
        TrainData = []
        TestData = []
        TestPicture = []
        Data = []
    """
    X_train = np.array(temp_img_array_list)
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0
    Y_train = np_utils.to_categorical(temp_label_array_list, 46)
    print(Trecount)
    return X_train, Y_train
    """
main()