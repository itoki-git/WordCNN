from PIL import Image
import os, glob
import random, math
import pickle
import random
import keras
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

# 分類対象のカテゴリを選ぶ
def main():
    ImageData = []
    LabelData = []
    LabelName = []
    SaveData = []
    imagecount = 0
    count = 0
    categories = []
    nb_classes = len(categories)
    FolderPath = '../alcon2019/dataset/Normal64_gray/'
    path = glob.glob(FolderPath + '**')
    for index, folder in enumerate(path):
        FolderName = os.path.basename(folder)
        categories.append(FolderName)
        ImagePath = glob.glob(folder + '/*.jpg')
        savefile = "Dataset\PickleData/Normal/TestPicture_2/" + os.path.basename(folder)
        #os.mkdir(savefile)
        """
        for ImageFile in ImagePath:
            if imagecount == 43:
                #print(imagecount)
                #SaveData.append(ImageFile)
                imagecount = 0
            
            else:
                #print(imagecount)
                image = Image.open(ImageFile)
                image = image.convert("RGB")
                data = np.asarray(image)
                ImageData.append(data)
                LabelData.append(index)
                imagecount+=1
                count += 1
        print(count)
        print(len(SaveData))
        SaveData = random.sample(SaveData, 10)
        for saveimg in range(10):
            
            img = Image.open(SaveData[saveimg])
            #img.save(savefile + '/' + os.path.basename(SaveData[saveimg]))
        print("テスト画像保存")
        SaveData = []
        print(folder)
        
    print(count)
    ImageData = np.array(ImageData)
    LabelData = np.array(LabelData)
    ImageData = ImageData.astype('float32')
    ImageData = ImageData / 255.0
    #print(ImageData)
    LabelData = np_utils.to_categorical(LabelData, 46)
    #Data = ImageData, LabelData
    print("Dump!")
    """
    """
    with open('Dataset/PickleData/Normal/Dataset_2.pkl', "wb") as f:
        pickle.dump(ImageData, f, protocol=4)
    with  open('Dataset/PickleData/Normal/DatasetLabel_2.pkl', "wb") as f:
        pickle.dump(LabelData, f)
    """
    with  open('../alcon2019/dataset/Normal/LabelName.pkl', "wb") as f:
        pickle.dump(categories, f)
    return os.path.basename(__file__)
main()


