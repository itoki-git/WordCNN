import os
import keras
import numpy as np
import random
import cv2
import glob
import pickle

class CreateData:
    def __init__(self):
        self.Width = 100
        self.Height = 100
        # 別の書き方を後で考える
        self.Voca = "U+304AU+304BU+304DU+304FU+305BU+305DU+305FU+306AU+306BU+306CU+306DU+306EU+306FU+307BU+307EU+307FU+308AU+308BU+308CU+308DU+308FU+3042U+3044U+3046U+3048U+3051U+3053U+3055U+3057U+3059U+3061U+3064U+3066U+3068U+3072U+3075U+3078U+3080U+3081U+3082U+3084U+3086U+3088U+3089U+3090U+3091U+3092U+3093@"
        self.Vocabrary = []
        count = 0
        word = 0
        for i, char in enumerate(self.Voca):
            count += 1
            if count % 6 == 0:
                self.Vocabrary.append(self.Voca[word:i+1])
                word+=6
                count = 0
        self.CMVocabrary = self.Vocabrary
        self.Vocabrary.append('@')
        #print(self.Vocabrary)
        self.VocaNum = int((len(self.Voca) / 6) + 1)
        self.TrainDir = "./Dataset/WordPredict_100/"
        # Data Path
        self.VocabraryPath = "./Dataset/word.txt"
        self.FilePath = "./Dataset/PredictName_100.txt"
        
        self.NameLength = 3 
        self.EOS = "@"
        self.NameLength += 2
       
    def DataLoad(self, NameLength, VovabraryNum):
        imgs = np.zeros((None, self.Height, self.Width, 1), dtype=np.float32)
        DecoderInput = np.zeros((None, NameLength, self.VovabraryNum), dtype=np.float32)
        DecoderOutput = np.zeros_like(DecoderInput) 

    def GetOnehot(self, ImageName):
        GetVec = np.zeros((self.NameLength, self.VocaNum), dtype=np.float32)
        print(self.NameLength)
        FileName = os.path.basename(ImageName)
        WordID, _ = FileName.split('_')
        Name = self.GetID()[WordID]
        GetVec[0, self.Vocabrary.index(self.EOS)] = 1.
        print(GetVec)
        count = 0
        word = 0
        WordCount = 0
        chars = ""
        for i, char in enumerate(Name):
            count += 1
            if count%6 == 0:
                chars = Name[word:i+1]
                print(count/6)
                GetVec[int(count/6), self.Vocabrary.index(chars)] = 1.
                chars = ""
                word+=6
        GetVec[int((count/6)+1), self.Vocabrary.index(self.EOS)] = 1.
        return GetVec

    def GetID(self):
        id = {}
        with open(self.FilePath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                WordID, Name = line.split(' ')
                id[WordID] = Name
        return id
            
    def TrainData(self):
        # Train Directory Path
        Path = glob.glob(self.TrainDir + "*")
        # 全データを入れる変数の初期化
        AllDecoderInput = []
        AllDecoderOutput = []
        AllImages = []
        for count, ImageFile in enumerate(Path):
            Image = cv2.imread(ImageFile)
            AllImages.append(Image)
            DecoderInput = self.GetOnehot(ImageFile)
            print(DecoderInput)
            DecoderOutput = np.zeros_like(DecoderInput)
            DecoderOutput[0:self.NameLength-1] = DecoderInput[1:]
            AllDecoderInput.append(DecoderInput)
            AllDecoderOutput.append(DecoderOutput)
            if count % 1000 == 0:
                if count == 0:
                    continue
                print(count)
                print(len(AllImages))
                print(len(AllDecoderInput))
                print(len(AllDecoderOutput))

        # list型をndarrayへ
        AllDecoderInput = np.array(AllDecoderInput)
        AllDecoderOutput = np.array(AllDecoderOutput)
        # 画像の正規化
        AllImages = np.array(AllImages)
        AllImages = AllImages.astype('float32')
        AllImages = AllImages / 255.0
        return AllDecoderInput, AllDecoderOutput, AllImages
    def main(self):
        DecoderInput, DecoderOutput, Images = self.TrainData()
        print("Dump")
        Pickle = DecoderInput, DecoderOutput, Images
        with open("Dataset.pkl", "wb") as f:
            pickle.dump(Pickle, f, protocol=-1)