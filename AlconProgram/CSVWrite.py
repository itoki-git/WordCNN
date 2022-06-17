from BiLSTM import Train 
from CreateDataset import CreateData
import os
import numpy as np
import cv2
import glob
import pickle
import copy
from Prob import Proba
from tqdm import tqdm
import csv
from keras.models import Sequential, Model, load_model
from keras.layers import Input

class CSVWrite(Train):
    def __init__(self):
        super().__init__()
        self.Width = 100
        self.Height = 100
        self.ColorCh = 3
        # モデル
        self.Weight0 = "./Result/CV1/model_1_0.h5"
        self.Weight1 = "./Result/CV1/model_1_1.h5"
        self.Weight2 = "./Result/CV1/model_1_2.h5"
        self.Weight3 = "./Result/CV1/model_1_3.h5"
        self.Weight4 = "./Result/CV1/model_1_4.h5"
        # 画像ディレクトリ
        self.AlconDir = "./AlconTest/"
        self.NameLength = 3 

        self.EOS = "@"
        self.Train = Train()
        Train.__init__(self)
    def TestEncoderModel(self, Dim):
        input = Input(shape=(self.Height, self.Width, self.ColorCh), name='EncoderInput')
        encoder_states1, encoder_states2 = self.Train.EncoderModel(input,Dim)
        ES = encoder_states1+encoder_states2
        EncoderModel1 = Model(inputs=input,outputs=ES)
        return EncoderModel1
    
    def TestDecoderModel(self, Dim):
        DecInput, DecLSTM1, DecLSTM2, DecDense = self.DecoderModel(self.VocaNum, Dim)
        DSI_h1 = Input(shape=(Dim, ))
        DSI_c1 = Input(shape=(Dim, ))
        DSI_h2 = Input(shape=(Dim, ))
        DSI_c2 = Input(shape=(Dim, ))
        DecOutput1, DS_h1, DS_c1 = DecLSTM1(DecInput, initial_state=[DSI_h1, DSI_c1])
        DecOutput2, DS_h2, DS_c2 = DecLSTM2(DecOutput1, initial_state=[DSI_h2, DSI_c2])
        DecOutput = DecDense(DecOutput2)
        DecInputStates = [DSI_h1, DSI_c1, DSI_h2, DSI_c2]
        DecOutputStates = [DS_h1, DS_c1, DS_h2, DS_c2]
        ModelIn = [DecInput] + DecInputStates
        ModelOut = [DecOutput] + DecOutputStates
        DecoderModel = Model(ModelIn, ModelOut)
        return DecoderModel
    def LSTM(self, ImageFile, NameLength):
        OutName = ""

        input = np.zeros((1, 1, self.VocaNum), dtype=np.float32)
        input[0, 0, self.Vocabrary.index(self.EOS)] = 1

        TestImage = cv2.imread(ImageFile)
        TestImage = np.array(TestImage)
        TestImage = TestImage.astype('float32')
        TestImage = TestImage / 255.0
        TestImage = TestImage.reshape((1, self.Height, self.Width, self.ColorCh))
        Name1 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        Name2 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        Name3 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        # 隠れマルコフ　
        ProbaName1 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        ProbaName2 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        ProbaName3 = np.zeros((1,1,self.VocaNum), dtype=np.float32)
        Output1 = []
        Output2 = []
        Output3 = []
        for count in range(5):
            EncoderModel, DecoderModel = self.LoadModel(count)
            States = EncoderModel.predict(TestImage)
            for i in range(3):
                Output = DecoderModel.predict([input] + States)
                y = Output[0]
                y = y[0, 0]
                States = Output[1:]
                PredWord = y
                if i==0:
                    Name1 = PredWord+Name1
                elif i == 1:
                    Name2 = PredWord+Name2
                else:
                    Name3 = PredWord+Name3
        Name1 = np.delete(Name1, 48)
        Name2 = np.delete(Name2, 48)
        Name3 = np.delete(Name3, 48)

        
        Output1 = self.Vocabrary[Name1.argmax()]
        score2 = Proba(Name2, self.Vocabrary, Output1)
        Output2 = self.Vocabrary[score2.argmax()]
        score3 = Proba(Name3, self.Vocabrary, Output2)
        Output3 = self.Vocabrary[score3.argmax()]
        return Output1, Output2, Output3
    def LoadModel(self, i):
        # 0
        if self.j == 0:
            print("LoadModel:"+str(i))
            if i == 0:
                Dim = 128
                self.EncoderModel0 = self.TestEncoderModel(Dim)
                self.EncoderModel0.load_weights(self.Weight0, by_name=True)
                self.DecoderModel0 = self.TestDecoderModel(Dim)
                self.DecoderModel0.load_weights(self.Weight0, by_name=True)
                return self.EncoderModel0, self.DecoderModel0
            # 1
            elif i == 1:
                Dim = 128
                self.EncoderModel1 = self.TestEncoderModel(Dim)
                self.EncoderModel1.load_weights(self.Weight1, by_name=True)
                self.DecoderModel1 = self.TestDecoderModel(Dim)
                self.DecoderModel1.load_weights(self.Weight1, by_name=True)
                return self.EncoderModel1, self.DecoderModel1
            # 2
            elif i == 2:
                Dim = 128
                self.EncoderModel2 = self.TestEncoderModel(Dim)
                self.EncoderModel2.load_weights(self.Weight2, by_name=True)
                self.DecoderModel2 = self.TestDecoderModel(Dim)
                self.DecoderModel2.load_weights(self.Weight2, by_name=True)
                return self.EncoderModel2, self.DecoderModel2
            # 3
            elif i == 3:
                Dim = 128
                self.EncoderModel3 = self.TestEncoderModel(Dim)
                self.EncoderModel3.load_weights(self.Weight3, by_name=True)
                self.DecoderModel3 = self.TestDecoderModel(Dim)
                self.DecoderModel3.load_weights(self.Weight3, by_name=True)
                return self.EncoderModel3, self.DecoderModel3
            # 4
            elif i == 4:
                Dim = 128
                self.EncoderModel4 = self.TestEncoderModel(Dim)
                self.EncoderModel4.load_weights(self.Weight4, by_name=True)
                self.DecoderModel4 = self.TestDecoderModel(Dim)
                self.DecoderModel4.load_weights(self.Weight4, by_name=True)
                self.j +=1
                return self.EncoderModel4, self.DecoderModel4            
        else:
            if i == 0:
                return self.EncoderModel0, self.DecoderModel0
            # 1
            elif i == 1:
                return self.EncoderModel1, self.DecoderModel1
            # 2
            elif i == 2:
                return self.EncoderModel2, self.DecoderModel2
            # 3
            elif i == 3:
                return self.EncoderModel3, self.DecoderModel3
            # 4
            elif i == 4:
                return self.EncoderModel4, self.DecoderModel4

    def main(self):            
        Path = glob.glob(self.AlconDir + "*")
        WriteResult = []
        SortFile = []
        PreResult = []
        self.j = 0
        for ImageFile in Path:
            FileName = os.path.basename(os.path.splitext(ImageFile)[0])
            SortFile.append(int(FileName))    
        SortName = sorted(SortFile)
        for ImageName in tqdm(SortName):
            Output = []
            ImageFile = self.AlconDir+str(ImageName)+'.jpg'
            Output1, Output2, Output3= self.LSTM(ImageFile, 3)
            Output = [str(ImageName), Output1, Output2, Output3]
            PreResult.append(Output)
        with open("./test_prediction.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(PreResult)  
#CSVWrite().main()