import keras
import tensorflow as tf
import numpy as np
import pickle
from keras import optimizers
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, CuDNNLSTM, Input, Conv2D, MaxPooling2D, Flatten, RepeatVector, Reshape, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
class Train:
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
        self.Vocabrary.append('@')
        self.VocaNum = int((len(self.Voca) / 6) + 1)
        # LSTM parameters
        self.Dim = 800

    def EncoderModel(self, input, Dim):
        FineTuning = VGG16(include_top=False, weights='imagenet', input_tensor=input)
        x = Flatten()(FineTuning.output)  
        x = Dense(1024, activation='relu', name='fc1')(x)     
        x = Dropout(0.5, name='dropout1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='dropout2')(x)
        x = RepeatVector(n=5)(x)
        x1, ES_h1, ES_c1 = CuDNNLSTM(Dim, return_state=True,return_sequences=True, name='encoder1')(x)
        x2, ES_h2, ES_c2 = CuDNNLSTM(Dim, return_state=True, name='encoder2')(x1)
        b1, ES_bh1, ES_bc1 = CuDNNLSTM(Dim, return_state=True,return_sequences=True, name='encoder_b1')(x)
        b2, ES_bh2, ES_bc2 = CuDNNLSTM(Dim, return_state=True,return_sequences=True, name='encoder_b21')(b1)
        EncoderOutput = keras.layers.add([x2, b2], name='EncoderOutput')
        state_h_1=keras.layers.add([ES_h1,ES_bh1],name='state_h_1')
        state_c_1=keras.layers.add([ES_c1,ES_bc1],name='state_c_1')
        state_h_2=keras.layers.add([ES_h2,ES_bh2],name='state_h_2')
        state_c_2=keras.layers.add([ES_c2,ES_bc2],name='state_c_2')
        encoder_states1 = [state_h_1,state_c_1] 
        encoder_states2 = [state_h_2,state_c_2]
        #return EncoderOutput, state_h_1, state_c_1, state_h_2, state_c_2
        return encoder_states1, encoder_states2
    def DecoderModel(self, VocabraryNum, Dim):
        DecInput = Input(shape=(None, VocabraryNum), name='DecoderInput')
        DecLSTM1 = CuDNNLSTM(Dim, return_sequences=True, return_state=True, name='Decoder1')
        DecLSTM2 = CuDNNLSTM(Dim, return_sequences=True, return_state=True, name='Decoder2')
        DecDense = Dense(VocabraryNum, activation='softmax', name='DecoderOutput')
        return DecInput, DecLSTM1, DecLSTM2, DecDense
    # Training Model
    def ModelTrain(self):
        input = Input(shape=(self.Height, self.Width, 3), name='EncoderInput')
        ES1, ES2 = self.EncoderModel(input, self.Dim)
        DecInput, DecLSTM1, DecLSTM2, DecDense = self.DecoderModel(self.VocaNum, self.Dim)
        DecLSTMOutput1, _, _ = DecLSTM1(DecInput, initial_state=ES1)
        DecLSTMOutput2, _, _ = DecLSTM2(DecLSTMOutput1, initial_state=ES2)
        DecOutput = DecDense(DecLSTMOutput2)
        model = Model(inputs=[input, DecInput], outputs=[DecOutput])
        return model
    def Graph(self, count):
        # accracyのグラフ
        plt.plot(self.history.history['acc'],label="training")
        plt.plot(self.history.history['val_acc'],label="validation")
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(loc='upper right')
        plt.savefig("./"+self.FileName+ self.Name+"_"+"acc"+"_"+str(count)+".png")
        plt.close()
        # lossのグラフ
        plt.plot(self.history.history['loss'],label="training")
        plt.plot(self.history.history['val_loss'],label="validation")
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig("./"+self.FileName+ self.Name+"_"+"loss"+"_"+str(count)+".png")
        plt.close()
    def main(self):
        AllLoss = []
        AllValLoss = []
        AllAcc = []
        AllValAcc = []
        epoch = 30
        count = 0
        # データのロード
        print("データロード開始")
        with open("Dataset/TrainData.pkl", "rb") as f:
            DecoderInput, DecoderOutput, Images = pickle.load(f)
        with open("Dataset/PredictData.pkl", "rb") as f:
            DecoderInputTest, DecoderOutputTest, ImagesTest = pickle.load(f)
        self.FileName = "Result/CV/CV10/"
        self.Name = "model_1"    
        cvscores = []
        csvsoresTest = []
        kfold = KFold(n_splits=10)
        for train, test in kfold.split(Images, DecoderInput, DecoderOutput):
            model = self.ModelTrain()
            model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.SGD(momentum=0.9),
                        metrics=['accuracy'])
            modelCheckpoint = ModelCheckpoint(filepath = self.FileName + self.Name +"_"+str(count)+ '.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',
                                    period=1)
            es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
            model.summary()
            self.history = model.fit(
                            {'EncoderInput':Images[train], 'DecoderInput':DecoderInput[train]},
                            {'DecoderOutput':DecoderOutput[train]},
                            batch_size=64,
                            epochs= epoch,
                            verbose=1,
                            callbacks = [CSVLogger(self.FileName + self.Name +"_"+str(count)+  '.csv'), modelCheckpoint],
                            validation_data=({'EncoderInput':Images[test], 'DecoderInput':DecoderInput[test]},
                            {'DecoderOutput':DecoderOutput[test]})
                            )
            score = model.evaluate({'EncoderInput':Images[test], 'DecoderInput':DecoderInput[test]},{'DecoderOutput':DecoderOutput[test]}, verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
            cvscores.append(score[1] * 100)
            
            TestScore = model.evaluate({'EncoderInput':ImagesTest, 'DecoderInput':DecoderInputTest},{'DecoderOutput':DecoderOutputTest}, verbose=1)
            print("%s: %.2f%%" % (model.metrics_names[1], TestScore[1]*100))
            csvsoresTest.append(TestScore[1] * 100)
            self.Graph(count)
            count+=1
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        print("Test")
        print("%.2f%% (+/- %.2f%%)" % (np.mean(csvsoresTest), np.std(csvsoresTest)))
#Train().main()
