import  numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from TellCompletionLINE import InterimReportLineNotification
import matplotlib.pyplot as plt
class History(Callback):
    def __init__(self):
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

        self.train_acc_append = []
        self.train_loss_append = []
        self.val_acc_append = []
        self.val_loss_append = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_acc = (logs['acc'])
        self.val_acc = (logs['val_acc'])
        self.train_loss = (logs['loss'])
        self.val_loss = (logs['val_loss'])

        self.train_acc_append.append(logs['acc'])
        self.val_acc_append.append(logs['val_acc'])
        self.train_loss_append.append(logs['loss'])
        self.val_loss_append.append(logs['val_loss'])
        epoch+=1
        if (epoch % 1) == 0:
            # グラフ絵画(accuracy)
            plt.figure(num=1, clear=True)
            plt.subplots_adjust(hspace=0.5)
            plt.subplot(2,1,1)
            plt.title('accuracy')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.plot(range(epoch), self.train_acc_append, marker='.', label='acc', color='deepskyblue')
            plt.plot(range(epoch), self.val_acc_append, marker='.', label='val_acc', color='red')
            plt.legend(loc='best', fontsize=10)
            plt.grid()
            # グラフ絵画(loss)
            plt.subplot(2, 1, 2)
            plt.title('Loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.plot(range(epoch), self.train_loss_append, marker='.', label='loss', color='green')
            plt.plot(range(epoch), self.val_loss_append, marker='.', label='val_loss', color='orange')
            plt.legend(loc='best', fontsize=10)
            plt.grid()
            plt.savefig('Dataset/PickleData/Data.jpg')

            InterimReportLineNotification(round(self.train_acc, 4), round(self.val_acc, 4)
                                          , round(self.train_loss, 4), round(self.val_loss, 4), epoch)