from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense
from keras import optimizers
import pickle
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt

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
    elif epoch > 30:
        lrate = 0.0005
    elif epoch > 50:
        lrate = 0.0001
    return lrate

with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f) 

classes = categories #分類するクラス
nb_classes = len(classes)
train_data_dir = '../alcon2019/dataset/Normal/train/'
validation_data_dir = '../alcon2019/dataset/Normal/test/'
img_width, img_height = 64, 64
batch_size = 512
FileName = '../alcon2019/dataset/Normal/Result/'
Name = 'model_SGD_512_2'
train_datagen=ImageDataGenerator(
    rotation_range=12,
    rescale = 1.0/255,
    shear_range = 0.3,
    zoom_range = 0.3,
    width_shift_range=0.3,
    height_shift_range=0.3)

validation_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)
input_tensor = Input(shape=(img_width, img_height, 3))

ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
 
top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
top_model.add(Dense(nb_classes, activation='softmax', name='output1'))

 
model = Model(input=ResNet50.input, output=top_model(ResNet50.output))
for i, layer in enumerate(model.layers):
    print(i, layer.name) 


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(momentum=0.9),
              metrics=['accuracy'])
model.summary()
modelCheckpoint = ModelCheckpoint(filepath = FileName + Name + '.h5',
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  period=1)
es_cb = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

history = model.fit_generator(train_generator,
                             epochs = 1000,
                             steps_per_epoch = 387666 / batch_size,
                             verbose = 1,
                             validation_data = validation_generator,
                             callbacks = [CSVLogger(FileName + Name + '.csv'), modelCheckpoint, LearningRateScheduler(lr_schedule), es_cb])
                             #, LearningRateScheduler(lr_schedule), es_cb
plot_history(history, FileName)