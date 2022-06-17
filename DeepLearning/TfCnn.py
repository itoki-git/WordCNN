from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras import Model
import keras    
from keras.layers import Dense, Dropout, Activation, Flatten, Input, add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
import pickle
import numpy as np

def rescell(data, filters, kernel_size, option=False):
    strides=(1,1)
    if option:
        strides=(2,2)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    
    data=Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=add([x,data])
    x=Activation('relu')(x)
    return x

def ResNet(img_rows, img_cols, img_channels):
	input=Input(shape=(img_rows,img_cols,img_channels))
	x=Conv2D(64,(7,7), padding="same", input_shape=(64,64,3),activation="relu")(input)
	x=MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=Flatten()(x)
	x=Dense(units=10,kernel_initializer="he_normal",activation="softmax")(x)
	model=Model(inputs=input,outputs=[x])
	return model


# 入力画像の次元とチャンネル
img_rows, img_cols, img_channels = 64, 64, 3
num_classes = 48
epochs = 50

# The data, split between train and test sets:
with open('../alcon2019/dataset/Normal/LabelName.pkl', 'rb') as f:
        categories = pickle.load(f) 

classes = categories #分類するクラス
nb_classes = len(classes)
train_data_dir = '../alcon2019/dataset/Normal/train/'
validation_data_dir = '../alcon2019/dataset/Normal/test/'
img_width, img_height = 64, 64
batch_size = 64
FileName = '../alcon2019/dataset/Normal/Result/'
Name = 'model_SGD_1_2'
train_datagen=ImageDataGenerator(
    rotation_range=12,
    rescale = 1.0/255,
    shear_range = 0.3,
    zoom_range = 0.3,
    width_shift_range=0.3,
    height_shift_range=0.3)

validation_datagen=ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(img_width, img_height,3),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


validation_generator = validation_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(img_width, img_height,3),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)


model=ResNet(img_rows,img_cols,img_channels)
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = model.fit_generator(train_generator,
                    
					steps_per_epoch = 387666 / batch_size,
                    epochs=50,
                    verbose=1,
                    validation_data=validation_generator,
                    )