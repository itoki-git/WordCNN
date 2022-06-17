import numpy as np
from PIL import Image
from PIL import ImageOps
from keras.models import model_from_json
from keras import backend as K
import cv2

FileName = 'LSTMmodel/'
Name = 'LSTM_Adam_64_1'
json_string = open(FileName + 'CNNLSTM.json', 'r').read()
model = model_from_json(json_string)
model.load_weights(FileName+Name+'.h5')

# load image
images = np.empty([0, 64, 64], np.float32)
img_ori = Image.open('../alcon2019/dataset/Normal/test/U+304A/U+304A_100241706_00011_2_X0928_Y2639.jpg')
img_gray = img_ori.convert("RGB")

img_ary = np.asarray(img_gray)
img_ary = np.array(img_gray)
img_ary = 255 - img_ary
img_ary = img_ary.astype('float32') / 255.0
#images = np.append(images, [img_ary], axis=0)

images = img_ary.reshape((1,64, 64, 3))

# predict
ret = model.predict(images, 1, 1)
print(ret)

# output
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[0].output])
layer_output = get_1st_layer_output([images,])
print(layer_output[0].shape)
#np.save('output/convolution2d_out.npy', layer_output[0], allow_pickle=False)