import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
img = image.load_img("../alcon2019/dataset/Normal/train/U+304A/U+304A_100241706_00008_1_X2023_Y1382.jpg")

# PIL形式をnumpyのndarray形式に変換

x = image.img_to_array(img)

# (height, width, 3) -> (1, height, width, 3)

x = x.reshape((1,) + x.shape)

datagen = ImageDataGenerator(

           rotation_range=0,

           width_shift_range=0,

           height_shift_range=0,

           shear_range=0,

           zoom_range=0,

           horizontal_flip=False,

           vertical_flip=False)
max_img_num = 16

imgs = []

for d in datagen.flow(x, batch_size=1):

    # このあと画像を表示するためにndarrayをPIL形式に変換して保存する

    imgs.append(image.array_to_img(d[0], scale=True))

    # datagen.flowは無限ループするため必要な枚数取得できたらループを抜ける

    if (len(imgs) % max_img_num) == 0:

        break
print(image.img_to_array(d[0]).shape)