from PIL import Image
import glob
import os
import cv2
import numpy as np
# フォルダ内一括処理
def BatchProcessing(filename, convertpath, size):
    path = glob.glob(filename + '**')
    for folder in path:
        # 変換後のフォルダパス
        savefile = convertpath + os.path.basename(folder)
        # フォルダを作る
        os.mkdir(savefile)
        # 変換対象の画像を取得する
        ImagePath = glob.glob(folder + '/*.jpg')
        for ImageFile in ImagePath:
            image = ImageCompletion(ImageFile, size)
            print(str(ImageFile) + 'を' + str(size) + 'に補完しました。')
            # 保存
            image = image.convert('L')
            image.save(savefile + '/' + os.path.basename(ImageFile), quality = 100)
        print(str(savefile) + 'に保存しました')
    print('終了')
# 画像の足りない領域を補完する
def ImageCompletion(ConvertImage, setsize):
    # 単色の画像を生成し、その上に変換対象画像を貼るイメージ
    #setWidth, setHeight = setsize
    color = 255, 255, 255 # 白
    size = 64, 64
    image = Image.open(ConvertImage).convert('L')
    #print(image)
    width, height = image.size[0], image.size[1]

    if (width > height) and (width > size[1]):
        setsize = width, width
        setWidth, setHeight = setsize
    elif (height > width) and (height > size[1]):
        setsize = height, height
        setWidth, setHeight = setsize
    elif height == width:
        setsize = height, height
        setWidth, setHeight = setsize
    else:
        setWidth, setHeight = size
    # 領域を決定する計算
    setWidthA = setWidth / 2
    setHeightA = setHeight / 2
    setHalfWidth = setWidthA - (width / 2)
    setHalfHeight = setHeightA - (height / 2)
    new_width = width + setHalfWidth * 2
    new_height = height + setHalfHeight * 2
    #image = np.asarray(image)
    #image = cv2.rectangle(image, (1, 1),(width-1, height-1), (255, 0, 0), thickness=3)
    # 単色画像を生成
    #image = Image.fromarray(image)
    result = Image.new('RGB', (int(new_width), int(new_height)), color)
    # 対象画像を貼る
    result.paste(image, (int(setHalfWidth), int(setHalfHeight)))
    result = result.resize(size, Image.LANCZOS)
    #result.save("Dataset/Data/Level2/データテスト.jpg")
    return result

def main():
    # 補完する画像のパス(サブフォルダの中身も全部変える)
    FileName = '../alcon2019/dataset/train_kana/'
    # 補完した画像を置くパス
    ConvertFolderPath = '../alcon2019/dataset/Normal64/'
    os.mkdir(ConvertFolderPath)
    # 補完するサイズ
    setsize = 64, 64
    Execution = BatchProcessing(FileName, ConvertFolderPath, setsize)
    # プログラム名を返す
    ImageCompletion(ConvertFolderPath, setsize)
    return os.path.basename(__file__)
main()