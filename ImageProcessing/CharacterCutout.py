import cv2
import numpy
import os
import copy
import glob
from RGBtoBinary import RGB_Processing

def BatchProcessing(filename, binpath):
    path = glob.glob(filename + '**')
    for folder in path:
        # 変換後のフォルダパス
        savefile = binpath + os.path.basename(folder)
        # フォルダ作成
        os.mkdir(savefile)
        # 変換対象の画像取得
        ImagePath = glob.glob(folder + '/*.jpg')
        for ImageFile in ImagePath:
            SaveImageFile = savefile + '/' + os.path.basename(ImageFile)
            image = CutoutProcessing(ImageFile, SaveImageFile)
            cv2.imwrite(SaveImageFile + '/' + os.path.basename(ImageFile), image)
        print('\n\n\n' + str(SaveImageFile) + 'に保存しました。\n\n\n' )
    print('終了')


def CutoutProcessing(image, savefile):

    # 画像が入ったフォルダのパス
    Cutout = RGB_Processing()
    Cutout.Zero(image)
    Cutout.RGB_Separation()
    Cutout.RGB_Gray()
    Cutout.RGB_Binary()
    Cutout.Color_Change()
    Cutout.RGB_Composite(savefile)
def main():
    FIleName = 'dataset.tar/dataset/characters/'
    # 2値化した画像を置くパス
    BinaryFolderPath = 'Dataset/characters/Binary/'
    os.mkdir(BinaryFolderPath)
    BatchProcessing(FIleName, BinaryFolderPath)