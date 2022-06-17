from PIL import Image
import glob
import os
import numpy as np
import cv2
# リサイズするサイズ
def Resize(ImagePath, ResizedImagePath):
    size = 32, 32
    # 返還対象のフォルダのパス
    FileName = ImagePath
    path = glob.glob(FileName + '**')
    for folder in path:
        # 変換後のフォルダのパス
        savefile = ResizedImagePath + os.path.basename(folder)
        # フォルダを作成する
        os.mkdir(savefile)
        # 変換対象の画像を取得
        ImagePath = glob.glob(folder + '/*.jpg')
        for ImageFile in ImagePath:
            image = Image.open(ImageFile).convert('L')
            # リサイズする
            image = image.resize(size, Image.LANCZOS)
            print(str(ImageFile) + 'を' + str(size) + 'にリサイズしました。')
            # 保存
            image.save(savefile + '/' + os.path.basename(ImageFile), quality=100)
        print(str(savefile) + 'に保存しました。')
    print('終了')
def main():
    FileName = '../Dataset/characters/Progress32/'
    ResizedImageFolder = '../Dataset/characters/Normal32/'
    os.mkdir(ResizedImageFolder)
    Resize(FileName, ResizedImageFolder)
    return os.path.basename(__file__)
main()