import cv2
import numpy as np
import datetime
import os
import glob
import copy
from collections import Counter
# 画像による色検知
class Name:
    # ファイル名、カメラ番号、サイズ、FPS、PNG圧縮率
    def __init__(self):
        date = datetime.datetime.now()
        # nameは拡張子なしファイル名 imagenameはpng形式ファイル名
        self.name = './image/pic' + date.strftime('%Y%m%d_%H%M%S.%f')
        self.imagename = self.name + ".png"
class RGB_Processing(Name):
    # 画像に対して前処理をするクラス
    def Zero(self,picture):
        #print(picture)
        # ゼロ埋めの画像配列
        picture = cv2.imread(picture)
        if len(picture.shape) == 3:
            height, width, channels = picture.shape[:3]
        else:
            height, width = picture.shape[:2]
        channels = 1
        zeros = np.zeros((height, width), picture.dtype)
        self.zero = zeros
        self.picture = picture

    def RGB_Separation(self):
        zero = self.zero
        pic = self.picture
        pic = cv2.blur(pic, (5, 5))
        # RGBごとに画像を分ける
        blue, green, red = cv2.split(pic)
        self.blue = cv2.merge((blue, zero, zero))
        self.green = cv2.merge((zero, green, zero))
        self.red = cv2.merge((zero, zero, red))
    def RGB_Gray(self):
        # グレースケール化をする
        self.blue_gray = cv2.cvtColor(self.blue, cv2.COLOR_BGR2GRAY)
        self.green_gray = cv2.cvtColor(self.green, cv2.COLOR_BGR2GRAY)
        self.red_gray = cv2.cvtColor(self.red, cv2.COLOR_BGR2GRAY)
    def RGB_Binary(self):
        # RGBごとに2値化
        # 大津の2値化を使う
        th_blue, bin_blue = cv2.threshold(self.blue_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_green, bin_green = cv2.threshold(self.green_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_red, bin_red = cv2.threshold(self.red_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('./TempPicture/blue_bin.png', bin_blue)
        cv2.imwrite('./TempPicture/green_bin.png', bin_green)
        cv2.imwrite('./TempPicture/red_bin.png', bin_red)
    def Color_Change(self):
        # RGBごとに色を変える(2値画像の白い部分はRGBごとの色になる)
        # ここの部分は3次元配列にするため(2値画像なのに3次元配列になる？後で変えるかも)
        self.blue_bin = cv2.imread('./TempPicture/blue_bin.png')
        self.green_bin = cv2.imread('./TempPicture/green_bin.png')
        self.red_bin = cv2.imread('./TempPicture/red_bin.png')
        # マスクする色の指定
        m_white = [255, 255, 255] # 白
        m_blue = [255, 0, 0]      # 青
        m_green = [0, 255, 0]     # 緑
        m_red = [0, 0, 255]       # 赤
        # 色を変える
        self.blue_bin[np.where((self.blue_bin == m_white).all(axis=2))] = m_blue
        self.green_bin[np.where((self.green_bin == m_white).all(axis=2))] = m_green
        self.red_bin[np.where((self.red_bin == m_white).all(axis=2))] = m_red

    def RGB_Composite(self, save):    # RGB_Compositeで1枚の画像に合成からエッジ化までをする
        # 分けていたRGBの画像を1つに合成する
        Composite = self.blue_bin + self.green_bin + self.red_bin
        # グレースケール化
        Composite_Gray = cv2.cvtColor(Composite, cv2.COLOR_BGR2GRAY)
        # 大津の2値化
        retval, Composite_Binary = cv2.threshold(Composite_Gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # 5*5のカーネル
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        # クロージング処理
        #Composite_Binary = cv2.morphologyEx(Composite_Binary, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(save, Composite_Binary)
        print(save)
def main():
    FileName = 'dataset.tar/dataset/characters/'
    # 補完した画像を置くパス
    SaveFolderPath = 'Dataset/characters/Binary_1/'
    os.mkdir(SaveFolderPath)
    # 補完するサイズ
    setsize = 64, 64
    path = glob.glob(FileName + '**')
    for folder in path:
        # 変換後のフォルダパス
        savefile = SaveFolderPath + os.path.basename(folder)
        # フォルダを作る
        os.mkdir(savefile)
        # 変換対象の画像を取得する
        ImagePath = glob.glob(folder + '/*.jpg')
        #print(savefile)
        for ImageFile in ImagePath:
            Exe = RGB_Processing()
            Exe.Zero(ImageFile)
            Exe.RGB_Separation()
            Exe.RGB_Gray()
            Exe.RGB_Binary()
            Exe.Color_Change()
            SaveImage = savefile + '/' + os.path.basename(ImageFile)
            Exe.RGB_Composite(SaveImage)
    return os.path.basename(__file__)
