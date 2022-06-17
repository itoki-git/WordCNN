from PIL import Image
import glob
import os
import copy
def Search():
    # 途中経過をLINEに出したい場合、ここにimportを書く
    #from TellCompletionLINE import InterimReportLineNotification
    FileName = '../alcon2019/dataset/train_kana/'
    path = glob.glob(FileName + '**')
    WidthDict = {}
    HeightDict = {}
    SizeDict = {}
    count = 0
    imagecount = 0
    for folder in path:
        ImagePath = glob.glob(folder + '/*.jpg')
        for ImageFile in ImagePath:
            if imagecount == 1:
                count+=1
                image = Image.open(ImageFile)
                height, width = image.size
                WidthDict[ImageFile] = width
                HeightDict[ImageFile] = height
                SizeDict[ImageFile] = width * height
                MaxHeight = max(WidthDict.items(), key = lambda x: x[1])
                MaxWidth = max(HeightDict.items(), key = lambda y: y[1])
                MaxSize = max(SizeDict.items(), key = lambda z: z[1])
                imagecount=0
            else:
                imagecount+=1
            # 各フォルダの最大値
        print('************************' + str(os.path.basename(folder)) + '************************')
        print('Height:' + str(MaxHeight))
        print('Width:' + str(MaxWidth))
        print('size:' + str(MaxSize))
        # 途中経過のデータを送信
        #InterimReportLineNotification(MaxHeight, MaxWidth, MaxSize, os.path.basename(folder), os.path.basename(__file__))
        File = open('FileName.txt', 'a')
        File.write(str(os.path.basename(folder))+'\n')
        TextInput = open('MaxSize.txt', 'a')
        TextInput.write('***************************************' + str(os.path.basename(folder)) +
                        '***************************************\n' )
        TextInput.write('Height:' + str(MaxHeight) +'\n' + 'Width:' + str(MaxWidth) + '\n' + 'size:' + str(MaxSize) + '\n\n')


        HeightDict = {}
        WidthDict = {}
        SizeDict = {}
    # 全フォルダのなかでの最大値

    """
    print( 'AllHeightMax:' + str(AllHeightMax))
    TextInput.write('#############################################################################################\n\n' +
                    'AllHeightMax:' + str(AllHeightMax) + '\n')
    
    print( 'AllWidthMax:' + str(AllWidthMax))
    TextInput.write('#############################################################################################\n\n' +
                    'AllWidthMax:' + str(AllWidthMax) + '\n')
    
    print('AllSizeMax:' + str(max(AllSizeMax)))
    #TextInput.write('#############################################################################################\n\n' +
                    'AllSizeMax:' + str(AllSizeMax) + '\n')
    """
    File.close()
    TextInput.close()
    print(count)

def main():
    Search()
    #return os.path.basename(__file__)
main()