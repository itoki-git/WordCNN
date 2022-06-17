#from ConvertImage import ImageCompletion
from PIL import Image
import cv2
import numpy as np
import copy
import glob
import os

def red_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    return mask1 + mask2

def XMLInfo(ConvertPath, OriginalFolderName):
    path = glob.glob(ConvertPath + '**')
    for folder in  path:
        count = 0
        ImagePath = glob.glob(folder + '/*.jpg')
        for ImageFile in ImagePath:
            if count%30 == 0:
                image = cv2.imread(ImageFile)
                mask = red_detect(image)
                Find, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for i, contour in enumerate(contours):
                    # 小さい領域は排除する
                    area = cv2.contourArea(contour)
                    if area < 1000:
                        continue
                x, y, w, h = cv2.boundingRect(contour)
                Xmin = x
                Xmax = x + w
                Ymin = y
                Ymax = y + h
                #print(Xmin, Xmax, Ymin, Ymax)
                #image = cv2.rectangle(image, (Xmin, Ymin),(Xmax, Ymax), (255, 0, 0), thickness=3)
                
                #print(os.path.basename(folder), os.path.basename(ImageFile))
                FolderName = os.path.basename(folder)
                FileName = os.path.basename(ImageFile)
                XML(Xmin, Xmax, Ymin, Ymax, FolderName, FileName, OriginalFolderName)
                count = 0
            else:
                count+=1
                
            #return Xmin, Xmax, Ymin, Ymax, os.path.basename(folder), os.path.basename(ImageFile)
from xml.etree import ElementTree as ET
import xml.dom.minidom
def XML(xmin, xmax, ymin, ymax, FolderName, FileName, OriginalFolderName):
    XXX = "XXX"
    ImageRead = cv2.imread(OriginalFolderName+FolderName+'/'+FileName)
    print(ImageRead.shape)
    category = FolderName
    dom = xml.dom.minidom.Document()
    # Topannotation
    TopAnnotationTag = dom.appendChild(dom.createElement('annotaion'))
    # Topannotation/folder
    FolderTag = TopAnnotationTag.appendChild(dom.createElement('folder'))
    FolderTag.appendChild(dom.createTextNode(OriginalFolderName+FolderName))
    # Topannotation/filename
    FileNameTag = TopAnnotationTag.appendChild(dom.createElement('filename'))
    FileNameTag.appendChild(dom.createTextNode(FileName))
    # Topannotation/source
    SourceTag = TopAnnotationTag.appendChild(dom.createElement('source'))
    # Topannotation/source/database
    DatabeseTag = SourceTag.appendChild(dom.createElement('database'))
    DatabeseTag.appendChild(dom.createTextNode(XXX))
    # Topannotation/source/annotation
    AnnotationTag = SourceTag.appendChild(dom.createElement('annotation'))
    AnnotationTag.appendChild(dom.createTextNode(XXX))
    # Topannotation/source/image
    ImageTag = SourceTag.appendChild(dom.createElement('image'))
    ImageTag.appendChild(dom.createTextNode(XXX))
    # Topannotation/source/flickrid
    FlickridTag = SourceTag.appendChild(dom.createElement('flickrid'))
    FlickridTag.appendChild(dom.createTextNode(XXX))
    # Topannotation/owner
    OwnerTag = TopAnnotationTag.appendChild(dom.createElement('owner'))
    # Topannotation/owner/flickrid
    FlickridTag = OwnerTag.appendChild(dom.createElement('flickrid'))
    FlickridTag.appendChild(dom.createTextNode(XXX))
    # Topannotation/owner/name
    NameTag = OwnerTag.appendChild(dom.createElement('name'))
    NameTag.appendChild(dom.createTextNode('?'))
    # Topannotation/size
    SizeTag = TopAnnotationTag.appendChild(dom.createElement('size'))
    # Topannotation/size/width
    WidthTag = SizeTag.appendChild(dom.createElement('width'))
    WidthTag.appendChild(dom.createTextNode(str(ImageRead.shape[1])))
    # Topannotation/size/height
    HeightTag = SizeTag.appendChild(dom.createElement('height'))
    HeightTag.appendChild(dom.createTextNode(str(ImageRead.shape[0])))
    # Topannotation/size/depth
    DepthTag = SizeTag.appendChild(dom.createElement('depth'))
    DepthTag.appendChild(dom.createTextNode(str(ImageRead.shape[2])))
    # Topannotation/segmented
    SegmentedTag = TopAnnotationTag.appendChild(dom.createElement('segmented'))
    SegmentedTag.appendChild(dom.createTextNode("0"))
    # Topannotation/object
    ObjectTag = TopAnnotationTag.appendChild(dom.createElement('object'))
    # Topannotation/object/name
    NameTag = ObjectTag.appendChild(dom.createElement('name'))
    NameTag.appendChild(dom.createTextNode(category))
    # Topannotation/object/pose
    PoseTag = ObjectTag.appendChild(dom.createElement('pose'))
    PoseTag.appendChild(dom.createTextNode("Unspecified"))
    # Topannotation/object/truncated
    TruncatedTag = ObjectTag.appendChild(dom.createElement('truncated'))
    TruncatedTag.appendChild(dom.createTextNode("0"))
    #Topannotation/object/difficult
    DifficultTag = ObjectTag.appendChild(dom.createElement('difficult'))
    DifficultTag.appendChild(dom.createTextNode("1"))
    #Topannotation/object/bndbox
    BndboxTag = ObjectTag.appendChild(dom.createElement('bndbox'))
    #Topannotation/object/bndbox/xmin
    XminTag = BndboxTag.appendChild(dom.createElement('xmin'))
    XminTag.appendChild(dom.createTextNode(str(xmin)))
    #Topannotation/object/bndbox/ymin
    YminTag = BndboxTag.appendChild(dom.createElement('ymin'))
    YminTag.appendChild(dom.createTextNode(str(ymin)))
    #Topannotation/object/bndbox/xmax
    XmaxTag = BndboxTag.appendChild(dom.createElement('xmax'))
    XmaxTag.appendChild(dom.createTextNode(str(xmax)))
    #Topannotation/object/bndbox/ymax
    YmaxTag = BndboxTag.appendChild(dom.createElement('ymax'))
    YmaxTag.appendChild(dom.createTextNode(str(ymax)))
    save, ext = os.path.splitext(FileName)
    with open('SSD/Annotation/'+save+'.xml', mode='w', encoding='utf-8') as f:
        f.write(dom.toprettyxml())
    
def main():
    OriginalFolderName = 'Dataset/characters/Normal300/'
    ConvertName = 'Dataset/characters/NormalRed300/'
    XMLInfo(ConvertName, OriginalFolderName)
    

main()