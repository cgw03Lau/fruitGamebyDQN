import sys

from PIL import Image
from PIL import ImageGrab
import pytesseract
import re
import pyautogui as pg
import cv2
import numpy
from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

#https://joyhong.tistory.com/79 여기서 확인
 # 200% 돋보기로 확대했을 때만 적용됨..


#Preprocessing
def screen_to_data():
    location = pg.locateOnScreen('FruitGame.png', confidence = 0.8) #searching location to screen
    if location is None :
        print("200% 확대된 상태에서 화면을 찍어주세요.")
        sys.exit()
    else :
        print(location)                                             # game screen location ( to use for dragging(pyautogui) )


    pg.screenshot('test1.png', region=location)                     # save image

    #Image Preprocessing
    img = cv2.imread('test1.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.inRange(img, (240, 240, 240),(255, 255, 255))
    img = 255 - img                                           #reverse

    #Image Bold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) #커널 (3, 3)이 중심점
    img = cv2.erode(img, kernel, iterations=2)   #침식(erode) - 어두운 영역 확대, 팽창(dilate) - 흰 영역 확대

    #Imgae Save
    #cv2.imwrite("test2.png", img)
    return img, location

import numpy as np


def load_img():
    img, screen_location = screen_to_data()

    #tesseract는 폰트가 12여야 잘 인식하기 때문에 이미지 크기 변환시킴
    img = cv2.resize(img, dsize=(420, 250), interpolation=cv2. INTER_AREA)
    #이미지를 글자로 읽어옴
    txt = pytesseract.image_to_string(img, lang = None, config='--psm 6 --oem 1 -c tessedit_chat_whitelist=0123456789')

    # print(txt) #확인

    temp = list(txt)
    lis = []
    for i in temp:
        if i <= '9' and i>='1':
            lis.append(int(i))

    lis = np.array(lis)
    # 처리한 숫자 및 이미지 확인
    #plt.imshow(img, cmap="gray"), plt.axis("off")  # 이미지 출력
    #plt.show()
    if(lis.size != 170):

        print("원소 인식 실패 다른 게임을 캡쳐해주세요.")
        print(np.array(lis).reshape(10,))
        sys.exit()

    return lis.reshape(10, 17), screen_location





