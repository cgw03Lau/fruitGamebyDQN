import time

import pyautogui as pg

#https://blankspace-dev.tistory.com/416 여기 확인

from Data import *
from GameAlgorithm import *



#200% 확대된 상태에서의 화면
grid, screen_location =load_img()


x, y = screen_location[2] , screen_location[3]
screen_x, screen_y = screen_location[0] + 32, screen_location[1] + 51
algorithm = ApplePy(grid)
coordinate_list = algorithm.run()
list_size = len(coordinate_list)

radio_x = 100 #x / 17
radio_y = 100 #y / 10

for i in range(list_size):
    p1, p2 = coordinate_list[i]
    y1, x1 = p1
    y2, x2 = p2

    if p1[1] > p2[1]: # ← (드래그 방향)
        x1 = p2[1]
        x2 = p1[1]
    if p1[0] > p2[0]: # ↑ (드래그 방향)
        y1 = p2[0]
        y2 = p1[0]

    X1 = screen_x + (x1 * radio_x) -30
    Y1 = screen_y + (y1 * radio_y) -29

    X2 = screen_x + (x2 * radio_x) + radio_x + 33
    Y2 = screen_y + (y2 * radio_y) + radio_y + 28
    pg.moveTo(X1, Y1)
    speed = 0.3*((x1-x2)**2+(y1-y2)**2)**0.5 + 0.3
    pg.dragTo(X2, Y2, speed, button="left")
