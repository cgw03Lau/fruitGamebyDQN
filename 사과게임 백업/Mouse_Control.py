import pyautogui as pg

#https://blankspace-dev.tistory.com/416 여기 확인

#스크린 위치
screenWidth, screenHeight = pg.size()
print('{0}, {1}'.format(screenHeight, screenHeight))

#마우스 위치 반환
currentMouseX, currentMouseY = pg.position()
print('{0},{1}'.format(currentMouseX, currentMouseY))

#x,y 좌표가 스크린 안에 위치해있는지 확인
print(pg.onScreen(screenWidth, screenHeight))