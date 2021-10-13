import math
import random
from Data import *




#선택한 공간에 합이
#좌표는 0 ~ 11(y좌표)   0 ~ 18(x좌표)
def inputLocate():
    while(True):
        x1, y1 = input("첫번쨰 좌표[input ex:1 15] : ").split()
        x2, y2 = input("두번째 좌표[input ex:1 17] : ").split()
        if(is_X_Integer(x1) and is_Y_Integer(y1)):
            p1 = (int(x1)-1, int(y1)-1)
            print("좌표:{}".format(p1))
            if(is_X_Integer(x2) and is_Y_Integer(y2)):
                p2 = (int(x2)-1, int(y2)-1)
                print("좌표:{}".format(p2))
                break
        else:
            print("잘못된 좌표[ x좌표 범위 : 1~17, y좌표 범위 : 1~10")
            continue
    return p1, p2


def is_X_Integer(n):
    n = int(n)
    if not (n >= 1 and n <= 10):
        return False
    else:
        return True


def is_Y_Integer(n):
    n = int(n)
    if not (n >= 1 and n <= 17):
        return False
    else:
        return True


'''
# 만약 
    [ 0 ,  5 ,  4 ,  3 
      4 , 15 , 17 , 18 ] 
    가 있을 때 
    원소 17을 표현하고 싶다면  p1에 경우 (1, 2)   [y좌표 먼저 나옴] 
'''
#@return grid, reward
def action(p1, p2, grid):
    y1, y2, x1, x2 = p1[0],p2[0],p1[1],p2[1]
    if p1[0] > p2[0]: # ← (드래그 방향)
        y1 = p2[0]
        y2 = p1[0]
    if p1[1] > p2[1]: # ↑ (드래그 방향)
        x1 = p2[1]
        x2 = p1[1]
    # ↘ (드래그 방향이 이쪽으로 되도록 처리 끝난 상황)

    sumDrag = np.sum(grid[y1:y2 + 1, x1:x2 + 1])
    if np.sum(grid[y1:y2+1, x1:x2+1]) == 10 :
        reward = np.sum(grid[y1:y2+1, x1:x2+1] != 0) # 0이 아닌 공간의 개수를 reward로 줌
        grid[y1:y2+1, x1:x2+1] = 0 #빈공간으로 만듦
        return grid, reward
    #
    reward = -np.log((10 - sumDrag)**2) #
    return grid, reward

    #완전 다 끝내면..(170점이면..) (퍽!! 그런게 가능 할리가 없잖아!!)
    # if np.sum(grid) == 0:
    #     success = True
    #     return 0;

def generate_fix_sum_random_vec(limit, num_elem, tries=10):
    v = np.random.randint(1, 9, num_elem) #1~9 숫자 값만 가지도록
    s = sum(v)

    grid_a = np.round(v / s * limit)
    grid_b = np.floor(v / s * limit)
    grid_c = np.ceil(v / s * limit)

    if(np.sum(grid_a) == limit):
        if(np.sum(grid_a >= 10) == 0): #10이 있으면 다른 걸로
            return np.array(grid_a.astype(int)).reshape(10, 17)
        return generate_fix_sum_random_vec(limit, num_elem, tries-1)
    elif(np.sum(grid_b) == limit):
        if (np.sum(grid_b >= 10) == 0):
            return np.array(grid_b.astype(int)).reshape(10, 17)
        return generate_fix_sum_random_vec(limit, num_elem, tries - 1)
    elif(np.sum(grid_c) == limit):
        if (np.sum(grid_c >= 10) == 0):
            return np.array(grid_c.astype(int)).reshape(10, 17)
        return generate_fix_sum_random_vec(limit, num_elem, tries - 1)
    else:
        return generate_fix_sum_random_vec(limit, num_elem, tries-1)





def play(score):
    success = False
    score = 0
    grid = generate_fix_sum_random_vec(900, 170)

    # grid = load_img(False) #True면 기존에 있던 데이터(test2.png)로 False면 스크린샷 찍음
    # ㄴ 실제 게임에 적용할 때 active하기

    while(True):
        if(success == True):
            print("게임 클리어")
            return 0

        for rows in grid:
            for d in rows:
                print(d, end=' ')
            print()
        print("점수 : {} ".format(score))
        p1, p2 = inputLocate()
        grid, reward = action(p1, p2, grid)
        score += reward

    #score += action(inputLocate())


#play(0)
