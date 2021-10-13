import numpy as np
import string

from Game import *


class ApplePy:
    def __init__(self, grid):
        self.grid = grid
        self.height, self.width = grid.shape
        self.n = 0
        self.score = 0; self.seq = []
        self.maxScore = 0; self.maxSeq = []
        self.visited = {}

    def dfs(self):
        if self.n > 5:
            return

        if self.dupCheck():
            return

        v = []
        for k in range(self.height * self.width):
            x, y = divmod(k, self.width)
            dx = x; dy = y

            s = 0

            while dx < self.height:
                s += self.grid[dx][y]
                dx += 1
                if dx < self.height and s + self.grid[dx][y] > 10:
                    break

            s = 0
            while dy < self.width:
                s += self.grid[x][dy]
                dy += 1
                if dy < self.width and s + self.grid[x][dy] > 10:
                    break

            cum = [[-1] * self.width for _ in range(self.height)]
            d = dy
            for i in range(x, dx):
                cum[i][y] = self.grid[i][y]
                for j in range(y+1, d):
                    cum[i][j] = cum[i][j-1] + self.grid[i][j]
                    if cum[i][j] >= 10:
                        d = j
                        break

            for j in range(y, dy):
                for i in range(x+1, dx):
                    if cum[i][j] == -1:
                        break
                    cum[i][j] = cum[i-1][j] + cum[i][j]

            for i in range(x, dx):
                for j in range(y, dy):
                    if cum[i][j] == 10:
                        if self.optAreaCheck(k, i*self.width+j):
                            v.append((k, i*self.width+j))

        if len(v) == 0:
            self.n += 1
            if self.maxScore < self.score:
                self.maxScore = self.score
                self.maxSeq = self.seq.copy()
        else:
            for a, b in v:
                ax, ay = divmod(a, self.width)
                bx, by = divmod(b, self.width)

                temp = self.grid[ax:bx+1,ay:by+1].copy()
                self.score += np.sum(temp[:,:]!=0)
                self.grid[ax:bx+1, ay:by+1] = 0

                self.seq.append((a, b))
                self.dfs()
                self.seq.pop()

                self.score -= np.sum(temp[:,:]!=0)
                self.grid[ax:bx+1, ay:by+1] = temp


    def dupCheck(self):
        key = ApplePy.convert(int(''.join(map(str, self.grid.reshape(-1)))), 100)
        if key in self.visited.keys():
            return True
        else:
            self.visited[key] = True
            return False

    def convert(n, base):
        c = string.digits + string.ascii_letters + "!@#$%^&*(){}[]ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"
        q, r = divmod(n, base)
        if q == 0:
            return c[r]
        else:
            return ApplePy.convert(q, base) + c[r]

    def optAreaCheck(self, a, b):
        ax, ay = divmod(a, self.width)
        bx, by = divmod(b, self.width)
        return any(self.grid[ax, ay:by+1]) and any(self.grid[bx, ay:by+1]) and any(self.grid[ax:bx+1, ay]) and any(self.grid[ax:bx+1, by])

    def run(self):
        self.dfs()

        list = []
        for a, b in self.maxSeq:
            ax, ay = np.array(divmod(a, self.width))
            bx, by = np.array(divmod(b, self.width))
            p1 = ax, ay
            p2 = bx, by
            list.append([p1, p2])
        return list